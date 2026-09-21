#!/usr/bin/env python3
"""
Stage C: the single scoring path for every baseline config.

Reads one ``tracks.json`` per scene (the stage-B/stage-C interchange format,
see ``track_io.py``) and scores it with the *frozen* metric code from
``benchmarks/sam_vggt_3dtracking_benchmark.py`` -- ``load_scene_inputs``,
``evaluate_scene`` and ``write_dataset_summary`` are imported and called
unmodified, so baseline numbers are produced by exactly the code that produced
the SAM-V numbers (CLAUDE.md §7: eval code is frozen).

Output schema matches a SAM-V run's: ``dataset_summary.json`` +
``precision_recall_summary.csv`` at the top, one ``summary.json`` per scene.

Usage (project env)::

    export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
    python benchmarks/baselines/score_baseline.py \
        --preds_dir results/baselines/panst3r_v2_scannetpp/preds \
        --benchmark_root "$BENCH/scannetpp" \
        --output_dir results/baselines/panst3r_v2_scannetpp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.sam_vggt_3dtracking_benchmark import (  # noqa: E402
    TARGET_SIZE,
    discover_scene_dirs,
    evaluate_scene,
    load_scene_inputs,
    write_dataset_summary,
)
from utils.provenance import write_run_provenance  # noqa: E402
from benchmarks.baselines.track_io import (  # noqa: E402
    read_tracks,
    total_infer_seconds,
    tracks_to_annotations,
)


class FrameAlignmentError(RuntimeError):
    """The predicted frame list does not match the loaded scene's frame list."""


def assert_frame_names_match(
    scene_id: str,
    predicted: Sequence[str],
    loaded: Sequence[str],
) -> None:
    """
    Hard failure, never a warning: a silent misalignment here produces
    plausible but wrong numbers.
    """
    if list(predicted) != list(loaded):
        raise FrameAlignmentError(
            f"{scene_id}: frame_names mismatch between tracks.json and the benchmark scene.\n"
            f"  tracks.json ({len(predicted)}): {list(predicted)}\n"
            f"  scene       ({len(loaded)}): {list(loaded)}"
        )


def score_scene(
    scene_dir: Path,
    tracks_path: Path,
    ignore_instance_ids: Sequence[int],
    target_size: Sequence[int] = TARGET_SIZE,
) -> Dict[str, Any]:
    """Score one scene, returning the dict ``write_dataset_summary`` consumes."""
    payload = read_tracks(tracks_path)
    scene_inputs = load_scene_inputs(scene_dir, tuple(target_size))

    assert_frame_names_match(scene_dir.name, payload["frame_names"], scene_inputs.frame_names)

    annotations = tracks_to_annotations(payload)
    metrics = evaluate_scene(annotations, scene_inputs, ignore_instance_ids)

    # Mirrors sam_vggt_3dtracking_benchmark.py:873 and :1350-1363 -- no metric
    # code is reimplemented here, only the assembly of the summary dict.
    per_match_pooled_ious = [iou for _, _, iou in metrics["matches"]]
    num_gt = int(metrics["num_gt_objects"])

    return {
        "scene_id": scene_dir.name,
        "best_t_sr": float(metrics["t_sr"]),
        "best_t_sr_at_half": float(metrics["t_sr_at_half"]),
        "best_t_miou": float(metrics["t_miou"]),
        "mean_pooled_iou": (
            float(sum(per_match_pooled_ious) / num_gt) if num_gt > 0 else math.nan
        ),
        "best_run_name": payload.get("method", "baseline"),
        "num_gt_objects": num_gt,
        "num_matches": len(metrics["matches"]),
        "num_predictions": int(metrics["num_predictions"]),
        "per_match_pooled_ious": per_match_pooled_ious,
        "precision_recall": metrics.get("precision_recall", {}),
        "per_object_metrics": metrics["per_object_metrics"],
        "method": payload.get("method", "baseline"),
        "infer_seconds": dict(payload.get("infer_seconds", {})),
        "total_infer_seconds": total_infer_seconds(payload.get("infer_seconds")),
        "config": payload.get("config", {}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True, type=Path,
                   help="directory holding <scene_id>/tracks.json")
    p.add_argument("--benchmark_root", required=True, type=Path,
                   help="e.g. .../3DTrackingBenchmark/scannetpp")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--scene_ids", nargs="*", default=None,
                   help="restrict to these scenes (default: every scene with a tracks.json)")
    p.add_argument("--ignore_instance_ids", nargs="*", type=int, default=[],
                   help="GT instance ids to exclude (SAM-V runs used none)")
    p.add_argument("--tracks_name", default="tracks.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_dirs = discover_scene_dirs(args.benchmark_root, args.scene_ids)
    if not scene_dirs:
        raise SystemExit(f"no scenes discovered under {args.benchmark_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(args.output_dir, config=vars(args) | {
        "preds_dir": str(args.preds_dir),
        "benchmark_root": str(args.benchmark_root),
        "output_dir": str(args.output_dir),
        "target_size": list(TARGET_SIZE),
    })

    dataset_results: List[Dict[str, Any]] = []
    missing: List[str] = []
    for scene_dir in scene_dirs:
        tracks_path = args.preds_dir / scene_dir.name / args.tracks_name
        if not tracks_path.is_file():
            missing.append(scene_dir.name)
            continue
        result = score_scene(scene_dir, tracks_path, args.ignore_instance_ids)
        scene_out = args.output_dir / scene_dir.name
        scene_out.mkdir(parents=True, exist_ok=True)
        with (scene_out / "summary.json").open("w") as f:
            json.dump(result, f, indent=2)
        dataset_results.append(result)
        print(
            f"[{result['scene_id']}] T-mIoU={result['best_t_miou']:.4f} "
            f"pooled={result['mean_pooled_iou']:.4f} "
            f"matches={result['num_matches']}/{result['num_gt_objects']} GT, "
            f"{result['num_predictions']} preds, "
            f"{result['total_infer_seconds']:.1f}s"
        )

    if missing:
        print(f"WARNING: no {args.tracks_name} for {len(missing)} scene(s): {missing}", file=sys.stderr)
    if not dataset_results:
        raise SystemExit("no scenes scored")

    write_dataset_summary(dataset_results, args.output_dir)

    with (args.output_dir / "dataset_summary.json").open() as f:
        summary = json.load(f)
    pr = summary["mean_precision_recall_by_threshold"].get("0.5", {})
    print("\n=== dataset summary ===")
    print(f"  scenes           : {summary['num_scenes']}  ({summary['total_gt_objects']} GT objects)")
    print(f"  T-mIoU           : {summary['mean_best_t_miou']:.4f}")
    print(f"  O-IoU (pooled)   : {summary['mean_pooled_iou']:.4f}")
    print(f"  P@0.5 / R@0.5    : {pr.get('mean_precision', float('nan')):.4f} / "
          f"{pr.get('mean_recall', float('nan')):.4f}")
    total_time = sum(r["total_infer_seconds"] for r in dataset_results)
    print(f"  infer_seconds    : {total_time:.1f}s total, "
          f"{total_time / len(dataset_results):.1f}s/scene")
    print(f"  written to       : {args.output_dir}")


if __name__ == "__main__":
    main()
