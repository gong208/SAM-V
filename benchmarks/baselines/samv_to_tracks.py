#!/usr/bin/env python3
"""
Verification utility (not part of the baseline pipeline).

Converts an existing SAM-V benchmark run into the baseline ``tracks.json``
interchange format, so the run can be re-scored through
``score_baseline.py``.  If the round-trip does not reproduce the published
SAM-V numbers exactly, the scoring adapter is wrong and no baseline number it
produces is trustworthy -- this is the gate described in the plan's
Verification section.

Reads ``<run_dir>/<scene>/best_run/annotations_with_masks.json`` (a list of
annotation dicts whose ``per_view_masks`` are already RLE in exactly the
encoding ``decode_mask`` consumes) and ``<run_dir>/<scene>/summary.json`` (for
``frame_names``, an independent source from the benchmark scene directory, so
the frame-alignment assert in ``score_baseline.py`` is actually exercised).

Usage (project env)::

    python benchmarks/baselines/samv_to_tracks.py \
        --run_dir results/3dtracking_finetune_eval/epoch0020_scannetpp \
        --out_dir results/baselines/_roundtrip/scannetpp/preds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines.track_io import make_infer_seconds, write_tracks  # noqa: E402


def convert_scene(scene_run_dir: Path, out_dir: Path) -> Path:
    summary_path = scene_run_dir / "summary.json"
    ann_path = scene_run_dir / "best_run" / "annotations_with_masks.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    with summary_path.open() as f:
        summary = json.load(f)
    with ann_path.open() as f:
        annotations = json.load(f)

    tracks = []
    for ann in annotations:
        # per_view_masks are already RLE dicts; make_track passes dicts through
        # untouched, so no re-encoding happens and the bytes are preserved.
        tracks.append({
            "score": float(ann.get("predicted_iou", 0.0)),
            "per_view_masks": ann["per_view_masks"],
            "stability_score": float(ann.get("stability_score", 0.0)),
            "source_view": int(ann.get("source_view", -1)),
        })

    return write_tracks(
        out_dir / summary["scene_id"] / "tracks.json",
        scene_id=summary["scene_id"],
        frame_names=summary["frame_names"],
        tracks=tracks,
        method="sam_v_roundtrip",
        config={
            "source_run": str(scene_run_dir),
            "best_run_name": summary.get("best_run", {}).get("run_name")
            if isinstance(summary.get("best_run"), dict) else summary.get("best_run"),
            "prompt_source": summary.get("prompt_source"),
            "prompt_sampling_method": summary.get("prompt_sampling_method"),
            "prompt_points_per_mask": summary.get("prompt_points_per_mask"),
            "ignore_instance_ids": summary.get("ignore_instance_ids", []),
        },
        target_size=summary.get("target_size", (1024, 1024)),
        infer_seconds=make_infer_seconds(),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    args = p.parse_args()

    scene_dirs = sorted(
        d for d in args.run_dir.iterdir()
        if d.is_dir() and (d / "summary.json").is_file()
    )
    if not scene_dirs:
        raise SystemExit(f"no scene run directories under {args.run_dir}")

    for d in scene_dirs:
        out = convert_scene(d, args.out_dir)
        print(f"  {d.name} -> {out}")
    print(f"converted {len(scene_dirs)} scene(s)")


if __name__ == "__main__":
    main()
