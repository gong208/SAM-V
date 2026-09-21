#!/usr/bin/env python3
"""
Stage B for ODIN: dumped geometry + benchmark RGB -> Q x V masks -> ``tracks.json``.

Runs in the ODIN env (``$ODIN_ENV``), and installs nothing.  Plan:
``notes/plans/odin-baseline.md`` §4 Phase 3; contract: ``benchmarks/baselines/README.md``.

The three-stage shape is Point-SAM's, not PanSt3R's, because ODIN needs geometry it
cannot produce itself::

    stage A  (project env)  RGB -> VGGT -> scaled, gravity-aligned geometry
                            odin_geometry_probe.py --mode dump  ->  <scene>/geometry.npz
    stage B  (odin env)     geometry -> Q x V masks -> preds/<scene>/tracks.json   [this file]
    stage C  (project env)  score_baseline.py, frozen                              [unchanged]

Everything about the forward pass -- the config ``OPTS``, the reimplemented eval path
(``ODIN.forward`` is never called, plan §6 item 5), the geometry injection as
``multi_scale_xyz``, and the network/stub instrumentation -- is *imported* from
``odin_probe_forward.py`` rather than copied, so Phase 2's kill-switch and Phase 3's row
cannot drift apart.

What this file adds on top of that is the contract, and four decisions:

1. **Query -> track is max-over-class per query** (plan §6 item 6, §7), not upstream's
   literal ``topk(100)`` over the flattened ``Q x C``.  Upstream's rule lets one query
   occupy several slots under different labels and emit duplicate masks, and it makes
   *which* queries survive depend on the class posterior, which would turn the vocabulary
   ceiling from soft into partly hard.  Here every query is considered exactly once and
   the class is read only for the report.  The count of distinct queries upstream's rule
   would have kept is measured per scene and recorded in ``probe.json``
   (``upstream_topk_distinct_queries``) so the two rules can be compared without rerunning.
2. **No NMS** (plan §0, §4 stage B).  ODIN is a set predictor whose queries are solved
   jointly over all views; PanSt3R's panoptic argmax got no NMS either.  Recorded as
   ``nms: "none"`` in the config block rather than left implicit.
3. **Masks reach the 1024x1024 evaluation grid by one bilinear on the logits, then
   ``> 0``.**  ODIN runs at ``IMAGE_SIZE 512`` (``ResizeShortestEdge``, a pure resize with
   no crop), so a single interpolation is exact -- the same argument
   ``panst3r_common.masks_to_target`` makes for its trivial-crop case, and ``> 0`` is
   upstream's own mask threshold (``odin_model.py:1286``).
4. **A query with no pixels anywhere at target resolution is not emitted.**  It proposes
   nothing.  This is metric-neutral: ``hungarian_match`` drops zero-IoU assignments
   (``sam_vggt_3dtracking_benchmark.py:585``) and the P/R counters iterate over GT
   objects and their matched predictions only, so an unmatched prediction can never
   become a false positive.  Both counts are recorded.

``frame_names`` comes from ``track_io.list_scene_frame_paths`` -- the frozen module's own
listing rule -- and the geometry dump's frame stems are asserted against it element-wise,
so a stage-A/stage-B misalignment fails here rather than turning into plausible numbers.

Usage::

    PYTHONPATH=$PWD/submodules/odin $ODIN_ENV/bin/python \\
        benchmarks/baselines/odin_infer.py \\
        --stage_a_dir results/baselines/odin_phase3/stage_a \\
        --benchmark_root "$BENCH/scannetpp" --scene_ids 0a76e06478 \\
        --output_dir results/baselines/odin_phase3/scannetpp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines.odin_probe_forward import (  # noqa: E402
    CFG_FILE,
    CKPT,
    NETWORK_ATTEMPTS,
    OPTS,
    STUB_CALLS,
    block_network,
    build_inputs,
    instrument_stubs,
    run_forward,
)
from utils.provenance import write_run_provenance  # noqa: E402
from benchmarks.baselines.track_io import (  # noqa: E402
    list_scene_frame_paths,
    make_infer_seconds,
    make_track,
    write_tracks,
)

METHOD = "odin_scannet200_swin"
TARGET_SIZE = (1024, 1024)

#: Query -> track rule.  See decision 1 in the module docstring.
QUERY_RULE = "max_over_class_per_query"

#: Post-processing applied between the model's query set and ``tracks.json``.
NMS = "none"

#: Chunk size for the logits -> target-grid resize.  Bounds peak memory; has no effect on
#: the result (each query is resized independently either way).
RESIZE_CHUNK = 8


def discover_scenes(benchmark_root: Path, scene_ids: Optional[Sequence[str]]) -> List[Path]:
    """Scene directories, matching the frozen module's own discovery rule."""
    root = Path(benchmark_root)
    if scene_ids:
        dirs = [root / s for s in scene_ids]
        missing = [str(d) for d in dirs if not (d / "images").is_dir()]
        if missing:
            raise FileNotFoundError(f"no images/ in: {missing}")
        return dirs
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "images").is_dir())


def assert_frames_align(scene_id: str, geom_stems: Sequence[str],
                        frame_names: Sequence[str]) -> None:
    """Stage A's frame order must be the frozen listing's, element-wise."""
    expected = [Path(n).stem for n in frame_names]
    if list(geom_stems) != expected:
        raise RuntimeError(
            f"{scene_id}: geometry.npz frame order does not match the benchmark listing.\n"
            f"  geometry.npz ({len(geom_stems)}): {list(geom_stems)}\n"
            f"  benchmark    ({len(expected)}): {expected}"
        )


def masks_to_target(logits: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
    """``[Q, V, h, w]`` logits -> ``[Q, V, 1024, 1024]`` bool on the evaluation grid.

    One bilinear on the logits followed by ``> 0``.  ODIN's input transform is
    ``ResizeShortestEdge`` -- a pure resize, no crop -- so resized -> native -> target
    composes into resized -> target exactly, and no second resampling is introduced.
    ``> 0`` is upstream's own threshold (``odin_model.py:1286``); the logits, not a
    thresholded mask, are what gets interpolated, which is SAM-V's convention too.
    """
    q = logits.shape[0]
    out: List[np.ndarray] = []
    for start in range(0, q, RESIZE_CHUNK):
        chunk = logits[start:start + RESIZE_CHUNK].float()
        v = chunk.shape[1]
        up = F.interpolate(chunk.flatten(0, 1)[:, None], size=target_size,
                           mode="bilinear", align_corners=False)
        out.append((up[:, 0] > 0).view(-1, v, *target_size).cpu().numpy())
    return np.concatenate(out, axis=0)


def upstream_topk_distinct_queries(mask_cls: torch.Tensor, cfg) -> int:
    """How many *distinct* queries upstream's ``topk`` rule would have kept.

    Reproduces ``inference_2d_per_image``'s selection (``odin_model.py:1258-1279``) on the
    same logits, purely to measure the duplication the plan's rule removes.  Nothing
    downstream reads this.
    """
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    if cfg.SKIP_CLASSES is not None:
        keep = torch.ones(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, device=scores.device)
        keep[torch.tensor(cfg.SKIP_CLASSES, device=scores.device) - 1] = 0
        scores = scores[:, keep.bool()]
    num_classes = scores.shape[-1]
    k = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
    _, topk_indices = scores.flatten(0, 1).topk(k, sorted=False)
    return int(torch.unique(topk_indices // num_classes).numel())


def infer_scene(model, cfg, scene_dir: Path, geom_npz: Path, geometry: str,
                device: str, class_names: Sequence[str]) -> Dict[str, Any]:
    """One scene: forward pass -> per-query tracks on the evaluation grid."""
    frame_names = [p.name for p in list_scene_frame_paths(scene_dir)]

    g = np.load(geom_npz)
    prepare_s = float(g["prepare_seconds"]) if "prepare_seconds" in g.files else 0.0
    prepare_warmup = int(g["timing_warmup"]) if "timing_warmup" in g.files else 0

    # Measured but NOT charged into `infer_seconds`: reading stage A's geometry.npz back
    # off disk and reading + resizing the scene's JPEGs.  The npz round-trip exists only
    # because stage A and stage B live in different envs, and Point-SAM's stage B excludes
    # its own `inputs.npz` load for the same reason -- so excluding it keeps the two
    # baselines' clocks comparable.  Recorded per scene so the exclusion is a number, not
    # a claim (Phase 5, `notes/plans/odin-phase5-timing.md`).
    torch.cuda.synchronize()
    t_build = time.perf_counter()
    inputs = build_inputs(geom_npz, scene_dir, geometry, 1.0, cfg, device)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t_build
    inputs["scene_id"] = scene_dir.name
    assert_frames_align(scene_dir.name, inputs["frames"], frame_names)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = run_forward(model, inputs, cfg, device, with_instances=False)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    logits = res["per_query_logits"]                     # [Q, V, h, w]
    cls_score = res["query_class_score"]                 # [Q]  max over kept classes
    cls_idx = res["query_class_idx"]                     # [Q]
    n_queries, n_views = int(logits.shape[0]), int(logits.shape[1])
    if n_views != len(frame_names):
        raise RuntimeError(
            f"{scene_dir.name}: model returned {n_views} views for {len(frame_names)} frames"
        )

    # upstream's per-image mask score, pooled over views: mean sigmoid inside the mask.
    # Report-only -- the frozen matcher never reads a prediction's score (plan §6 item 9).
    with torch.no_grad():
        prob = logits.sigmoid().flatten(1)
        binary_native = (logits > 0).flatten(1)
        mask_score = ((prob * binary_native).sum(1)
                      / (binary_native.sum(1).clamp(min=1))).cpu().numpy()
    masks = masks_to_target(logits, TARGET_SIZE)         # [Q, V, 1024, 1024] bool
    cls_score_np = cls_score.cpu().numpy()
    cls_idx_np = cls_idx.cpu().numpy()

    tracks: List[Dict[str, Any]] = []
    query_records: List[Dict[str, Any]] = []
    n_empty = 0
    for qi in range(n_queries):
        per_view = {v: masks[qi, v] for v in range(n_views) if masks[qi, v].any()}
        areas = [int(masks[qi, v].sum()) for v in range(n_views)]
        name = (class_names[int(cls_idx_np[qi])]
                if 0 <= int(cls_idx_np[qi]) < len(class_names) else None)
        record = {
            "query": qi,
            "class_index": int(cls_idx_np[qi]),
            "class_name": name,
            "class_score": float(cls_score_np[qi]),
            "mask_score": float(mask_score[qi]),
            "score": float(cls_score_np[qi] * mask_score[qi]),
            "n_views_present": int(sum(a > 0 for a in areas)),
            "areas_target": areas,
        }
        query_records.append(record)
        if not per_view:
            n_empty += 1
            continue
        tracks.append(make_track(
            per_view, score=record["score"], query_id=qi,
            class_index=record["class_index"], class_name=name,
            class_score=record["class_score"], mask_score=record["mask_score"],
            source_view=-1,
        ))
    torch.cuda.synchronize()
    assemble_s = time.perf_counter() - t0

    probe = {
        "scene_id": scene_dir.name,
        "geometry": geometry,
        "n_frames": len(frame_names),
        "n_queries": n_queries,
        "n_tracks_emitted": len(tracks),
        "n_queries_empty_at_target": n_empty,
        "query_rule": QUERY_RULE,
        "upstream_topk_distinct_queries": upstream_topk_distinct_queries(res["mask_cls"], cfg),
        "nms": NMS,
        "n_tracks_present_in_all_views": int(
            sum(r["n_views_present"] == len(frame_names) for r in query_records)),
        "n_tracks_present_in_one_view": int(
            sum(r["n_views_present"] == 1 for r in query_records)),
        "n_voxels_per_scale_res5_to_res2": res["n_voxels_per_scale"],
        "native_hw": list(inputs["native_hw"]),
        "resized_hw": list(inputs["resized_hw"]),
        "padded_hw": list(res["padded_hw"]),
        "scale_m_per_unit": inputs["scale_m_per_unit"],
        "timing": {"prepare": prepare_s, "infer": infer_s, "assemble": assemble_s,
                   "build_inputs_excluded": build_s,
                   "stage_a_timing_warmup": prepare_warmup,
                   "forward_breakdown": res["seconds"]},
        "queries": query_records,
        "stub_calls_reached": list(STUB_CALLS),
        "network_attempts": list(NETWORK_ATTEMPTS),
    }
    return {
        "frame_names": frame_names,
        "tracks": tracks,
        "probe": probe,
        "infer_seconds": make_infer_seconds(prepare=prepare_s, infer=infer_s,
                                            assemble=assemble_s),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage_a_dir", required=True, type=Path,
                   help="stage-A output holding <scene>/geometry.npz")
    p.add_argument("--benchmark_root", required=True, type=Path,
                   help="e.g. .../3DTrackingBenchmark/scannetpp")
    p.add_argument("--scene_ids", nargs="*", default=None)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--geometry", choices=["vggt", "gt"], default="vggt",
                   help="'gt' is the plan's §4 Phase-6 appendix probe only; the row is vggt")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing_warmup", type=int, default=0,
                   help="whole-scene passes on the first scene, discarded before the timed "
                        "loop (Phase 5 uses 2, the frozen harness convention). 0 = the "
                        "Phase-3/4 bare wall clock, not a reportable timing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    block_network()

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model
    from detectron2.data import MetadataCatalog
    from odin import add_maskformer2_config, add_maskformer2_video_config

    stub_files = instrument_stubs()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(CFG_FILE)
    cfg.merge_from_list(OPTS)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    model.to(args.device)

    class_names = list(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)
    if cfg.SKIP_CLASSES is not None:
        skip = {c - 1 for c in cfg.SKIP_CLASSES}
        class_names = [n for i, n in enumerate(class_names) if i not in skip]

    scene_dirs = discover_scenes(args.benchmark_root, args.scene_ids)
    if not scene_dirs:
        raise SystemExit(f"no scenes under {args.benchmark_root}")

    resolved = {
        "method": METHOD,
        "ckpt": str(args.ckpt),
        "cfg_file": CFG_FILE,
        "opts": OPTS,
        "skip_classes": list(cfg.SKIP_CLASSES) if cfg.SKIP_CLASSES is not None else None,
        # Upstream's own typo, recorded verbatim rather than corrected (plan §7):
        # ScanNet200 id 119 is `mini fridge`, 199 `wall`, 200 `floor`, so upstream drops
        # mini fridge + floor and keeps wall while calling it "floor and wall".
        # `submodules/odin` is unpatched.
        "skip_classes_note": "upstream [119,200] = mini fridge + floor, NOT wall+floor",
        "vocabulary": "scannet200",
        "num_classes_after_skip": len(class_names),
        "query_rule": QUERY_RULE,
        "nms": NMS,
        "mask_threshold": "logit > 0",
        "mask_to_target": "single bilinear on logits, resized -> 1024x1024",
        "geometry_source": args.geometry,
        "image_size": 512,
        "target_size": list(TARGET_SIZE),
        "stage_a": "odin_geometry_probe.py --mode dump",
        # Phase 3/4 were plumbing and the row: their clock is a bare perf_counter with no
        # warmups and no exclusive-node discipline.  A Phase-5 timing run passes
        # `--timing_warmup 2` and runs strictly serially on an otherwise idle L40S
        # (CLAUDE.md §8); only then is `total_infer_seconds` reportable.
        "timing_is_phase3_wiring_only": args.timing_warmup == 0,
        "timing_warmup_scene_passes": args.timing_warmup,
        "timing_protocol": (
            "perf_counter + torch.cuda.synchronize on both ends; "
            f"{args.timing_warmup} discarded whole-scene warmup passes; "
            "geometry.npz read-back and image read/resize measured but NOT charged"
            if args.timing_warmup else
            "bare perf_counter, no warmup, no node discipline -- NOT a timing measurement"
        ),
    }
    print(json.dumps({k: v for k, v in resolved.items() if k != "opts"}, indent=2))

    preds_dir = args.output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    def geom_for(scene_dir: Path) -> Path:
        geom_npz = args.stage_a_dir / scene_dir.name / "geometry.npz"
        if not geom_npz.is_file():
            raise FileNotFoundError(
                f"{scene_dir.name}: no stage-A geometry at {geom_npz}. Run "
                f"odin_geometry_probe.py --mode dump first."
            )
        return geom_npz

    # Discarded warmup passes on the first scene, so the recorded seconds exclude cuDNN
    # autotuning and lazy CUDA-module loading -- the frozen harness convention (SAM-V's
    # `benchmark_inference_time.py`, Point-SAM's stage B).  Nothing is written.
    for wi in range(args.timing_warmup):
        print(f"[warmup {wi + 1}/{args.timing_warmup}] {scene_dirs[0].name}", flush=True)
        infer_scene(model, cfg, scene_dirs[0], geom_for(scene_dirs[0]), args.geometry,
                    args.device, class_names)
    if args.timing_warmup:
        torch.cuda.synchronize()

    all_probes: List[Dict[str, Any]] = []
    for scene_dir in scene_dirs:
        geom_npz = geom_for(scene_dir)
        out = infer_scene(model, cfg, scene_dir, geom_npz, args.geometry,
                          args.device, class_names)
        config = {**resolved, "scene_dir": str(scene_dir),
                  "geometry_npz": str(geom_npz),
                  "scale_m_per_unit": out["probe"]["scale_m_per_unit"]}
        write_tracks(
            preds_dir / scene_dir.name / "tracks.json",
            scene_id=scene_dir.name, frame_names=out["frame_names"],
            tracks=out["tracks"], method=METHOD, config=config,
            target_size=TARGET_SIZE, infer_seconds=out["infer_seconds"],
        )
        with (preds_dir / scene_dir.name / "probe.json").open("w") as f:
            json.dump(out["probe"], f, indent=2)
        all_probes.append(out["probe"])

        pr = out["probe"]
        print(
            f"[{scene_dir.name}] frames={pr['n_frames']} "
            f"tracks={pr['n_tracks_emitted']}/{pr['n_queries']} "
            f"(empty {pr['n_queries_empty_at_target']}, upstream-topk distinct "
            f"{pr['upstream_topk_distinct_queries']}) "
            f"allviews={pr['n_tracks_present_in_all_views']} "
            f"{pr['timing']['prepare']:.1f}s+{pr['timing']['infer']:.1f}s"
            f"+{pr['timing']['assemble']:.1f}s"
        )

    with (args.output_dir / "probe_summary.json").open("w") as f:
        json.dump({"method": METHOD, "model": resolved, "stub_modules": stub_files,
                   "scenes": all_probes}, f, indent=2)
    write_run_provenance(args.output_dir, config={**resolved, **vars(args) | {
        "stage_a_dir": str(args.stage_a_dir),
        "benchmark_root": str(args.benchmark_root),
        "output_dir": str(args.output_dir),
    }})
    print(f"\nwrote {len(all_probes)} scene(s) to {preds_dir}")
    print("stub calls reached:", STUB_CALLS or "NONE")
    print("network attempts  :", NETWORK_ATTEMPTS or "NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
