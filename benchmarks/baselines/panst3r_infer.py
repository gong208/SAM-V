#!/usr/bin/env python3
"""
Stage B for PanSt3R: benchmark RGB frames -> multi-view panoptic -> tracks.json.

Runs in the PanSt3R env (``$PANST3R_ENV``).  One adapter drives both
released checkpoints; ``--variant {v1,v2}`` picks the weights, and each
checkpoint keeps whatever post-processing default its own ``args`` carry
(v1 = ``qubo``, v2 = ``standard_v2``) unless ``--postprocess`` overrides it.

Cross-view tracks
-----------------
``panoptic_inference_v2`` / ``panoptic_inference_qubo`` compute the panoptic
argmax **jointly over all views at once** -- ``cur_masks`` is ``[K, V, H, W]``
and ``panoptic_seg`` is ``[V, H, W]`` painted in a single pass
(``postprocess.py:65-113``).  So a segment ``id`` is *already* cross-view
consistent, and ``id`` and ``query_id`` are in 1:1 correspondence within one
call.  Grouping by ``query_id`` is therefore equivalent to grouping by ``id``;
the adapter asserts the bijection rather than assuming it.

Every survivor is a "thing": ``postprocess.py:81`` hardcodes ``isthing = True``
with a TODO, so no stuff-merging happens and there is no thing/stuff filter to
apply.

Usage::

    $PANST3R_ENV/bin/python benchmarks/baselines/panst3r_infer.py \\
        --variant v2 --vocab train_union \\
        --benchmark_root "$BENCH/scannetpp" --scene_ids 0a76e06478 \\
        --output_dir results/baselines/panst3r_v2_train_union_scannetpp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Baseline checkpoints are not redistributed with this repository. Stage them
# yourself and point $BASELINE_WEIGHTS at the directory holding them; the
# per-baseline subdirectory layout is documented in benchmarks/baselines/README.md.
_BASELINE_WEIGHTS = Path(os.environ.get(
    "BASELINE_WEIGHTS", str(REPO_ROOT / "checkpoints" / "baselines")))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines.panst3r_common import (  # noqa: E402
    crop_box_from_to_orig,
    load_scene_frames,
    masks_to_target,
)
from utils.provenance import write_run_provenance  # noqa: E402
from benchmarks.baselines.track_io import (  # noqa: E402
    make_infer_seconds,
    make_track,
    write_tracks,
)
from benchmarks.baselines.vocab import VOCAB_CHOICES, resolve_vocab  # noqa: E402

CHECKPOINTS = {
    "v1": str(_BASELINE_WEIGHTS / "panst3r" / "panst3r_v1_512_5ds.pth"),
    "v2": str(_BASELINE_WEIGHTS / "panst3r" / "panst3r_v2_512_5ds.pth"),
}
TARGET_SIZE = (1024, 1024)


def discover_scenes(benchmark_root: Path, scene_ids) -> List[Path]:
    """Scene directories, matching the frozen module's own discovery rule."""
    root = Path(benchmark_root)
    if scene_ids:
        dirs = [root / s for s in scene_ids]
        missing = [str(d) for d in dirs if not (d / "images").is_dir()]
        if missing:
            raise FileNotFoundError(f"no images/ in: {missing}")
        return dirs
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "images").is_dir())


def run_postprocess(model, pan_out, true_shape_np, which: str):
    from panst3r.engine import (
        panoptic_inference_qubo,
        panoptic_inference_v1,
        panoptic_inference_v2,
    )

    label_mode = model.panoptic_decoder.label_mode
    kwargs = dict(label_mode=label_mode, device="cpu", multi_ar=True)
    if which == "qubo":
        return panoptic_inference_qubo(pan_out["pred_logits"], pan_out["pred_masks"],
                                       true_shape_np, **kwargs)
    if which == "standard_v1":
        return panoptic_inference_v1(pan_out["pred_logits"], pan_out["pred_masks"],
                                     true_shape_np, **kwargs)
    if which == "standard_v2":
        return panoptic_inference_v2(pan_out["pred_logits"], pan_out["pred_masks"],
                                     true_shape_np, **kwargs)
    raise ValueError(f"unknown postprocess {which!r}")


def infer_scene(
    model,
    scene_dir: Path,
    class_names: List[str],
    args,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one scene end to end, returning everything needed to write tracks.json."""
    amp = False if args.amp == "False" else args.amp

    t_load = time.time()
    paths, views = load_scene_frames(scene_dir, args.image_size,
                                     model.must3r_encoder.patch_size, verbose=args.verbose)
    n = len(views)
    if args.num_keyframes not in (0, None) and args.num_keyframes != n:
        raise ValueError(
            f"{scene_dir.name}: --num_keyframes={args.num_keyframes} != {n} frames. "
            "The plan fixes num_keyframes = N so every view is decoded jointly from the "
            "shared query set; anything else takes the non-keyframe render path and "
            "reorders frames."
        )
    num_keyframes = n

    imgs = [v["img"].to(device) for v in views]
    true_shape = torch.stack([torch.from_numpy(v["true_shape"]).to(device) for v in views], dim=0)
    load_s = time.time() - t_load

    t_infer = time.time()
    with torch.no_grad():
        out_3D, pan_out = model.forward_inference_multi_ar(
            imgs, true_shape, class_names, num_keyframes=num_keyframes,
            use_retrieval=False, max_bs=1, outdevice="cpu", amp=amp,
        )
    infer_s = time.time() - t_infer

    t_asm = time.time()
    which = args.postprocess or model.postprocess_default
    pan_preds = run_postprocess(model, pan_out, true_shape.cpu().numpy(), which)
    pan = pan_preds[0]
    pan_maps = [np.asarray(p.cpu().numpy() if torch.is_tensor(p) else p) for p in pan["pan"]]
    segments = list(pan["segments_info"])

    # Per-query max class score: sigmoid(logit_scale * cos).max over the
    # vocabulary, the quantity that gates survival (postprocess.py:41-42).
    logits = pan_out["pred_logits"]
    logits = logits[0] if logits.dim() == 3 else logits          # [Q, C]
    query_scores = torch.sigmoid(logits.float()).max(-1).values.cpu().numpy()
    n_queries = int(query_scores.shape[0])

    # id <-> query_id must be a bijection (see module docstring).
    ids = [int(s["id"]) for s in segments]
    qids = [int(s["query_id"]) for s in segments]
    bijective = len(set(ids)) == len(ids) and len(set(qids)) == len(qids) and len(ids) == len(qids)

    # Build one track per segment, mapping each view's mask to the 1024^2 grid.
    tracks: List[Dict[str, Any]] = []
    seg_records: List[Dict[str, Any]] = []
    per_view_proc = {int(s["id"]): [] for s in segments}
    for view_idx, pmap in enumerate(pan_maps):
        for s in segments:
            per_view_proc[int(s["id"])].append(pmap == int(s["id"]))

    # Validate the crop/resize inverse once per scene, independently of whether
    # anything survived post-processing -- a run with zero segments must still
    # be able to state that its geometry mapping was well-formed.
    proc_hw = (int(pan_maps[0].shape[0]), int(pan_maps[0].shape[1]))
    crop_boxes = [crop_box_from_to_orig(v["to_orig"], proc_hw, v["native_hw"]) for v in views]
    if any(cb != crop_boxes[0] for cb in crop_boxes):
        raise ValueError(f"{scene_dir.name}: frames disagree on the crop box: {crop_boxes}")
    crop_info = crop_boxes[0]

    for s in segments:
        sid, qid = int(s["id"]), int(s["query_id"])
        masks_proc = np.stack(per_view_proc[sid], axis=0)         # [V, h, w]
        per_view: Dict[int, np.ndarray] = {}
        areas_proc, areas_target = [], []
        for view_idx, view in enumerate(views):
            m = masks_to_target(masks_proc[view_idx][None], view["to_orig"],
                                view["native_hw"], TARGET_SIZE)[0]
            areas_proc.append(int(masks_proc[view_idx].sum()))
            areas_target.append(int(m.sum()))
            if m.any():
                per_view[view_idx] = m
        category_id = int(s["category_id"]) if not torch.is_tensor(s["category_id"]) \
            else int(s["category_id"].item())
        score = float(query_scores[qid]) if qid < n_queries else 0.0
        record = {
            "id": sid, "query_id": qid, "category_id": category_id,
            "category_name": class_names[category_id] if 0 <= category_id < len(class_names) else None,
            "score": score,
            "n_views_present": int(sum(a > 0 for a in areas_target)),
            "areas_proc": areas_proc, "areas_target": areas_target,
        }
        for extra in ("class_prob", "mask_conf", "area"):
            if extra in s:
                v = s[extra]
                record[extra] = float(v.item()) if torch.is_tensor(v) else float(v)
        seg_records.append(record)
        tracks.append(make_track(per_view, score=score, query_id=qid, source_view=-1))
    assemble_s = time.time() - t_asm

    probe = {
        "scene_id": scene_dir.name,
        "n_frames": n,
        "num_keyframes": num_keyframes,
        "all_frames_are_keyframes": True,
        "postprocess": which,
        "n_queries": n_queries,
        "n_instances_survived": len(segments),
        "survival_fraction": len(segments) / n_queries if n_queries else 0.0,
        "id_query_id_bijective": bool(bijective),
        "query_max_class_score": {
            "p10": float(np.percentile(query_scores, 10)),
            "p50": float(np.percentile(query_scores, 50)),
            "p90": float(np.percentile(query_scores, 90)),
            "min": float(query_scores.min()), "max": float(query_scores.max()),
            "mean": float(query_scores.mean()),
        },
        "survivor_scores": sorted((r["score"] for r in seg_records), reverse=True),
        "n_tracks_present_in_all_views": int(sum(r["n_views_present"] == n for r in seg_records)),
        "n_tracks_present_in_one_view": int(sum(r["n_views_present"] == 1 for r in seg_records)),
        "processed_hw": [int(pan_maps[0].shape[0]), int(pan_maps[0].shape[1])],
        "native_hw": [list(v["native_hw"]) for v in views],
        "crop_box": crop_info,
        "segments": seg_records,
        "timing": {"load": load_s, "infer": infer_s, "assemble": assemble_s},
    }
    return {
        "frame_names": [v["name"] for v in views],
        "tracks": tracks,
        "probe": probe,
        "infer_seconds": make_infer_seconds(prepare=0.0, infer=infer_s, assemble=assemble_s),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["v1", "v2"], default="v2")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--benchmark_root", required=True, type=Path)
    p.add_argument("--scene_ids", nargs="*", default=None)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--vocab", choices=VOCAB_CHOICES, default="demo_default")
    p.add_argument("--vocab-file", dest="vocab_file", type=Path, default=None)
    p.add_argument("--split", choices=["scannetpp", "scannet"], default=None,
                   help="needed by --vocab split_labels; inferred from benchmark_root if omitted")
    p.add_argument("--postprocess", choices=["qubo", "standard_v1", "standard_v2"], default=None,
                   help="default: whatever the checkpoint's own args carry")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_keyframes", type=int, default=0, help="0 = all frames (required setting)")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "False"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from panst3r import PanSt3R

    split = args.split or (args.benchmark_root.name
                           if args.benchmark_root.name in ("scannetpp", "scannet") else None)
    weights = args.weights or Path(CHECKPOINTS[args.variant])
    device = torch.device(args.device)
    method = f"panst3r_{args.variant}"

    scene_dirs = discover_scenes(args.benchmark_root, args.scene_ids)
    if not scene_dirs:
        raise SystemExit(f"no scenes under {args.benchmark_root}")

    print(f"loading {args.variant} from {weights}")
    model = PanSt3R.from_checkpoint(str(weights)).to(device).eval()
    resolved_model = {
        "variant": args.variant,
        "weights": str(weights),
        "label_mode": model.panoptic_decoder.label_mode,
        "postprocess_default": model.postprocess_default,
        "postprocess_used": args.postprocess or model.postprocess_default,
        "postprocess_forced": args.postprocess is not None,
        "qubo_enabled": bool(model.qubo_enabled),
        "patch_size": int(model.must3r_encoder.patch_size),
        "image_size": args.image_size,
        "amp": args.amp,
        "use_retrieval": False,
        "num_keyframes": "all",
        "target_size": list(TARGET_SIZE),
        "curope": False,
    }
    print(json.dumps(resolved_model, indent=2))

    preds_dir = args.output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    all_probes: List[Dict[str, Any]] = []
    for scene_dir in scene_dirs:
        names, vocab_meta = resolve_vocab(
            vocab=args.vocab if args.vocab_file is None else None,
            vocab_file=args.vocab_file, scene_dir=scene_dir, split=split,
        )
        model.set_vocab(names, device=device)

        out = infer_scene(model, scene_dir, names, args, device)
        config = {**resolved_model, **{k: v for k, v in vocab_meta.items()
                                       if k != "vocab_class_names"},
                  "split": split, "scene_dir": str(scene_dir)}
        # The full list, not just the flag name or its hash (CLAUDE.md §6).
        config["vocab_class_names"] = vocab_meta["vocab_class_names"]

        write_tracks(
            preds_dir / scene_dir.name / "tracks.json",
            scene_id=scene_dir.name, frame_names=out["frame_names"],
            tracks=out["tracks"], method=method, config=config,
            target_size=TARGET_SIZE, infer_seconds=out["infer_seconds"],
        )
        probe = {**out["probe"], "vocab": {k: v for k, v in vocab_meta.items()
                                           if k != "vocab_class_names"}}
        with (preds_dir / scene_dir.name / "probe.json").open("w") as f:
            json.dump(probe, f, indent=2)
        all_probes.append(probe)

        pr = probe["query_max_class_score"]
        print(
            f"[{scene_dir.name}] vocab={vocab_meta['vocab_id']} C={vocab_meta['vocab_size']} "
            f"survived={probe['n_instances_survived']}/{probe['n_queries']} "
            f"score p10/p50/p90={pr['p10']:.3f}/{pr['p50']:.3f}/{pr['p90']:.3f} "
            f"allviews={probe['n_tracks_present_in_all_views']} "
            f"bijective={probe['id_query_id_bijective']} "
            f"{probe['timing']['infer']:.1f}s+{probe['timing']['assemble']:.1f}s"
        )

    with (args.output_dir / "probe_summary.json").open("w") as f:
        json.dump({"method": method, "model": resolved_model, "scenes": all_probes}, f, indent=2)
    write_run_provenance(args.output_dir, config={**resolved_model, **vars(args) | {
        "benchmark_root": str(args.benchmark_root), "output_dir": str(args.output_dir),
        "weights": str(weights), "vocab_file": str(args.vocab_file), "split": split,
    }})
    print(f"\nwrote {len(all_probes)} scene(s) to {preds_dir}")


if __name__ == "__main__":
    main()
