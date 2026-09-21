"""Stage C: Point-SAM's per-point masks -> per-view masks -> NMS -> `tracks.json`.

Runs in the project env, because the NMS is the frozen benchmark's own:
`SamVGGTAutomaticMaskGenerator._mask_nms`, a `@staticmethod`, called directly and unmodified.
Nothing frozen is edited or reimplemented here -- this file only assembles.

The route, per scene:

    masks.npz (Stage B)      per-point boolean mask over the stride cloud, one per prompt group
      -> output floors       `iou_preds > 0.40` then `stability_score >= 0.2`, SAM-V's own
                             decoder-output filters, same operators and same order
      -> scatter to pixels   via the point -> pixel map Stage A wrote, i.e. a LOOKUP, never a
                             rendering: each retained pixel paints the stride x stride block it
                             stands for (`pointsam_sampling.scatter_to_full_masks`)
      -> panoramic NMS       at 256 x (256*F), ranked by `stability_score`, iou_threshold 0.95
      -> tracks.json         full-resolution per-view RLE, frames where the track is empty
                             omitted, matching SAM-V's own convention

**The NMS runs on a 256 x (256*F) panorama, not at full resolution.** That is what the frozen
AMG does (`masks/automatic_mask_generator.py:117-118`, `_LR_H = _LR_W = 256`): the panoramic
mask matrix is materialised densely for the IoU matmul, and 381 tracks x 1024 x (1024*6) would
be ~9.6 GB. Downsampling matches the frozen path rather than deviating from it.

**The NMS is ranked by `stability_score`, not `iou_preds`** (`pointsam_sampling.NMS_SCORE`).
Ranking by a head score that loses to chance would undo the selection rule at the last step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines import pointsam_sampling as ps  # noqa: E402
from utils.provenance import write_run_provenance  # noqa: E402
from benchmarks.baselines.track_io import (  # noqa: E402
    encode_mask_to_rle,
    make_infer_seconds,
    make_track,
    write_tracks,
)
from benchmarks.sam_vggt_3dtracking_benchmark import TARGET_SIZE  # noqa: E402
from masks.automatic_mask_generator import SamVGGTAutomaticMaskGenerator  # noqa: E402

LR = 256          # the frozen AMG's panorama resolution for NMS (_LR_H / _LR_W)


def to_panorama_lowres(masks: np.ndarray, lr: int = LR) -> np.ndarray:
    """[F,H,W] per-view masks -> [lr, lr*F] panorama, nearest-downsampled.

    Width-concatenated across views, which is the layout every panoramic quantity in this
    project uses -- the same one O-IoU is computed over (CLAUDE.md §10).
    """
    f, h, w = masks.shape
    rows = (np.arange(lr) * h // lr).clip(0, h - 1)
    cols = (np.arange(lr) * w // lr).clip(0, w - 1)
    small = masks[:, rows][:, :, cols]                 # [F,lr,lr]
    return np.concatenate(list(small), axis=1)         # [lr, lr*F]


def assemble_scene(stage_a_dir: Path, infer_dir: Path, out_dir: Path) -> Dict[str, Any]:
    data = np.load(stage_a_dir / "inputs.npz", allow_pickle=True)
    prep = json.loads((stage_a_dir / "prepare_summary.json").read_text())
    mdata = np.load(infer_dir / "masks.npz", allow_pickle=True)
    binfo = json.loads((infer_dir / "infer_summary.json").read_text())

    pixel = data["pixel"]
    stride = int(data["stride"])
    frame_names = [str(x) for x in data["frame_names"]]
    n_frames = len(frame_names)
    n_points = int(mdata["num_points"])
    packed = mdata["masks_packed"]
    scores = mdata["stability"] if ps.NMS_SCORE == "stability_score" else mdata["iou_pred"]
    m = len(packed)

    # ---- output floors: SAM-V's own, mirrored ------------------------------------------
    # The frozen generator filters its decoder's output masks before the NMS, IoU first and
    # strictly, then stability inclusively (`masks/automatic_mask_generator.py:366-367`,
    # `:390-391`). Same operators, same order, same values here -- otherwise Point-SAM enters
    # Hungarian matching unfiltered while SAM-V's row is filtered, and the precision comparison
    # is not a comparison.
    iou_preds = np.asarray(mdata["iou_pred"])
    stabilities = np.asarray(mdata["stability"])
    pass_iou = iou_preds > ps.PRED_IOU_FLOOR                          # strict, as upstream
    pass_stab = pass_iou & (stabilities >= ps.STABILITY_FLOOR)        # inclusive, as upstream

    t0 = time.perf_counter()
    per_view: List[Dict[int, np.ndarray]] = []
    lowres: List[np.ndarray] = []
    keep_idx: List[int] = []
    n_dropped_empty = 0
    for g in range(m):
        if not pass_stab[g]:
            continue
        point_mask = np.unpackbits(packed[g], count=n_points).astype(bool)
        if not point_mask.any():
            n_dropped_empty += 1
            continue
        views = ps.scatter_to_full_masks(point_mask, pixel, n_frames, stride, TARGET_SIZE)
        if not views.any():
            n_dropped_empty += 1
            continue
        per_view.append({f: views[f] for f in range(n_frames) if views[f].any()})
        lowres.append(views)
        keep_idx.append(g)

    n_nonempty = len(keep_idx)
    if n_nonempty == 0:
        raise RuntimeError(
            f"{prep['scene_id']}: nothing survived the output floors "
            f"(M={m}, {int(pass_iou.sum())} passed iou>{ps.PRED_IOU_FLOOR}, "
            f"{int(pass_stab.sum())} passed stability>={ps.STABILITY_FLOOR}, "
            f"{n_dropped_empty} of those had an empty mask)")

    # ---- panoramic NMS: the frozen static method, called directly ------------------------
    rles = [encode_mask_to_rle(to_panorama_lowres(v)) for v in lowres]
    nms_scores = torch.as_tensor(np.asarray(scores)[keep_idx], dtype=torch.float32)
    kept = SamVGGTAutomaticMaskGenerator._mask_nms(rles, nms_scores, ps.NMS_IOU_THRESHOLD)
    kept = [int(i) for i in kept.tolist()]

    tracks = [make_track(per_view[i], score=float(nms_scores[i]),
                         stability_score=float(mdata["stability"][keep_idx[i]]),
                         predicted_iou=float(mdata["iou_pred"][keep_idx[i]]),
                         prompt_id=int(mdata["prompt_id"][keep_idx[i]]),
                         source_view=int(data["prompt_frame"][keep_idx[i]]))
              for i in kept]
    t_assemble = time.perf_counter() - t0

    infer_seconds = make_infer_seconds(
        prepare=float(binfo["infer_seconds"]["prepare"]),
        infer=float(binfo["infer_seconds"]["infer"]),
        assemble=float(t_assemble),
    )
    config = dict(ps.frozen_config())
    config.update(
        num_prompt_groups=int(binfo["num_prompt_groups"]),
        num_passed_pred_iou_floor=int(pass_iou.sum()),
        num_passed_stability_floor=int(pass_stab.sum()),
        num_dropped_empty=int(n_dropped_empty),
        num_nonempty_masks=int(n_nonempty),
        num_tracks_after_nms=int(len(tracks)),
        crop_radius=float(data["crop_radius"]),
        cloud_diagonal=float(data["cloud_diagonal"]),
        crop_voxels=binfo["crop_voxels"],
        seconds_per_group=binfo["seconds_per_group"],
        nms_panorama_resolution=[LR, LR * n_frames],
        prompts_cached_in_stage_a=bool(prep.get("prompts_cached", False)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tracks(out_dir / "tracks.json", prep["scene_id"], frame_names, tracks,
                 method="point_sam", config=config, target_size=TARGET_SIZE,
                 infer_seconds=infer_seconds)
    return dict(scene_id=prep["scene_id"], num_prompt_groups=int(binfo["num_prompt_groups"]),
                num_passed_pred_iou_floor=int(pass_iou.sum()),
                num_passed_stability_floor=int(pass_stab.sum()),
                num_nonempty_masks=int(n_nonempty), num_tracks=int(len(tracks)),
                assemble_seconds=t_assemble, infer_seconds=infer_seconds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage_a_dir", required=True, type=Path)
    ap.add_argument("--infer_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="the preds/ directory score_baseline.py reads")
    ap.add_argument("--scene_ids", nargs="*", default=None)
    args = ap.parse_args()

    scenes = sorted(p for p in args.infer_dir.iterdir()
                    if p.is_dir() and (p / "masks.npz").is_file())
    if args.scene_ids:
        scenes = [p for p in scenes if p.name in set(args.scene_ids)]
    if not scenes:
        raise SystemExit(f"no Stage-B scenes under {args.infer_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(args.output_dir, config=vars(args) | {
        "frozen_config": ps.frozen_config(), "nms_lowres": LR})

    summaries = []
    for sd in scenes:
        s = assemble_scene(args.stage_a_dir / sd.name, sd, args.output_dir / sd.name)
        summaries.append(s)
        t = s["infer_seconds"]
        print(f"[{s['scene_id']}] M={s['num_prompt_groups']} -> "
              f"{s['num_passed_pred_iou_floor']} after iou>{ps.PRED_IOU_FLOOR} -> "
              f"{s['num_passed_stability_floor']} after stab>={ps.STABILITY_FLOOR} -> "
              f"{s['num_nonempty_masks']} non-empty -> {s['num_tracks']} tracks after NMS  "
              f"(prepare {t['prepare']:.1f}s, infer {t['infer']:.1f}s, "
              f"assemble {t['assemble']:.1f}s)")

    (args.output_dir / "assemble_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n{len(summaries)} scene(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
