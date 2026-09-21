"""Stage B: run Point-SAM on Stage A's inputs, one call per prompt group.

Runs in the Point-SAM env (`$POINTSAM_ENV/bin/python`), which cannot import sam-hq, VGGT or
this repo's model package. It reads `inputs.npz` from Stage A and writes `masks.npz` -- packed
per-point masks plus scores and timings -- for Stage C to scatter, NMS and score.

Four things upstream forces, all of them checked rather than assumed (see
`benchmarks/baselines/README.md` and Phase 5.1 in `notes/results.md`):

- **`set_pointcloud()` does not exist.** There is no encode-once / prompt-many split: the point
  encoder re-runs on every `predict_masks` call. So cost scales with the number of prompt
  groups M, not just with point count, and M is recorded next to the timing -- the row is
  unreadable without it.
- **Colours normalise to [-1, 1]**, per upstream `evaluation/eval_kitti.py`.
  `evaluation/inference.py` only divides by 255 and is stale.
- **Crop per prompt group.** Upstream's own scene-scale benchmark does the same; Phase 5.1b
  measured +0.131 mean 3D IoU for it over the whole split.
- **Select by `stability_score`, not `iou_preds`.** Point-SAM's IoU head picks the best of its
  own 3 heads on 4/51 objects, worse than chance; stability gets 29/51.

The cloud each call sees comes from `pointsam_sampling.prompt_group_cloud` -- the same function
the published ceiling is measured through, so the ceiling row bounds *these* predictions.

Timing follows the frozen harness: `perf_counter` + `torch.cuda.synchronize()` and 2 warmup
passes before anything is recorded. `infer_seconds` is `{prepare, infer, assemble}`, never a
scalar; `prepare` is Stage A's cost (VGGT forward + AMG), carried through from its summary.
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

from benchmarks.baselines import pointsam_sampling as ps  # noqa: E402
from utils.provenance import write_run_provenance  # noqa: E402


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
def load_point_sam(repo: Path, ckpt: str, num_patches: int, patch_size: int):
    """Instantiate Point-SAM from upstream's own config and load the staged checkpoint.

    `num_patches` / `patch_size` are free parameters: the patch positional embedding is an MLP
    over 3D patch centres and timm's index-based pos_embed is bypassed, so no weight has a
    num_patches-shaped dimension. apex is not installed and is not needed -- it only swaps
    `nn.LayerNorm` for a fused one, same keys, same outputs.
    """
    sys.path.insert(0, str(repo))
    import hydra
    from omegaconf import OmegaConf
    from safetensors.torch import load_model

    cfg = OmegaConf.load(repo / "configs" / "model" / "default.yaml")
    cfg.pc_encoder.patch_embed.num_patches = num_patches
    cfg.pc_encoder.patch_embed.patch_size = patch_size
    model = hydra.utils.instantiate(cfg)
    missing = load_model(model, ckpt)
    if missing not in (None, ([], [])):
        print(f"[ckpt] load_model reported: {missing}")
    return model.eval().cuda()


def calculate_stability_score(masks: torch.Tensor, mask_threshold: float,
                              threshold_offset: float) -> torch.Tensor:
    """IoU between the mask thresholded high and low -- GT-free, no trained head.

    Transcribed from the frozen sam-hq `segment_anything/utils/amg.py:156`, the same quantity
    the frozen benchmark's PROPOSAL_STABILITY_SCORE_THRESH uses. Reduced over the last dim
    only, because Point-SAM's masks are per-point `[..., N]` rather than per-pixel
    `[..., H, W]`; the sam-hq version sums the final two dims for the same reason.
    """
    intersections = (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int32)
    return intersections / torch.clamp(unions, min=1)


def normalize_coords(xyz: torch.Tensor):
    """Point-SAM's own normalisation (upstream `evaluation/inference.py`)."""
    centre = xyz.mean(dim=1, keepdim=True)
    out = xyz - centre
    scale = out.norm(dim=2, keepdim=True).max()
    return out / scale, centre, scale


def normalize_colors(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    """rgb/255, then to [-1, 1] for `signed` -- upstream `evaluation/eval_kitti.py`."""
    out = rgb / 255.0
    if mode == "signed":
        out = (out - 0.5) / 0.5
    return out


# --------------------------------------------------------------------------------------
# one prompt group
# --------------------------------------------------------------------------------------
def run_group(model, xyz: np.ndarray, rgb: np.ndarray, anchor: np.ndarray,
              prompt_xyz: np.ndarray, crop_radius: float) -> Dict[str, Any]:
    """One Point-SAM call. Returns the per-point mask over the FULL scene cloud, and scores."""
    call = ps.prompt_group_cloud(xyz, anchor, crop_radius, ps.VOXEL_REL, ps.VOXEL_SCOPE)
    vxyz = call["voxel_xyz"]
    n_vox = len(vxyz)
    if n_vox == 0:
        return dict(point_mask=np.zeros(len(xyz), dtype=bool), stability=0.0, iou_pred=0.0,
                    head=-1, crop_voxels=0, voxel_edge=float(call["voxel_edge"]),
                    snap_distance=float("nan"))
    # voxel_of_point indexes the CROP's points, so the colours must be subset to match.
    vrgb = ps.voxel_mean_features(rgb[call["point_index"]].astype(np.float32),
                                  call["voxel_of_point"], n_vox)

    coords = torch.from_numpy(vxyz).float().cuda().unsqueeze(0)          # [1,V,3]
    coords_n, centre, scale = normalize_coords(coords)
    feats = normalize_colors(torch.from_numpy(vrgb).float().cuda().unsqueeze(0), ps.COLOR_NORM)

    # Snap each prompt to the nearest voxel of the cloud this call is given.
    #
    # A prompt is a pixel inside a SAM proposal, lifted to 3D off the full 1024 grid, so it is
    # not in general one of the voxel centroids -- and it need not even be inside the crop: the
    # crop is a ball around prompt 0, and prompts 1..k-1 are spread over the proposal by
    # `pole_plus_diverse`, so a prompt on a large proposal can land outside it. Point-SAM's
    # prompt encoder REJECTS that outright ("Input coordinates must be normalized to [-1, 1]"),
    # because normalisation is relative to the crop's own extent. Upstream prompts with actual
    # cloud points, so we do the same. `prompt_snap_distance` records how far this moves them.
    pxyz = np.asarray(prompt_xyz, dtype=np.float32)
    d2 = ((pxyz[:, None, :] - vxyz[None, :, :]) ** 2).sum(-1)
    nearest = d2.argmin(axis=1)
    snap_distance = float(np.sqrt(d2[np.arange(len(pxyz)), nearest]).max())
    prompt_coords = coords_n[:, nearest, :]
    prompt_labels = torch.ones(1, prompt_coords.shape[1], dtype=torch.bool, device="cuda")

    # Match upstream's grouper sizing for large clouds (eval_kitti.py:350-362).
    model.pc_encoder.patch_embed.grouper.num_groups = min(ps.NUM_PATCHES, n_vox)
    model.pc_encoder.patch_embed.grouper.group_size = ps.PATCH_SIZE

    with torch.inference_mode():
        masks, iou_preds = model.predict_masks(coords_n, feats, prompt_coords, prompt_labels,
                                               multimask_output=True)

    stability = calculate_stability_score(masks[0], 0.0, ps.STABILITY_OFFSET)
    head = int(stability.argmax())                                        # THE frozen rule
    voxel_mask = (masks[0, head] > 0).cpu().numpy()

    point_mask = np.zeros(len(xyz), dtype=bool)
    point_mask[call["point_index"]] = voxel_mask[call["voxel_of_point"]]
    return dict(point_mask=point_mask, stability=float(stability[head]),
                iou_pred=float(iou_preds[0, head]), head=head, crop_voxels=n_vox,
                voxel_edge=float(call["voxel_edge"]), snap_distance=snap_distance)


def infer_scene(stage_a_dir: Path, model, out_dir: Path, warmup: int,
                max_groups: int = 0) -> Dict[str, Any]:
    data = np.load(stage_a_dir / "inputs.npz", allow_pickle=True)
    prep = json.loads((stage_a_dir / "prepare_summary.json").read_text())
    xyz, rgb = data["xyz"], data["rgb"]
    anchor, prompt_xyz = data["anchor"], data["prompt_xyz"]
    crop_radius = float(data["crop_radius"])
    m_total = len(anchor)
    m = m_total if max_groups <= 0 else min(max_groups, m_total)

    # Warm up on real prompt groups, discarded, so the recorded time excludes cuDNN autotuning
    # and lazy CUDA-module loading -- the frozen timing harness's convention.
    for i in range(min(warmup, m)):
        run_group(model, xyz, rgb, anchor[i], prompt_xyz[i], crop_radius)
    torch.cuda.synchronize()

    results, per_group_seconds = [], []
    masks_packed = np.zeros((m, int(np.ceil(len(xyz) / 8))), dtype=np.uint8)
    t_infer0 = time.perf_counter()
    for g in range(m):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = run_group(model, xyz, rgb, anchor[g], prompt_xyz[g], crop_radius)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_group_seconds.append(dt)
        masks_packed[g] = np.packbits(r["point_mask"])
        results.append(dict(prompt_id=int(data["prompt_id"][g]), stability=r["stability"],
                            iou_pred=r["iou_pred"], head=r["head"],
                            crop_voxels=r["crop_voxels"], voxel_edge=r["voxel_edge"],
                            snap_distance=r["snap_distance"],
                            mask_points=int(r["point_mask"].sum()), seconds=dt))
        if (g + 1) % 50 == 0 or g == m - 1:
            print(f"    group {g + 1}/{m}  {np.mean(per_group_seconds):.3f}s/group  "
                  f"eta {(m - g - 1) * float(np.mean(per_group_seconds)):.0f}s", flush=True)
    t_infer = time.perf_counter() - t_infer0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "masks.npz", masks_packed=masks_packed, num_points=np.int64(len(xyz)),
        stability=np.array([r["stability"] for r in results], dtype=np.float32),
        iou_pred=np.array([r["iou_pred"] for r in results], dtype=np.float32),
        head=np.array([r["head"] for r in results], dtype=np.int32),
        prompt_id=np.array([r["prompt_id"] for r in results], dtype=np.int32),
        crop_voxels=np.array([r["crop_voxels"] for r in results], dtype=np.int32),
    )

    cv = np.array([r["crop_voxels"] for r in results])
    pg = np.array(per_group_seconds)
    sn = np.array([r["snap_distance"] for r in results], dtype=float)
    sn = sn[np.isfinite(sn)]
    summary = dict(
        scene_id=prep["scene_id"], method="point_sam",
        num_prompt_groups=int(m), num_prompt_groups_total=int(m_total),
        num_points=int(len(xyz)), crop_radius=crop_radius,
        cloud_diagonal=float(data["cloud_diagonal"]),
        crop_voxels=dict(min=int(cv.min()), median=float(np.median(cv)), max=int(cv.max())),
        seconds_per_group=dict(mean=float(pg.mean()), median=float(np.median(pg)),
                               min=float(pg.min()), max=float(pg.max())),
        # Per group, the largest distance a prompt moved when snapped onto the voxel cloud,
        # in VGGT units. Compare against the voxel edge: a snap of about one voxel is the
        # quantisation the ceiling already accounts for; much more means the prompt was
        # outside its own crop and got pulled to the crop's boundary.
        prompt_snap_distance=(dict(mean=float(sn.mean()), median=float(np.median(sn)),
                                   p90=float(np.percentile(sn, 90)), max=float(sn.max()))
                              if len(sn) else {}),
        voxel_edge=dict(median=float(np.median([r["voxel_edge"] for r in results]))),
        # prepare is Stage A's cost -- the VGGT forward plus the sam-hq AMG proposals -- and it
        # is charged to Point-SAM because Point-SAM cannot run without it.
        infer_seconds=dict(prepare=float(prep["prepare_seconds"]), infer=float(t_infer),
                           assemble=0.0),
        warmup_groups=int(min(warmup, m)),
        frozen_config=ps.frozen_config(),
        per_group=results,
    )
    (out_dir / "infer_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage_a_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--scene_ids", nargs="*", default=None)
    ap.add_argument("--ckpt",
                    default=str(_BASELINE_WEIGHTS / "point_sam" / "model.safetensors"))
    ap.add_argument("--repo", type=Path, default=REPO_ROOT / "submodules" / "point-sam")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup groups, discarded before timing (frozen harness convention)")
    ap.add_argument("--max_groups", type=int, default=0,
                    help="cap prompt groups per scene; 0 = all. For cost extrapolation only -- "
                         "a capped run is NOT a scoreable row.")
    args = ap.parse_args()

    scenes = sorted(p for p in args.stage_a_dir.iterdir()
                    if p.is_dir() and (p / "inputs.npz").is_file())
    if args.scene_ids:
        scenes = [p for p in scenes if p.name in set(args.scene_ids)]
    if not scenes:
        raise SystemExit(f"no Stage-A scenes under {args.stage_a_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(args.output_dir, config=vars(args) | {
        "frozen_config": ps.frozen_config(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "apex": False,
    })

    model = load_point_sam(args.repo, args.ckpt, ps.NUM_PATCHES, ps.PATCH_SIZE)
    summaries: List[Dict[str, Any]] = []
    for sd in scenes:
        print(f"[{sd.name}] ...", flush=True)
        s = infer_scene(sd, model, args.output_dir / sd.name, args.warmup, args.max_groups)
        summaries.append(s)
        t = s["infer_seconds"]
        m_str = (str(s["num_prompt_groups"])
                 if s["num_prompt_groups"] == s["num_prompt_groups_total"]
                 else f"{s['num_prompt_groups']}/{s['num_prompt_groups_total']}")
        print(f"[{s['scene_id']}] M={m_str}"
              f"  crop {s['crop_voxels']['min']}-{s['crop_voxels']['max']} vox "
              f"(med {s['crop_voxels']['median']:.0f})  "
              f"{s['seconds_per_group']['mean']:.3f}s/group  "
              f"infer {t['infer']:.1f}s  prepare {t['prepare']:.1f}s", flush=True)

    (args.output_dir / "infer_summary.json").write_text(json.dumps(summaries, indent=2))
    tot = sum(s["infer_seconds"]["infer"] for s in summaries)
    print(f"\n{len(summaries)} scene(s), {sum(s['num_prompt_groups'] for s in summaries)} "
          f"prompt groups, {tot:.1f}s infer total -> {args.output_dir}")


if __name__ == "__main__":
    main()
