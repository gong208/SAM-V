"""The Point-SAM sampling path, shared by the ceiling measurement and Stage A.

`pointsam_ceiling.py` (Phase 5.2) and `prepare_pointsam_inputs.py` (Stage 5.3-A) both import
from here. That is the point: the ceiling is only meaningful if GT masks travel the *identical*
route predictions do, and sharing one implementation makes that true by construction rather
than by two files being carefully kept in step.

The route is:

    RGB frames (native, e.g. 690x920)
      -> VGGT forward at 896x896 (centre-padded to square first!)
      -> crop the padding away, resample XYZ to the benchmark's 1024x1024 grid
      -> stride-N pixel grid, drop low world_points_conf
      -> voxel merge; one voxel carries one value
      -> scatter each voxel's value back to every pixel that fell into it
      -> nearest-upsample the strided grid to 1024x1024

**The padding is not cosmetic.** `SamVGGT.preprocess_vggt_images` centre-pads to a square and
then resizes, while the frozen benchmark reaches 1024x1024 by a direct, non-aspect-preserving
`F.interpolate` (`sam_vggt_3dtracking_benchmark.py:191-200`, `:203-209`). For a 690x920 frame
VGGT pads 115 rows top and bottom, so real image content occupies rows 112..784 of the 896
output. Resampling the 896 map straight onto the 1024 grid — which the execution plan assumed
was safe, on the grounds that "both are full-frame square resizes" — would shift content by up
to 12.5% of frame height and silently produce plausible, wrong numbers. `world_points_to_grid`
inverts the pad using the `coords` box `preprocess_vggt_images` itself returns.

Project env (`python`) with the mandatory PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------------------
# THE FROZEN POINT-SAM CONFIG
# --------------------------------------------------------------------------------------
# One config, one place. Stage A (`prepare_pointsam_inputs.py`), Stage B
# (`pointsam_infer.py`), Stage C (`pointsam_assemble.py`) and the ceiling measurement
# (`pointsam_ceiling.py`) all import these names. None of them may take these as loose CLI
# defaults: a ceiling measured at settings the run does not use is not that run's ceiling,
# and that is the condition under which the Point-SAM row was approved for the paper
# (notes/results.md, Phase 5.2).
#
# Every value below is a measurement or the direct consequence of one. Changing any of them
# invalidates the published ceiling and requires re-running `pointsam_ceiling.py`.

#: Pixel stride of the sampling grid. Phase 5.2 measured stride 4 and stride 2; stride 2 buys
#: ~0.02-0.05 ceiling IoU for 1.4-2.4x the points and still does not reach the 0.95 gate, so
#: it is not an in-budget alternative. Point-SAM's own guidance (group_number=2048 /
#: group_size=256) targets the 100-150k regime; stride 4 lands at 121k-303k voxels/scene.
STRIDE = 4

#: Voxel edge as a fraction of the scene cloud's bbox diagonal. VGGT's world scale is
#: ARBITRARY and differs per scene (measured: 1 unit = 1.28-4.04 m across the 8 scenes with GT
#: depth), so an absolute voxel size is a different physical voxel in every scene and makes
#: ceilings incomparable. 0.0023 is the only value Phase 5.2 measured at STRIDE=4; the finer
#: 0.0012 was only ever measured at stride 2, where it costs a median 711k voxels/scene --
#: 5-7x the budget. Do not re-pick this; it is read off the published ceiling.
VOXEL_REL = 0.0023

#: `world_points_conf` floor. Phase 5.2's ceilings were all measured at 0 (no filtering), and
#: the published ceiling must match the run, so the run does not filter either.
CONF_THRESH = 0.0

#: Crop radius around a prompt group, as a fraction of the scene cloud's bbox diagonal.
#:
#: Phase 5.1b measured none / 1.5 m / 3.0 m and 1.5 m won (+0.131 mean 3D IoU over no crop,
#: 8 scenes / 51 objects, under stability selection). That measurement is METRIC -- it ran on
#: GT-depth clouds -- and does not transfer to the VGGT path as written. Re-expressed against
#: each scene's own GT cloud diagonal, 1.5 m is:
#:
#:     ebc200e928 0.4242 | 0cf2e9402d 0.2572 | e9e16b6043 0.2440 | e898c76c1f 0.1781
#:     0a76e06478 0.1729 | f6659a3107 0.1630 | 0b031f3119 0.1506 | e050c15a8d 0.0884
#:
#: i.e. a 4.8x spread, mean 0.2098, median 0.1755. **A single relative radius therefore
#: cannot reproduce 1.5 m on every scene** -- it is ~2x too generous on the smallest scene
#: and ~2x too tight on the largest. It is still the right form: with no metric scale on the
#: VGGT path, a fixed fraction of the diagonal at least holds the crop's share of the scene
#: constant, whereas a fixed VGGT radius would vary by the same 1.28-4.04 m/unit factor with
#: nothing to justify it. Frozen at the median. Every run records the per-scene diagonal and
#: the radius in VGGT units it resolved against.
CROP_REL = 0.175

#: Where the voxel merge happens: inside each prompt group's crop, not once over the scene.
#:
#: Measured on 0a76e06478 + ebc200e928 (13 objects) at the frozen stride/voxel_rel/crop_rel,
#: median ceiling IoU by GT area bin:
#:
#:     bin         n   global   crop_local
#:     10k-50k     3   0.9124   0.9386
#:     50k-200k    7   0.9238   0.9474
#:     >200k       3   0.9481   0.9686
#:     ALL        13   0.9238   0.9474   (min 0.7932 -> 0.8505)
#:
#: A crop holds a fraction of the scene's points, so `voxel_rel` x the *crop's* diagonal is a
#: finer absolute voxel than `voxel_rel` x the scene's, at a comparable per-call point count
#: (35k-140k voxels/call vs 18k-69k). It lifts every bin, including the sub-50k bin that the
#: Phase-5.2 gate failed on, which is the condition under which this lever was to be adopted.
#:
#: The cost is architectural: Stage A can no longer pre-voxelise. It hands Stage B the
#: un-voxelised stride grid, and Stage B voxelises once per prompt group.
VOXEL_SCOPE = "crop_local"

#: Which of Point-SAM's 3 multimask heads to keep. Phase 5.1b, 8 scenes / 51 objects:
#: `stability_score` 0.421 vs `argmax(iou_preds)` 0.140 uncropped (0.552 vs 0.430 cropped);
#: `iou_preds` picks its own best head on 4/51 objects, materially worse than the 1/3 you get
#: by guessing, while stability gets 29/51. Stage C's NMS ranks by the same score.
SELECTION_RULE = "stability_score"

#: Stage C output filtering: the floors SAM-V's own reported run applies to ITS OWN output
#: masks, mirrored exactly onto Point-SAM's.
#:
#: These are not the sam-hq *proposal* filters (`PROPOSAL_PRED_IOU_THRESH = 0.88`,
#: `PROPOSAL_STABILITY_SCORE_THRESH = 0.95`) -- those act on the AMG proposals that become
#: prompts, and Point-SAM already inherits them by construction, because Stage A prompts from
#: the same proposal cache. These are the *decoder output* floors: the benchmark's grid
#: defaults `--pred_iou_thresh_values [0.40]` and `--stability_score_thresh_values [0.2]`
#: (`sam_vggt_3dtracking_benchmark.py:1506-1509`), enforced in
#: `masks/automatic_mask_generator.py:366-367` and `:390-391`.
#:
#: The comparison operators and their ORDER are copied from there and matter:
#: `iou_preds > PRED_IOU_FLOOR` is **strict**, `stability_score >= STABILITY_FLOOR` is not, and
#: the IoU filter runs first. Applied before the NMS, as in the frozen generator.
#:
#: Without them Point-SAM entered Hungarian matching with 335 predictions against SAM-V's 121
#: for the same 7 GT objects, which is a precision penalty it does not deserve: SAM-V's row is
#: filtered and Point-SAM's was not. Measured effect on 0a76e06478: 353/381 masks survive the
#: IoU floor, and the stability floor removes nothing further (min stability 0.2792 > 0.2), so
#: this is a filter, not a cull.
PRED_IOU_FLOOR = 0.40
STABILITY_FLOOR = 0.2

#: Stage C panoramic NMS. The score is SELECTION_RULE, NOT `iou_preds` -- ranking the NMS by a
#: head score that loses to chance would undo the selection rule at the last step. The
#: threshold matches SAM-V's own `--box_nms_thresh_values [0.95]`.
NMS_SCORE = "stability_score"
NMS_IOU_THRESHOLD = 0.95

#: Logit offset for `calculate_stability_score`; SAM's own default, and the value the frozen
#: benchmark's PROPOSAL_STABILITY_SCORE_THRESH is calibrated against.
STABILITY_OFFSET = 1.0

#: Point-SAM colour normalisation. Upstream `evaluation/eval_kitti.py` maps to [-1, 1];
#: `evaluation/inference.py` only divides by 255 and is stale (its `model(**...)` call does
#: not even match `forward`'s signature). Phase 5.1 measured signed > unit.
COLOR_NORM = "signed"

#: Point-SAM patch encoder sizing, upstream's own recommendation for clouds over 100k points.
NUM_PATCHES = 2048
PATCH_SIZE = 256

#: Prompt generation -- the paper's "samprompt" configuration, so Point-SAM is prompted from
#: exactly the proposals SAM-V is prompted from. The PROPOSAL_* constants live in the frozen
#: `sam_vggt_3dtracking_benchmark` module and are imported from there, never copied.
PROMPT_SAMPLING_METHOD = "pole_plus_diverse"
PROMPT_POINTS_PER_MASK = 5

#: `np.random.default_rng` seed for prompt sampling; matches the frozen benchmark's own.
PROMPT_SEED = 0


def frozen_config() -> Dict[str, object]:
    """The frozen config as a dict, for provenance and for `tracks.json`'s `config` field."""
    return dict(
        stride=STRIDE, voxel_rel=VOXEL_REL, conf_thresh=CONF_THRESH, crop_rel=CROP_REL,
        voxel_scope=VOXEL_SCOPE, selection_rule=SELECTION_RULE,
        pred_iou_floor=PRED_IOU_FLOOR, stability_floor=STABILITY_FLOOR, nms_score=NMS_SCORE,
        nms_iou_threshold=NMS_IOU_THRESHOLD, stability_offset=STABILITY_OFFSET,
        color_norm=COLOR_NORM, num_patches=NUM_PATCHES, patch_size=PATCH_SIZE,
        prompt_sampling_method=PROMPT_SAMPLING_METHOD,
        prompt_points_per_mask=PROMPT_POINTS_PER_MASK, prompt_seed=PROMPT_SEED,
    )


# --------------------------------------------------------------------------------------
# VGGT geometry
# --------------------------------------------------------------------------------------
def load_native_rgb(image_paths: Sequence[Path]) -> torch.Tensor:
    """[N,3,H,W] float32 in [0,255] at the frames' native resolution."""
    arrs = []
    for p in image_paths:
        arrs.append(np.array(Image.open(p).convert("RGB"), dtype=np.float32))
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"frames disagree in size: {shapes}")
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def load_vggt(ckpt_path: str, device: str):
    """VGGT-1B, weights from the pre-staged checkpoint (no runtime download)."""
    from vggt.models.vggt import VGGT

    model = VGGT()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"VGGT checkpoint is missing {len(missing)} keys, e.g. {missing[:5]}")
    return model.to(device).eval()


def vggt_forward(model, images_native: torch.Tensor, device: str, target_size: int = 896
                 ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Run VGGT. Returns (world_points [N,S,S,3], world_points_conf [N,S,S], coords [N,6]).

    Preprocessing goes through the repo's own `SamVGGT.preprocess_vggt_images` so this path
    cannot drift from the model's. It is called unbound (`self=None`) deliberately: the method
    touches no attribute of `self`, and instantiating `SamVGGT` would needlessly build SAM.
    """
    from model.sam_vggt_model import SamVGGT

    imgs, coords = SamVGGT.preprocess_vggt_images(None, images_native, target_size=target_size)
    imgs = imgs.to(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            pred = model(imgs)
    wp = pred["world_points"][0].float().cpu().numpy()
    conf = pred["world_points_conf"][0].float().cpu().numpy()
    return wp, conf, coords


def world_points_to_grid(world_points: np.ndarray, conf: np.ndarray, coords: torch.Tensor,
                         out_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Undo the square padding, then resample onto the benchmark's evaluation grid.

    See the module docstring: this is the step the plan got wrong. `coords` is
    `[x1, y1, x2, y2, W, H]` in the padded/resized frame, as returned by
    `preprocess_vggt_images`.
    """
    n = world_points.shape[0]
    out_h, out_w = out_hw
    xyz_out = np.empty((n, out_h, out_w, 3), dtype=np.float32)
    conf_out = np.empty((n, out_h, out_w), dtype=np.float32)
    for i in range(n):
        x1, y1, x2, y2 = [float(v) for v in coords[i][:4]]
        r0, r1 = int(round(y1)), int(round(y2))
        c0, c1 = int(round(x1)), int(round(x2))
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise ValueError(f"degenerate content box for frame {i}: rows {r0}:{r1} cols {c0}:{c1}")
        xyz = torch.from_numpy(world_points[i, r0:r1, c0:c1, :]).permute(2, 0, 1).unsqueeze(0)
        cf = torch.from_numpy(conf[i, r0:r1, c0:c1]).unsqueeze(0).unsqueeze(0)
        xyz_out[i] = F.interpolate(xyz, size=out_hw, mode="bilinear", align_corners=False
                                   )[0].permute(1, 2, 0).numpy()
        conf_out[i] = F.interpolate(cf, size=out_hw, mode="bilinear", align_corners=False)[0, 0].numpy()
    return xyz_out, conf_out


# --------------------------------------------------------------------------------------
# sampling: stride grid -> voxel merge -> scatter back -> upsample
# --------------------------------------------------------------------------------------
def stride_subsample(xyz_grid: np.ndarray, conf_grid: np.ndarray, stride: int,
                     conf_thresh: float) -> Dict[str, np.ndarray]:
    """Take every `stride`-th pixel, drop those below `conf_thresh`.

    Returns `xyz` [M,3], `conf` [M], and `pixel` [M,3] as (frame, row, col) on the FULL grid.
    """
    n, h, w, _ = xyz_grid.shape
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xyz, conf, pix = [], [], []
    for f in range(n):
        v = conf_grid[f][rr, cc]
        keep = v >= conf_thresh
        xyz.append(xyz_grid[f][rr, cc][keep])
        conf.append(v[keep])
        pix.append(np.stack([np.full(keep.sum(), f), rr[keep], cc[keep]], axis=1))
    return dict(xyz=np.concatenate(xyz).astype(np.float32),
                conf=np.concatenate(conf).astype(np.float32),
                pixel=np.concatenate(pix).astype(np.int32),
                grid_shape=(len(rows), len(cols)))


def voxel_merge(xyz: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise to a voxel grid.

    Returns (`voxel_id` [M] mapping each point to its voxel, `voxel_xyz` [V,3] the mean point
    of each voxel). The one-to-many voxel->pixel map is `voxel_id` read backwards, which keeps
    the 3D->2D transfer a lookup rather than a rendering.
    """
    if voxel_size <= 0:
        return np.arange(len(xyz), dtype=np.int64), xyz.copy()
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, voxel_id = np.unique(keys, axis=0, return_inverse=True)
    n_vox = int(voxel_id.max()) + 1 if len(voxel_id) else 0
    sums = np.zeros((n_vox, 3), dtype=np.float64)
    counts = np.zeros(n_vox, dtype=np.int64)
    np.add.at(sums, voxel_id, xyz)
    np.add.at(counts, voxel_id, 1)
    return voxel_id.astype(np.int64), (sums / counts[:, None]).astype(np.float32)


def majority_label_per_voxel(voxel_id: np.ndarray, labels: np.ndarray, n_voxels: int
                             ) -> np.ndarray:
    """The single label each voxel carries: the most common label among its points.

    This is where the sampling path loses information, and is exactly what the ceiling
    measures — a voxel straddling an object boundary can only be all-object or all-not.
    """
    # Fast path: a single object's labels are binary, so the majority is a bincount
    # comparison. Exactly equivalent to the general path below, and ~1000x faster at the
    # 5x10^5-voxel settings the ceiling sweep explores.
    uniq = np.unique(labels)
    if len(uniq) <= 2 and set(uniq.tolist()) <= {0, 1}:
        ones = np.bincount(voxel_id, weights=labels.astype(np.float64), minlength=n_voxels)
        total = np.bincount(voxel_id, minlength=n_voxels)
        # ties go to 0, matching np.argmax's first-wins on a sorted unique([0,1])
        return (2 * ones > total).astype(labels.dtype)

    order = np.lexsort((labels, voxel_id))
    v_sorted, l_sorted = voxel_id[order], labels[order]
    out = np.zeros(n_voxels, dtype=labels.dtype)
    starts = np.flatnonzero(np.r_[True, np.diff(v_sorted) != 0])
    for s, e in zip(starts, np.r_[starts[1:], len(v_sorted)]):
        vals, cnts = np.unique(l_sorted[s:e], return_counts=True)
        out[v_sorted[s]] = vals[np.argmax(cnts)]
    return out


def scatter_to_full_masks(point_values: np.ndarray, pixel: np.ndarray, n_frames: int,
                          stride: int, out_hw: Tuple[int, int]) -> np.ndarray:
    """Per-point values -> per-frame full-resolution masks by nearest-upsampling the grid.

    Each retained pixel paints the `stride` x `stride` block it stands for, which is the
    nearest-neighbour upsample of the strided grid back to `out_hw`.
    """
    out_h, out_w = out_hw
    masks = np.zeros((n_frames, out_h, out_w), dtype=bool)
    sel = point_values.astype(bool)
    px = pixel[sel]
    for dr in range(stride):
        for dc in range(stride):
            rr = np.clip(px[:, 1] + dr, 0, out_h - 1)
            cc = np.clip(px[:, 2] + dc, 0, out_w - 1)
            masks[px[:, 0], rr, cc] = True
    return masks


# --------------------------------------------------------------------------------------
# cropping (shared by Stage B and the ceiling)
# --------------------------------------------------------------------------------------
def cloud_diagonal(xyz: np.ndarray) -> float:
    """Bbox diagonal of a point cloud, the unit CROP_REL and VOXEL_REL are fractions of.

    Same definition the Phase-5.2 ceiling used (`ceiling_for_scene`) and the same one the
    GT-depth probe printed, so the metres-to-relative conversion recorded next to CROP_REL
    compares like with like.
    """
    return float(np.linalg.norm(xyz.max(0) - xyz.min(0)))


def crop_indices(xyz: np.ndarray, anchor: np.ndarray, radius: float,
                 always_keep: Sequence[int] = ()) -> np.ndarray:
    """Indices of points within `radius` of `anchor`, plus `always_keep` (the prompts).

    `radius <= 0` means no crop. The prompts are forced in because a prompt that fell
    outside its own crop would be a silent no-op, and Point-SAM would segment from nothing.
    """
    if radius <= 0:
        return np.arange(len(xyz), dtype=np.int64)
    keep = np.flatnonzero(np.linalg.norm(xyz - anchor[None, :], axis=1) <= radius)
    if len(always_keep):
        keep = np.union1d(keep, np.asarray(always_keep, dtype=np.int64))
    return keep.astype(np.int64)


def prompt_group_cloud(xyz: np.ndarray, anchor: np.ndarray, crop_radius: float,
                       voxel_rel: float, voxel_scope: str = VOXEL_SCOPE,
                       global_voxel_id: np.ndarray = None,
                       global_voxel_xyz: np.ndarray = None,
                       always_keep: Sequence[int] = ()) -> Dict[str, np.ndarray]:
    """The exact cloud ONE Point-SAM call sees, plus the map to scatter its mask back.

    This is the single definition of "what Stage B feeds the model". `pointsam_infer.py`
    calls it per prompt group and `pointsam_ceiling.py` calls it per GT object, which is what
    makes the published ceiling this run's ceiling rather than a different measurement that
    happens to sit next to it.

    Returns
        `voxel_xyz`      [V,3] the points handed to Point-SAM, in world coordinates
        `point_index`    [M]   indices into the caller's full stride-grid point array that
                               this call can reach; everything else is unreachable and its
                               pixels stay 0
        `voxel_of_point` [M]   voxel id in 0..V-1 for each of those points -- read backwards,
                               this is the voxel -> pixel map, so the 3D->2D transfer is a
                               lookup and never a rendering
        `voxel_edge`     the voxel edge length actually used, in VGGT units

    `voxel_scope`:
      - `crop_local` (frozen) -- crop the point cloud, then voxelise inside the crop at
        `voxel_rel` x the crop's own diagonal. A crop holds a fraction of the scene's points,
        so the same relative voxel is a finer absolute voxel, which is where the ceiling gain
        comes from (see VOXEL_SCOPE).
      - `global` -- use a scene-wide voxelisation computed once (pass `global_voxel_id` /
        `global_voxel_xyz`) and merely crop it. Cheaper, coarser. Kept because it is the
        variant the Phase-5.2 ceilings were measured under.
    """
    if voxel_scope == "crop_local":
        keep = crop_indices(xyz, anchor, crop_radius, always_keep)
        local_diag = cloud_diagonal(xyz[keep]) if len(keep) else 0.0
        edge = voxel_rel * local_diag
        vid, vxyz = voxel_merge(xyz[keep], edge)
        return dict(voxel_xyz=vxyz, point_index=keep, voxel_of_point=vid, voxel_edge=edge)

    if voxel_scope != "global":
        raise ValueError(f"unknown voxel_scope {voxel_scope!r}")
    if global_voxel_id is None or global_voxel_xyz is None:
        raise ValueError("voxel_scope='global' needs global_voxel_id and global_voxel_xyz")

    if crop_radius <= 0:
        return dict(voxel_xyz=global_voxel_xyz, point_index=np.arange(len(xyz), dtype=np.int64),
                    voxel_of_point=global_voxel_id, voxel_edge=float("nan"))

    # Stage B crops the VOXEL cloud in this scope, so the radius test is on voxel centroids.
    in_crop = np.linalg.norm(global_voxel_xyz - anchor[None, :], axis=1) <= crop_radius
    kept_vox = np.flatnonzero(in_crop)
    remap = np.full(len(global_voxel_xyz), -1, dtype=np.int64)
    remap[kept_vox] = np.arange(len(kept_vox))
    point_index = np.flatnonzero(in_crop[global_voxel_id]).astype(np.int64)
    return dict(voxel_xyz=global_voxel_xyz[kept_vox], point_index=point_index,
                voxel_of_point=remap[global_voxel_id[point_index]], voxel_edge=float("nan"))


def voxel_mean_features(values: np.ndarray, voxel_of_point: np.ndarray, n_voxels: int
                        ) -> np.ndarray:
    """Mean of `values` [M,C] over the points of each voxel -> [V,C].

    Used for the voxel colours Point-SAM takes as its point features. The geometry half of the
    same reduction lives in `voxel_merge`, which returns each voxel's mean position.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    sums = np.zeros((n_voxels, vals.shape[1]), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.int64)
    np.add.at(sums, voxel_of_point, vals)
    np.add.at(counts, voxel_of_point, 1)
    return (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
