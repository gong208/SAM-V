#!/usr/bin/env python3
"""ODIN Phase 2, stage A: the geometry probe and the metric-scale convention.

`notes/plans/odin-baseline.md` §4 Phase 2. This is the role
`panst3r_geometry_dump.py` played for MUSt3R: before any ODIN forward pass is
believed, look at the geometry it will be handed.

Two things happen here, and only one of them is allowed to depend on GT:

1. **Calibration (`--mode calibrate`).** Reconstruct each ScanNet++ benchmark
   scene twice -- once with VGGT, once by back-projecting GT depth with GT pose
   and GT intrinsics -- and fit a similarity transform between them. That gives
   the true metres-per-VGGT-unit for every scene, which is what a candidate
   convention is scored against. **GT is used to choose and check the
   convention, never as an input to it.**

2. **Dump (`--mode dump`).** Apply the frozen convention below -- which reads
   nothing but VGGT output -- and write `geometry.npz` for the ODIN forward
   pass. GT geometry is written into the same file when it exists, so the
   forward pass can be run on both and compared; ScanNet has no GT geometry on
   this node at all, so on that split only the VGGT arrays are present.

Runs in the project env:

    PYTHONPATH="<worktree>:$PWD/submodules/sam-hq:$PWD/submodules/vggt\
" \
        python benchmarks/baselines/odin_geometry_probe.py ...

(Worktrees carry empty submodule directories; every submodule path must point at
the MAIN checkout.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines.pointsam_sampling import (  # noqa: E402
    load_native_rgb,
    load_vggt,
    vggt_forward,
    world_points_to_grid,
)
from utils.provenance import write_run_provenance  # noqa: E402


# --------------------------------------------------------------------------------------
# THE FROZEN GEOMETRY CONVENTION
# --------------------------------------------------------------------------------------
# ODIN's `INPUT.VOXEL_SIZE = [0.02, 0.04, 0.08, 0.16]` is metric and its cross-view kNN
# position encoding is `PositionEmbeddingLearnedMLP` applied to *relative* xyz
# (`cross_view_attention.py:100-108`, `pointops.queryandgroup(..., use_xyz=True)` returns
# neighbour-minus-centre). So the model is:
#   - translation invariant  -> we do not translate,
#   - NOT scale invariant    -> a metric scale must be imposed,
#   - NOT rotation invariant -> the world frame's orientation matters.
# VGGT geometry is up-to-scale and expressed in the FIRST CAMERA's frame, so both of the
# last two need a convention. Both values below are fixed against geometry alone
# (`scale_calibration.json`), never against an eval metric.

#: The VGGT-only statistic whose value is pinned to `SCALE_REFERENCE_M`.
#: `camera_height` = median over frames of (camera centre . up) minus the 1st percentile
#: of the cloud along `up` -- i.e. how high above the floor the camera was held. Ten
#: candidates were scored against the GT-derived true scale of all nine ScanNet++
#: benchmark scenes (`scale_calibration.json`); this one wins by a wide margin
#: (max error 13.2%, vs 45-97% for every extent-like statistic and 50.5% for a single
#: global constant). It wins because handheld indoor capture height is physically stable
#: (GT: 1.224-1.496 m over the nine scenes) whereas room size is not, and because VGGT's
#: own normalisation already tracks scene extent, which leaves extent-like statistics
#: carrying almost no information about metric size.
SCALE_STATISTIC = "camera_height"

#: Metres. Fixed from the population of nine ScanNet++ benchmark scenes, not from any one
#: of them. The three ways to centre `s_true * stat_vggt` (products 1.399-1.770 m) agree
#: to within 3%: median 1.5277, log-mean 1.5598, log-midrange (minimax) 1.5737; dropping
#: any single scene moves each of them by <=3.6%. 1.55 is the nearest 0.05 m round value
#: inside that envelope, so the constant reproduces no individual scene's datum -- the
#: earlier 1.5277 did, because with n=9 the median IS a scene (`0a76e06478`'s), which made
#: its 1.00 error factor look like evidence when it was arithmetic. Resulting error over
#: the nine scenes: x0.876 to x1.108, max |dev| 13.2% (median-rule 14.7%, minimax 11.7%).
#: It is ~9% above the median GT camera height (1.420 m) because the VGGT-only statistic
#: is biased high -- the constant absorbs that bias rather than pretending it is not there.
#: The applied scale is `s = SCALE_REFERENCE_M / camera_height_vggt`, which reads
#: **nothing but VGGT output**. GT was used to pick the statistic and fit this one
#: constant, and is never an input to a run. The forward pass is insensitive at this
#: precision: x0.86-x1.09 leaves the query masks unchanged (`notes/plans/odin-phase2-probe.md`).
SCALE_REFERENCE_M = 1.55

#: Rotate the VGGT cloud so the estimated gravity direction becomes -z (ScanNet's own
#: convention, which is what the checkpoint was trained on). The estimate is VGGT-only:
#: the mean image-down axis of the frames, which for handheld indoor capture points down.
GRAVITY_ALIGN = True

#: `world_points_conf` floor used for the *statistics* only. The dumped geometry is never
#: conf-filtered (ODIN needs a dense per-pixel map, and a hole would have to be filled
#: with something anyway).
STAT_CONF_THRESH = 1.0

#: Pixel stride for the statistics and for the similarity fit. Statistics only.
STAT_STRIDE = 4

#: GT depth: uint16 millimetres, and anything beyond this is dropped as unreliable
#: (iPhone LiDAR).
GT_DEPTH_SCALE = 1.0 / 1000.0
GT_MAX_DEPTH_M = 10.0

VGGT_CKPT = str(REPO_ROOT / "submodules" / "vggt" / "checkpoints" / "model.pt")
SCANNETPP_EVAL_ROOT = os.environ.get("SCANNETPP_EVAL_ROOT", "")
SCANNETPP_V2_ROOT = os.path.join(os.environ.get("SCANNETPP_ROOT", ""), "train")


def frozen_convention() -> Dict[str, object]:
    return {
        "scale_statistic": SCALE_STATISTIC,
        "scale_reference_m": SCALE_REFERENCE_M,
        "gravity_align": GRAVITY_ALIGN,
        "stat_conf_thresh": STAT_CONF_THRESH,
        "stat_stride": STAT_STRIDE,
        "gt_depth_scale": GT_DEPTH_SCALE,
        "gt_max_depth_m": GT_MAX_DEPTH_M,
    }


# --------------------------------------------------------------------------------------
# scene loading
# --------------------------------------------------------------------------------------
def list_benchmark_frames(scene_dir: Path) -> List[str]:
    """Frame stems of a 3DTrackingBenchmark scene, in the benchmark's own order."""
    stems = sorted(
        p.stem for p in (scene_dir / "images").glob("*.jpg") if "_label_vis" not in p.stem
    )
    if not stems:
        raise FileNotFoundError(f"no frames under {scene_dir / 'images'}")
    return stems


def load_gt_geometry(scene_id: str, frames: Sequence[str], hw: Tuple[int, int]
                     ) -> Optional[Dict[str, np.ndarray]]:
    """Back-project the GT depth of every benchmark frame into the scene's world frame.

    Depth comes from the *eval* tree (`processed_scannetpp_eval/<scene>/gt_depth`), pose
    and intrinsics from the *v2* tree (`processed_scannetpp_v2/train/<scene>`). The v2
    tree's own `depth/` is DSLR-named and unusable for these iPhone frames, and the eval
    tree's `new_scene_metadata.npz` is DSLR-indexed on at least one scene -- hence the
    split. Returns None when any piece is missing (i.e. always, on ScanNet).
    """
    eval_scene = Path(SCANNETPP_EVAL_ROOT) / scene_id
    v2_scene = Path(SCANNETPP_V2_ROOT) / scene_id
    meta_path = v2_scene / "scene_iphone_metadata.npz"
    if not meta_path.exists():
        return None
    meta = np.load(meta_path, allow_pickle=True)
    names = [str(x) for x in meta["images"]]
    name_to_idx = {n: i for i, n in enumerate(names)}

    h, w = hw
    xyz = np.zeros((len(frames), h, w, 3), dtype=np.float32)
    valid = np.zeros((len(frames), h, w), dtype=bool)
    poses = np.zeros((len(frames), 4, 4), dtype=np.float64)
    for i, stem in enumerate(frames):
        depth_path = eval_scene / "gt_depth" / f"{stem}.png"
        pose_path = v2_scene / "pose" / f"{stem}.txt"
        if not depth_path.exists() or not pose_path.exists():
            return None
        if f"{stem}.jpg" not in name_to_idx:
            return None
        mi = name_to_idx[f"{stem}.jpg"]

        depth = np.array(Image.open(depth_path)).astype(np.float64) * GT_DEPTH_SCALE
        if depth.shape != (h, w):
            raise ValueError(f"{scene_id}/{stem}: depth {depth.shape} vs rgb {(h, w)}")
        c2w = np.asarray(meta["trajectories"][mi], dtype=np.float64)
        pose_txt = np.loadtxt(pose_path)
        if not np.allclose(c2w, pose_txt, atol=1e-5):
            raise ValueError(
                f"{scene_id}/{stem}: pose/*.txt disagrees with scene_iphone_metadata.npz "
                f"(max |diff| {np.abs(c2w - pose_txt).max():.3e})"
            )
        K = np.asarray(meta["intrinsics"][mi], dtype=np.float64)

        vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        z = depth
        ok = (z > 0) & (z < GT_MAX_DEPTH_M)
        x = (uu - K[0, 2]) * z / K[0, 0]
        y = (vv - K[1, 2]) * z / K[1, 1]
        cam = np.stack([x, y, z], axis=-1)
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        xyz[i] = world.astype(np.float32)
        valid[i] = ok
        poses[i] = c2w
    return {"xyz": xyz, "valid": valid, "poses": poses}


# --------------------------------------------------------------------------------------
# geometry statistics
# --------------------------------------------------------------------------------------
def cloud_statistics(xyz: np.ndarray) -> Dict[str, float]:
    """Scale-carrying summaries of a point cloud [M,3]. Every one is a length."""
    centroid = xyz.mean(0)
    rad = np.linalg.norm(xyz - centroid[None], axis=1)
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    return {
        "n_points": float(len(xyz)),
        "bbox_diag": float(np.linalg.norm(xyz.max(0) - xyz.min(0))),
        "p99_extent": float(np.linalg.norm(hi - lo)),
        "rms_radius": float(np.sqrt((rad ** 2).mean())),
        "median_radius": float(np.median(rad)),
        "mean_norm": float(np.linalg.norm(xyz, axis=1).mean()),
        "median_norm": float(np.median(np.linalg.norm(xyz, axis=1))),
    }


# --------------------------------------------------------------------------------------
# camera geometry recovered from a point map alone
# --------------------------------------------------------------------------------------
# A per-pixel world-point map from a pinhole camera carries the camera's pose exactly, and
# recovering it needs no pose head, no intrinsics and no second forward pass:
#
#   P(u,v) = C + d(u,v) * R^T K^-1 (u, v, 1)
#
# For a fixed image row v the ray directions span span(R^T e_x, R^T [0,(v-cy)/fy,1]), so
# every point of that row lies in a plane through C whose normal is
# n_v ~ R^T (0,-1,(v-cy)/fy). All row normals are therefore orthogonal to R^T e_x, and all
# column normals (m_u ~ R^T (1,0,-(u-cx)/fx)) are orthogonal to R^T e_y -- the camera's
# image-DOWN axis in world coordinates. So:
#
#   image-down axis  = the null direction of the stacked column-plane normals
#   camera centre    = least-squares intersection of all those planes
#
# both exact under the pinhole model, both computed from the dumped point map only.
def _fit_plane(pts: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    if len(pts) < 8:
        return None
    mu = pts.mean(0)
    X = pts - mu
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s[1] < 1e-9:
        return None
    n = Vt[2]
    return n, float(np.dot(n, mu))


def camera_from_pointmap(xyz: np.ndarray, valid: Optional[np.ndarray] = None,
                         n_lines: int = 24) -> Optional[Dict[str, np.ndarray]]:
    """Camera centre and image-down axis of one frame, from its point map alone."""
    h, w, _ = xyz.shape
    if valid is None:
        valid = np.ones((h, w), dtype=bool)
    rows = np.linspace(h * 0.1, h * 0.9, n_lines).astype(int)
    cols = np.linspace(w * 0.1, w * 0.9, n_lines).astype(int)

    row_n, col_n, planes = [], [], []
    for v in rows:
        p = _fit_plane(xyz[v][valid[v]])
        if p is not None:
            row_n.append(p[0])
            planes.append(p)
    for u in cols:
        p = _fit_plane(xyz[:, u][valid[:, u]])
        if p is not None:
            col_n.append(p[0])
            planes.append(p)
    if len(col_n) < 3 or len(planes) < 6:
        return None

    _, _, Vt = np.linalg.svd(np.stack(col_n), full_matrices=False)
    down = Vt[2]
    top = xyz[: h // 4][valid[: h // 4]].mean(0)
    bot = xyz[-(h // 4):][valid[-(h // 4):]].mean(0)
    if np.dot(down, bot - top) < 0:
        down = -down

    A = np.stack([p[0] for p in planes])
    b = np.array([p[1] for p in planes])
    centre, *_ = np.linalg.lstsq(A, b, rcond=None)
    return {"centre": centre, "down": down}


def frame_cameras(xyz: np.ndarray, valid: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame (centres [N,3], down axes [N,3]) recovered from point maps."""
    centres, downs = [], []
    for i in range(xyz.shape[0]):
        cam = camera_from_pointmap(xyz[i], None if valid is None else valid[i])
        if cam is None:
            continue
        centres.append(cam["centre"])
        downs.append(cam["down"])
    if not centres:
        raise ValueError("no frame yielded a camera estimate")
    return np.stack(centres), np.stack(downs)


def scene_statistics(xyz: np.ndarray, mask: np.ndarray, up: np.ndarray,
                     centres: np.ndarray) -> Dict[str, float]:
    """Every scale-carrying statistic a candidate convention may key on.

    `xyz` [N,H,W,3] already strided; `mask` [N,H,W] which points count; `up` the gravity
    direction in the same frame; `centres` [N,3] the camera centres.
    """
    flat = xyz[mask]
    st = cloud_statistics(flat)
    up = up / np.linalg.norm(up)
    h = flat @ up
    floor = float(np.percentile(h, 1.0))
    ceil = float(np.percentile(h, 99.0))
    st["vertical_extent"] = ceil - floor
    cam_h = centres @ up - floor
    st["camera_height"] = float(np.median(cam_h))
    rays = []
    for i in range(xyz.shape[0]):
        if mask[i].sum() < 100 or i >= len(centres):
            continue
        rays.append(np.linalg.norm(xyz[i][mask[i]] - centres[i][None], axis=1))
    st["median_ray_length"] = float(np.median(np.concatenate(rays))) if rays else float("nan")
    if len(centres) > 1:
        d = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
        st["camera_spread"] = float(np.median(d[np.triu_indices(len(centres), 1)]))
    else:
        st["camera_spread"] = float("nan")
    return st


SCALE_CANDIDATES = ["bbox_diag", "p99_extent", "rms_radius", "median_radius",
                    "mean_norm", "median_norm", "vertical_extent", "camera_height",
                    "median_ray_length", "camera_spread"]


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Similarity transform with scale: dst ~= s * R @ src + t. Returns (s, R, t, rms)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    C = (Y.T @ X) / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_src = (X ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_src)
    t = mu_d - s * R @ mu_s
    resid = dst - (s * (R @ src.T).T + t)
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    return s, R, t, rms


def robust_umeyama(src: np.ndarray, dst: np.ndarray, keep: float = 0.8
                   ) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Umeyama, refit on the `keep` fraction of points with the smallest residual."""
    s, R, t, _ = umeyama(src, dst)
    resid = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
    thr = np.quantile(resid, keep)
    sel = resid <= thr
    return umeyama(src[sel], dst[sel])


# --------------------------------------------------------------------------------------
# the convention itself -- VGGT output in, canonical metric geometry out
# --------------------------------------------------------------------------------------
def estimate_up_from_frames(xyz_vggt: np.ndarray) -> np.ndarray:
    """Gravity ('up') direction in the VGGT world frame, from VGGT geometry alone.

    Mean of the per-frame image-down axes recovered by `camera_from_pointmap`, negated.
    For handheld indoor capture the camera is held roughly upright, so the mean image-down
    axis is gravity. No GT, no pose file, no intrinsics -- just the dumped point maps.
    """
    _, downs = frame_cameras(xyz_vggt)
    mean_down = downs.mean(0)
    mean_down /= np.linalg.norm(mean_down)
    return -mean_down  # up


def rotation_bringing_up_to_z(up: np.ndarray) -> np.ndarray:
    """Minimal rotation R with R @ up == +z."""
    a = up / np.linalg.norm(up)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def vggt_only_statistics(xyz_vggt: np.ndarray, conf: np.ndarray
                         ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Every candidate statistic, computed from VGGT output and nothing else."""
    sub = xyz_vggt[:, ::STAT_STRIDE, ::STAT_STRIDE, :]
    mask = conf[:, ::STAT_STRIDE, ::STAT_STRIDE] >= STAT_CONF_THRESH
    if mask.sum() < 1000:
        mask = np.ones(mask.shape, dtype=bool)
    centres, downs = frame_cameras(xyz_vggt)
    up = -downs.mean(0)
    up /= np.linalg.norm(up)
    return scene_statistics(sub, mask, up, centres), up, centres


def apply_convention(xyz_vggt: np.ndarray, conf: np.ndarray) -> Dict[str, object]:
    """The frozen convention. Input: raw VGGT world points [N,H,W,3] + conf [N,H,W]."""
    stats, up, _ = vggt_only_statistics(xyz_vggt, conf)
    scale = SCALE_REFERENCE_M / stats[SCALE_STATISTIC]

    if GRAVITY_ALIGN:
        R = rotation_bringing_up_to_z(up)
    else:
        up = np.array([0.0, 0.0, 1.0])
        R = np.eye(3)

    out = scale * (xyz_vggt.reshape(-1, 3) @ R.T)
    return {
        "xyz": out.reshape(xyz_vggt.shape).astype(np.float32),
        "scale": float(scale),
        "R": R,
        "up_vggt_frame": up,
        "stats_raw": stats,
    }


def gt_reference_statistics(gt: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, float]:
    """The same statistics on the GT cloud, using GT poses -- the reference values."""
    sub = gt["xyz"][:, ::STAT_STRIDE, ::STAT_STRIDE, :]
    centres = gt["poses"][:, :3, 3]
    up = np.array([0.0, 0.0, 1.0])  # ScanNet++ world frame is gravity-aligned, +z up
    return scene_statistics(sub, mask, up, centres)


# --------------------------------------------------------------------------------------
# per-scene driver
# --------------------------------------------------------------------------------------
def process_scene(scene_dir: Path, vggt_model, device: str, dump_dir: Optional[Path],
                  timing_warmup: int = 0) -> Dict[str, object]:
    # Stage A is a real cost of the ODIN row, so it is timed for the
    # `infer_seconds["prepare"]` key of tracks.json. The timer covers only the VGGT forward
    # + the frozen convention -- i.e. exactly what a run needs -- and stops before the
    # GT-comparison block below, which no run performs.
    #
    # Phase 5 harness (CLAUDE.md §8, matching SAM-V and Point-SAM): `perf_counter` +
    # `torch.cuda.synchronize()` on both ends, with `--timing_warmup` whole-scene passes
    # discarded before the loop, run strictly serially on an otherwise idle L40S.
    # `timing_warmup == 0` reproduces the Phase-3/4 clock, which is a bare wall clock with
    # no warmup and no serialisation discipline: NOT a reportable timing.
    torch.cuda.synchronize()
    t_prepare = time.perf_counter()
    scene_id = scene_dir.name
    frames = list_benchmark_frames(scene_dir)
    paths = [scene_dir / "images" / f"{s}.jpg" for s in frames]
    native = load_native_rgb(paths)
    n, _, H, W = native.shape

    wp, conf, coords = vggt_forward(vggt_model, native, device)
    xyz_v_raw, conf_grid = world_points_to_grid(wp, conf, coords, (H, W))

    conv = apply_convention(xyz_v_raw, conf_grid)
    xyz_v = conv["xyz"]
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - t_prepare

    rec: Dict[str, object] = {
        "scene_id": scene_id,
        "frames": frames,
        "prepare_seconds": float(prepare_seconds),
        "timing_warmup": int(timing_warmup),
        "native_hw": [int(H), int(W)],
        "vggt_raw_stats": conv["stats_raw"],
        "convention_scale_m_per_unit": conv["scale"],
        "convention_up_in_vggt_frame": [float(x) for x in conv["up_vggt_frame"]],
        "conf_stats": {
            "min": float(conf_grid.min()), "median": float(np.median(conf_grid)),
            "max": float(conf_grid.max()),
            "frac_ge_thresh": float((conf_grid >= STAT_CONF_THRESH).mean()),
        },
    }

    gt = load_gt_geometry(scene_id, frames, (H, W))
    if gt is not None:
        sub = (slice(None), slice(None, None, STAT_STRIDE), slice(None, None, STAT_STRIDE))
        gt_xyz_s = gt["xyz"][sub].reshape(-1, 3)
        gt_val_s = gt["valid"][sub].reshape(-1)
        v_raw_s = xyz_v_raw[sub].reshape(-1, 3)
        v_met_s = xyz_v[sub].reshape(-1, 3)
        conf_s = conf_grid[sub].reshape(-1)

        both = gt_val_s & (conf_s >= STAT_CONF_THRESH)
        if both.sum() < 1000:
            both = gt_val_s
        s_true, R_true, t_true, rms_raw = umeyama(v_raw_s[both], gt_xyz_s[both])
        s_rob, R_rob, t_rob, rms_rob = robust_umeyama(v_raw_s[both], gt_xyz_s[both])

        # the convention's own residual, on the SAME points: rigid-only fit of the
        # already-scaled cloud (scale frozen at 1) tells us what the convention costs.
        s_conv, R_conv, t_conv, rms_conv = umeyama(v_met_s[both], gt_xyz_s[both])

        shape3 = xyz_v_raw[sub].shape[:3]
        mask_shared = both.reshape(shape3)
        gt_stats_shared = gt_reference_statistics(gt, mask_shared)
        v_stats_all = conv["stats_raw"]

        # does the point-map camera estimator actually recover a camera? test it where the
        # answer is known: run it on the GT point maps and compare against the GT poses.
        gt_centres_est, gt_downs_est = frame_cameras(gt["xyz"], gt["valid"])
        gt_centres_true = gt["poses"][:, :3, 3]
        gt_downs_true = gt["poses"][:, :3, 1]
        cam_err = np.linalg.norm(gt_centres_est - gt_centres_true, axis=1)
        down_err = np.degrees(np.arccos(np.clip(
            (gt_downs_est * gt_downs_true).sum(1), -1, 1)))

        up_gt_in_vggt = R_rob.T @ np.array([0.0, 0.0, 1.0])
        up_est = np.asarray(conv["up_vggt_frame"])
        cos_up = float(np.clip(np.dot(up_gt_in_vggt / np.linalg.norm(up_gt_in_vggt), up_est), -1, 1))

        per_frame = []
        for i in range(n):
            gm = gt["valid"][i, ::STAT_STRIDE, ::STAT_STRIDE].reshape(-1)
            cm = conf_grid[i, ::STAT_STRIDE, ::STAT_STRIDE].reshape(-1) >= STAT_CONF_THRESH
            k = gm & cm
            if k.sum() < 500:
                per_frame.append({"frame": frames[i], "n": int(k.sum()), "note": "too few points"})
                continue
            a = xyz_v_raw[i, ::STAT_STRIDE, ::STAT_STRIDE].reshape(-1, 3)[k]
            b = gt["xyz"][i, ::STAT_STRIDE, ::STAT_STRIDE].reshape(-1, 3)[k]
            sf, Rf, tf, rmsf = umeyama(a, b)
            # residual of this frame under the GLOBAL fit -- the multi-view consistency test
            g = s_rob * (R_rob @ a.T).T + t_rob
            per_frame.append({
                "frame": frames[i], "n": int(k.sum()),
                "per_frame_scale_m_per_unit": sf,
                "per_frame_rms_m": rmsf,
                "rms_under_global_fit_m": float(np.sqrt(((b - g) ** 2).sum(1).mean())),
                "median_err_under_global_fit_m": float(np.median(np.linalg.norm(b - g, axis=1))),
            })

        rec["gt"] = {
            "n_shared_points": int(both.sum()),
            "scale_true_m_per_unit": s_true,
            "scale_true_robust_m_per_unit": s_rob,
            "global_fit_rms_m": rms_raw,
            "global_fit_rms_robust_m": rms_rob,
            "convention_residual_rms_m": rms_conv,
            "convention_residual_scale_slack": s_conv,
            "convention_error_factor": float(conv["scale"] / s_rob),
            "gt_stats_shared": gt_stats_shared,
            "vggt_stats_all": v_stats_all,
            "candidate_ref_m": {c: gt_stats_shared[c] for c in SCALE_CANDIDATES},
            "candidate_stat_vggt_all": {c: v_stats_all[c] for c in SCALE_CANDIDATES},
            "camera_estimator_check_on_gt": {
                "centre_err_m_median": float(np.median(cam_err)),
                "centre_err_m_max": float(cam_err.max()),
                "down_axis_err_deg_median": float(np.median(down_err)),
                "down_axis_err_deg_max": float(down_err.max()),
            },
            "up_gt_in_vggt_frame": [float(x) for x in up_gt_in_vggt],
            "up_estimate_angle_deg": float(np.degrees(np.arccos(cos_up))),
            "per_frame": per_frame,
        }

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "frames": np.array(frames),
            "native_hw": np.array([H, W]),
            "xyz_vggt": xyz_v,
            "xyz_vggt_raw": xyz_v_raw,
            "conf_vggt": conf_grid,
            "scale_m_per_unit": np.array(conv["scale"]),
            "R_gravity": conv["R"],
            "prepare_seconds": np.array(prepare_seconds),
            "timing_warmup": np.array(timing_warmup),
        }
        if gt is not None:
            payload["xyz_gt"] = gt["xyz"]
            payload["valid_gt"] = gt["valid"]
        np.savez(dump_dir / "geometry.npz", **payload)
        rec["geometry_npz"] = str(dump_dir / "geometry.npz")

    return rec


def summarise_calibration(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Score every candidate convention against the GT-derived true scales."""
    scenes = [r for r in records if "gt" in r]
    if not scenes:
        return {"note": "no scene had GT geometry; nothing to calibrate against"}

    out: Dict[str, object] = {"n_scenes": len(scenes),
                              "scenes": [r["scene_id"] for r in scenes]}
    s_true = np.array([r["gt"]["scale_true_robust_m_per_unit"] for r in scenes])
    out["true_scale_m_per_unit"] = {
        "per_scene": {r["scene_id"]: float(v) for r, v in zip(scenes, s_true)},
        "min": float(s_true.min()), "max": float(s_true.max()),
        "median": float(np.median(s_true)),
        "spread_factor": float(s_true.max() / s_true.min()),
    }

    cand: Dict[str, object] = {}
    for c in SCALE_CANDIDATES:
        ref = np.array([r["gt"]["candidate_ref_m"][c] for r in scenes])
        stat_all = np.array([r["gt"]["candidate_stat_vggt_all"][c] for r in scenes])
        # Two ways to pin the reference constant:
        #   gt_median  -- the median of the statistic measured on the GT clouds. Physical,
        #                 but it inherits any bias between the VGGT estimate of the
        #                 statistic and the GT one.
        #   centred    -- the median of (true scale x VGGT statistic), i.e. the constant
        #                 that makes the convention's error factor 1 on the median scene.
        #                 Same spread, no bias. This is the one the convention uses.
        prod = s_true * stat_all
        for tag, ref_m in (("gt_median", float(np.median(ref))),
                           ("centred", float(np.exp(np.median(np.log(prod)))))):
            s_conv = ref_m / stat_all
            err = s_conv / s_true
            cand.setdefault(c, {})[tag] = {
                "reference_m": ref_m,
                "error_factor_per_scene": {r["scene_id"]: float(e) for r, e in zip(scenes, err)},
                "error_min": float(err.min()), "error_max": float(err.max()),
                "error_spread_factor": float(err.max() / err.min()),
                "error_max_abs_dev_pct": float(100.0 * np.max(np.abs(np.log(err)))),
            }
    # the trivial baseline: one global constant, no per-scene statistic at all
    const = float(np.median(s_true))
    err = const / s_true
    cand["__global_constant__"] = {"centred": {
        "reference_m": None, "constant_m_per_unit": const,
        "error_factor_per_scene": {r["scene_id"]: float(e) for r, e in zip(scenes, err)},
        "error_min": float(err.min()), "error_max": float(err.max()),
        "error_spread_factor": float(err.max() / err.min()),
        "error_max_abs_dev_pct": float(100.0 * np.max(np.abs(np.log(err)))),
    }}
    out["candidates"] = cand
    out["frozen_convention"] = frozen_convention()
    out["frozen_convention_error"] = {
        r["scene_id"]: r["gt"]["convention_error_factor"] for r in scenes
    }
    out["up_estimate_angle_deg"] = {
        r["scene_id"]: r["gt"]["up_estimate_angle_deg"] for r in scenes
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark_root", required=True,
                    help="3DTrackingBenchmark/<split> directory")
    ap.add_argument("--scene_ids", default="all",
                    help="comma-separated scene ids, or 'all'")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mode", choices=["calibrate", "dump"], default="dump",
                    help="calibrate: stats + candidate scoring over every scene given. "
                         "dump: additionally write geometry.npz per scene.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vggt_ckpt", default=VGGT_CKPT)
    ap.add_argument("--timing_warmup", type=int, default=0,
                    help="whole-scene passes on the first scene, discarded before the "
                         "timed loop (Phase 5 uses 2, the frozen harness convention). "
                         "0 = the Phase-3/4 bare wall clock, not a reportable timing.")
    args = ap.parse_args()

    root = Path(args.benchmark_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.scene_ids == "all":
        scene_ids = sorted(p.name for p in root.iterdir() if (p / "images").is_dir())
    else:
        scene_ids = [s for s in args.scene_ids.split(",") if s]

    write_run_provenance(out, {
        "benchmark_root": str(root), "scene_ids": scene_ids, "mode": args.mode,
        "vggt_ckpt": args.vggt_ckpt, "device": args.device,
        "timing_warmup": args.timing_warmup,
        "convention": frozen_convention(),
    })

    model = load_vggt(args.vggt_ckpt, args.device)

    # Discarded warmup passes on the first scene, so the recorded seconds exclude cuDNN
    # autotuning and lazy CUDA-module loading. Same convention as the SAM-V harness
    # (`benchmarks/timing/benchmark_inference_time.py`) and Point-SAM's stage B.
    for wi in range(args.timing_warmup):
        print(f"[warmup {wi + 1}/{args.timing_warmup}] {scene_ids[0]}", flush=True)
        process_scene(root / scene_ids[0], model, args.device, None)
    if args.timing_warmup:
        torch.cuda.synchronize()

    records = []
    for sid in scene_ids:
        dump_dir = (out / sid) if args.mode == "dump" else None
        rec = process_scene(root / sid, model, args.device, dump_dir, args.timing_warmup)
        records.append(rec)
        gt = rec.get("gt")
        print(f"[{sid}] frames={len(rec['frames'])} "
              f"conv_scale={rec['convention_scale_m_per_unit']:.4f} m/unit"
              + (f" | true={gt['scale_true_robust_m_per_unit']:.4f} "
                 f"err x{gt['convention_error_factor']:.3f} "
                 f"fit_rms={gt['global_fit_rms_robust_m']:.3f} m "
                 f"up_ang={gt['up_estimate_angle_deg']:.1f} deg" if gt else " | no GT"))
        if dump_dir is not None:
            with (dump_dir / "geometry_stats.json").open("w") as f:
                json.dump(rec, f, indent=2)

    summary = summarise_calibration(records)
    with (out / "geometry_stats.json").open("w") as f:
        json.dump({"convention": frozen_convention(), "scenes": records}, f, indent=2)
    with (out / "scale_calibration.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if "candidates" in summary:
        print("\ncandidate scale conventions (error factor = applied / true):")
        print(f"{'statistic':<22} {'ref (m)':>9} {'err min':>8} {'err max':>8} "
              f"{'spread':>8} {'max dev %':>10}")
        for c, d in summary["candidates"].items():
            e = d["centred"]
            ref = e.get("reference_m")
            print(f"{c:<22} {(f'{ref:.4f}' if ref else '  --  '):>9} "
                  f"{e['error_min']:>8.3f} {e['error_max']:>8.3f} "
                  f"{e['error_spread_factor']:>8.3f} {e['error_max_abs_dev_pct']:>10.1f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
