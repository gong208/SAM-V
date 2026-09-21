#!/usr/bin/env python3
"""
Baseline comparison pipeline: SAM2 vs SamVGGT on Hypersim.

For each scene/object in a split:
1) sample 16-32 frames by two strategies:
   - pose_near (nearest-point sampling in camera translation space)
   - pose_diverse (farthest-point sampling in camera translation space)
2) generate identical point prompts from GT masks
3) run both models with the same sampled frames/prompts
4) compute per-frame IoU and aggregate to object/scene/global averages
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i python benchmarks/compare_baseline_sam2.py \
    --dataset_root "$HYPERSIM_ROOT" \
    --split test \
    --sam_v_ckpt /path/to/sam_v.pth \
    --sam2_ckpt submodules/sam2/checkpoints/sam2.1_hiera_large.pt \
    --num_shards 8 --shard_id $i \
    --output_dir "results/sam2_baseline/shard_$i" > "results/sam2_baseline/shard_$i.log" 2>&1 & done
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import zlib
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
SAM2_ROOT = REPO_ROOT / "submodules" / "sam2"
SAM_HQ_ROOT = REPO_ROOT / "submodules" / "sam-hq"
VGGT_ROOT = REPO_ROOT / "submodules" / "vggt"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))
if str(SAM_HQ_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_HQ_ROOT))
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from utils.checkpoint import is_partial_checkpoint, load_partial_checkpoint
from utils.provenance import write_run_provenance

TARGET_SIZE = (1024, 1024)  # (H, W)
# IoU thresholds for per-frame detection-style precision/recall.
IOU_THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samvggt_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if is_partial_checkpoint(ckpt):
        # A released checkpoint, or one this repo's trainer wrote. The
        # shape-filtering path below loads non-strictly and would accept a damaged
        # one silently, so these go through the verifying loader instead.
        load_partial_checkpoint(model, ckpt)
        model.to(device).eval()
        return
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Filter out SAM parameters whose shapes don't match the current model
    # (e.g. checkpoint trained with vit_l but model now uses vit_h).
    # SAM weights are already loaded from the correct pretrained checkpoint
    # during build_sam_vggt, so skipping them here is safe.
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        else:
            skipped.append(k)
    if skipped:
        print(f"[load_samvggt_checkpoint] Skipped {len(skipped)} keys with shape/name mismatch:")
        for k in skipped:
            print(f"  {k}")
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()


def _find_first_existing(scene_dir: Path, names: list[str], expect_dir: bool = True) -> Path | None:
    for n in names:
        p = scene_dir / n
        if expect_dir and p.is_dir():
            return p
        if not expect_dir and p.is_file():
            return p
    return None


def _map_frame_stems(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            out[p.stem] = p
    return out


def _sort_stems_numeric(stems: list[str]) -> list[str]:
    try:
        return sorted(stems, key=lambda s: int(s))
    except ValueError:
        return sorted(stems)


def discover_scene(scene_dir: Path) -> dict[str, Any] | None:
    color_dir = _find_first_existing(scene_dir, ["color", "colors"])
    instance_dir = _find_first_existing(scene_dir, ["instance", "instances"])
    valid_dir = _find_first_existing(scene_dir, ["valid_ids"])
    pose_dir = _find_first_existing(scene_dir, ["pose"])
    if color_dir is None or instance_dir is None or valid_dir is None:
        return None

    color_map = _map_frame_stems(color_dir, (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))
    instance_map = _map_frame_stems(instance_dir, (".png", ".PNG", ".tiff", ".tif"))
    valid_map = _map_frame_stems(valid_dir, (".json",))
    common_stems = sorted(set(color_map) & set(instance_map) & set(valid_map))
    if not common_stems:
        return None

    valid_ids_per_frame: dict[str, list[int]] = {}
    for stem in common_stems:
        with open(valid_map[stem], "r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = payload.get("ids", [])
        valid_ids_per_frame[stem] = [int(x) for x in ids]

    all_ids: set[int] = set()
    for ids in valid_ids_per_frame.values():
        all_ids.update(ids)

    return {
        "scene_name": scene_dir.name,
        "scene_dir": scene_dir,
        "color_map": color_map,
        "instance_map": instance_map,
        "valid_ids_per_frame": valid_ids_per_frame,
        "pose_dir": pose_dir,
        "frame_stems": _sort_stems_numeric(common_stems),
        "object_ids": sorted(all_ids),
    }


def _load_instance_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim > 2:
        mask = mask[..., 0]
    return mask.astype(np.int64)


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W], 0..255


def object_frames(scene: dict[str, Any], object_id: int) -> list[str]:
    return [s for s in scene["frame_stems"] if object_id in scene["valid_ids_per_frame"][s]]


def _instance_ids_per_frame(scene: dict[str, Any]) -> dict[str, set[int]]:
    cached = scene.get("_instance_ids_per_frame")
    if cached is not None:
        return cached

    out: dict[str, set[int]] = {}
    for stem in scene["frame_stems"]:
        try:
            mask = _load_instance_mask(scene["instance_map"][stem])
        except Exception:
            out[stem] = set()
            continue
        out[stem] = set(int(x) for x in np.unique(mask))
    scene["_instance_ids_per_frame"] = out
    return out


def object_frame_pools(scene: dict[str, Any], object_id: int) -> dict[str, list[str]]:
    positives: list[str] = []
    negatives: list[str] = []
    ignored_small: list[str] = []
    instance_ids = _instance_ids_per_frame(scene)
    for stem in _sort_stems_numeric(scene["frame_stems"]):
        in_valid = int(object_id) in scene["valid_ids_per_frame"][stem]
        has_object = int(object_id) in instance_ids.get(stem, set())
        if in_valid:
            positives.append(stem)
        elif has_object:
            ignored_small.append(stem)
        else:
            negatives.append(stem)
    return {"positive": positives, "negative": negatives, "ignored_small": ignored_small}


def _load_pose_position(pose_path: Path) -> np.ndarray:
    pose = np.loadtxt(pose_path).astype(np.float32)
    return pose[:3, 3]


def _collect_pose_candidates(scene: dict[str, Any], stems: list[str]) -> tuple[list[str], np.ndarray] | None:
    if scene["pose_dir"] is None:
        return None
    valid_stems: list[str] = []
    positions: list[np.ndarray] = []
    for s in _sort_stems_numeric(stems):
        pose_path = scene["pose_dir"] / f"{s}.txt"
        if not pose_path.is_file():
            continue
        try:
            positions.append(_load_pose_position(pose_path))
        except Exception:
            continue
        valid_stems.append(s)
    if not valid_stems:
        return None
    return valid_stems, np.stack(positions, axis=0)


def _sample_pose_frames(
    scene: dict[str, Any], stems: list[str], frame_count: int, strategy: str
) -> list[str] | None:
    pose_data = _collect_pose_candidates(scene, stems)
    if pose_data is None:
        return None
    valid_stems, positions = pose_data
    if len(valid_stems) < frame_count:
        return None
    if strategy == "near":
        return nearest_point_sampling_local(valid_stems, positions, frame_count)
    return farthest_point_sampling_local(valid_stems, positions, frame_count)


def _sample_balanced_pose_frames(
    scene: dict[str, Any], object_id: int, frame_count: int, strategy: str
) -> list[str] | None:
    half = frame_count // 2
    pools = object_frame_pools(scene, object_id)
    positive_stems = pools["positive"]
    negative_stems = pools["negative"]

    sampled_positive = _sample_pose_frames(scene, positive_stems, half, strategy)
    if sampled_positive is None:
        return None
    sampled_negative = _sample_pose_frames(scene, negative_stems, half, strategy)
    if sampled_negative is None:
        return None
    return _sort_stems_numeric(sampled_positive + sampled_negative)


def sample_pose_near_frames(scene: dict[str, Any], object_id: int, frame_count: int) -> list[str] | None:
    return _sample_balanced_pose_frames(scene, object_id, frame_count, strategy="near")


def sample_pose_diverse_frames(scene: dict[str, Any], object_id: int, frame_count: int) -> list[str] | None:
    return _sample_balanced_pose_frames(scene, object_id, frame_count, strategy="diverse")


def sample_continuous_frames(scene: dict[str, Any], object_id: int, frame_count: int) -> list[str] | None:
    # Backward-compatible alias. "continuous" was replaced by pose_near in this baseline.
    return sample_pose_near_frames(scene, object_id, frame_count)


def farthest_point_sampling_local(frame_ids: list[str], poses: np.ndarray, num_samples: int) -> list[str]:
    n = len(frame_ids)
    if num_samples >= n:
        return frame_ids.copy()
    selected_indices = [0]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[0] = True
    distances = np.linalg.norm(poses - poses[0], axis=1)
    for _ in range(num_samples - 1):
        distances[selected_mask] = -np.inf
        farthest_idx = int(np.argmax(distances))
        selected_indices.append(farthest_idx)
        selected_mask[farthest_idx] = True
        new_dists = np.linalg.norm(poses - poses[farthest_idx], axis=1)
        distances = np.minimum(distances, new_dists)
    return [frame_ids[i] for i in selected_indices]


def nearest_point_sampling_local(frame_ids: list[str], poses: np.ndarray, num_samples: int) -> list[str]:
    n = len(frame_ids)
    if num_samples >= n:
        return frame_ids.copy()
    selected_indices = [0]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[0] = True
    distances = np.linalg.norm(poses - poses[0], axis=1)
    for _ in range(num_samples - 1):
        distances[selected_mask] = np.inf
        nearest_idx = int(np.argmin(distances))
        selected_indices.append(nearest_idx)
        selected_mask[nearest_idx] = True
        new_dists = np.linalg.norm(poses - poses[nearest_idx], axis=1)
        distances = np.minimum(distances, new_dists)
    return [frame_ids[i] for i in selected_indices]


def build_sample(
    scene: dict[str, Any],
    frame_stems: list[str],
    object_id: int,
    num_points: int,
    prompt_seed: int,
    prompt_frame_index: int | None = None,
) -> dict[str, Any] | None:
    images: list[torch.Tensor] = []
    masks: list[np.ndarray] = []

    for stem in frame_stems:
        images.append(_load_rgb_tensor(scene["color_map"][stem]))
        masks.append(_load_instance_mask(scene["instance_map"][stem]))

    if not images:
        return None

    H, W = masks[0].shape
    for m in masks:
        if m.shape != (H, W):
            return None

    # Resize images and instance masks to 1024x1024 before prompt generation/inference.
    images_t = torch.stack(images, dim=0).float()  # [N,3,H,W]
    labels_2d = torch.from_numpy(np.stack(masks, axis=0)).long()  # [N,H,W]
    if images_t.shape[-2:] != TARGET_SIZE:
        images_t = F.interpolate(images_t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    if labels_2d.shape[-2:] != TARGET_SIZE:
        labels_2d = (
            F.interpolate(labels_2d.unsqueeze(1).float(), size=TARGET_SIZE, mode="nearest")
            .squeeze(1)
            .long()
        )

    N = labels_2d.shape[0]
    H, W = labels_2d.shape[-2:]
    labels_cat = torch.cat([labels_2d[i] for i in range(N)], dim=1).unsqueeze(0)  # [1,H,W*N]
    chosen_ids = torch.tensor([object_id], dtype=torch.int64)

    torch.manual_seed(prompt_seed)
    if prompt_frame_index is not None:
        if prompt_frame_index < 0 or prompt_frame_index >= N:
            raise ValueError(f"prompt_frame_index out of range: {prompt_frame_index} (N={N})")
        frame_mask = labels_2d[prompt_frame_index] == int(object_id)
        ys, xs = torch.where(frame_mask)
        if xs.numel() == 0:
            return None
        if xs.numel() < num_points:
            sel = torch.randint(0, xs.numel(), (num_points,))
        else:
            sel = torch.randperm(xs.numel())[:num_points]
        point_coords = torch.stack([xs[sel], ys[sel]], dim=1).float()
        lbl = torch.ones(num_points, dtype=torch.long)
        frame_idx = torch.full((num_points,), int(prompt_frame_index), dtype=torch.long)
    else:
        from utils.misc import sample_points_for_instances_single_frame

        sampled_points, point_labels = sample_points_for_instances_single_frame(
            labels_cat,
            chosen_ids,
            k=num_points,
            num_frames=N,
            frame_width=W,
        )
        pts = sampled_points[0]  # [K,2] in concatenated x
        lbl = point_labels[0].long()
        x_all, y_all = pts[:, 0], pts[:, 1]
        frame_idx = (x_all // W).long()
        x_local = x_all % W
        point_coords = torch.stack([x_local, y_all], dim=1).float()

        unique_frames = torch.unique(frame_idx)
        if unique_frames.numel() != 1:
            # SAM2 point API is per-frame click input. Keep one frame for strict parity.
            values, counts = torch.unique(frame_idx, return_counts=True)
            keep_frame = values[counts.argmax()]
            keep = frame_idx == keep_frame
            point_coords = point_coords[keep]
            lbl = lbl[keep]
            frame_idx = frame_idx[keep]

    if point_coords.numel() == 0:
        return None

    binary_gt = (labels_2d.numpy() == int(object_id))
    if not binary_gt.any():
        return None

    return {
        "images": images_t,  # [N,3,1024,1024]
        "gt_binary": binary_gt,  # [N,H,W], bool
        "frame_stems": frame_stems,
        "point_coords": point_coords,
        "point_labels": lbl,
        "point_frame_indices": frame_idx,
        "H": H,
        "W": W,
    }


def infer_samvggt(
    model: torch.nn.Module,
    sample: dict[str, Any],
    device: torch.device,
    amp_dtype: str,
) -> dict[str, np.ndarray]:
    images = sample["images"].unsqueeze(0).to(device)  # [1,N,3,H,W]
    point_coords_list = [sample["point_coords"].to(device)]
    point_labels_list = [sample["point_labels"].to(device)]
    point_frame_indices_list = [sample["point_frame_indices"].to(device)]

    use_amp = device.type == "cuda" and amp_dtype in {"fp16", "bf16"}
    # print(f"use_amp: {use_amp}")
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()

    with torch.inference_mode(), amp_ctx:
        outputs = model.forward(
            sam_pre=images,
            point_coords_list=point_coords_list,
            point_labels_list=point_labels_list,
            point_frame_indices_list=point_frame_indices_list,
            multimask_output=False,
            visualize=False,
        )
    logits = outputs["low_res_logits"][:, 0:1]  # [1,1,h,w*N]
    H, W, N = sample["H"], sample["W"], sample["images"].shape[0]
    lr_h, lr_w = logits.shape[2], logits.shape[3] // N
    lr_batch = (logits[0, 0]                              # [lr_h, lr_w*N]
                .reshape(lr_h, N, lr_w)
                .permute(1, 0, 2)                         # [N, lr_h, lr_w]
                .unsqueeze(1))                             # [N, 1, lr_h, lr_w]
    hr_batch = F.interpolate(lr_batch, size=(H, W), mode="bilinear", align_corners=False)  # [N, 1, H, W]
    up = (hr_batch[:, 0]                                  # [N, H, W]
          .permute(1, 0, 2)                               # [H, N, W]
          .reshape(1, 1, H, N * W))                       # [1, 1, H, W*N]
    pano_logits = up[0, 0].detach().cpu().float().numpy()
    scores = np.stack(
        [
            1.0 / (1.0 + np.exp(-np.clip(pano_logits[:, i * W : (i + 1) * W], -20.0, 20.0)))
            for i in range(N)
        ],
        axis=0,
    )
    pred = scores >= 0.5
    return {"pred_binary": pred, "pred_scores": scores, "logits": pano_logits}


def _save_sampled_frames_to_jpg_dir(sample: dict[str, Any], temp_dir: Path) -> None:
    images = sample["images"]  # [N,3,H,W], float
    for idx in range(images.shape[0]):
        arr = images[idx].permute(1, 2, 0).cpu().numpy()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(temp_dir / f"{idx:05d}.jpg", quality=95)


def _extract_obj_mask(video_res_masks: torch.Tensor, obj_ids: list[int], target_obj_id: int = 1) -> np.ndarray:
    if target_obj_id not in obj_ids:
        raise RuntimeError(f"Object id {target_obj_id} not found in SAM2 outputs: {obj_ids}")
    obj_idx = obj_ids.index(target_obj_id)
    m = video_res_masks[obj_idx]
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 3:
        m = m[0]
    return m.detach().cpu().numpy()


def infer_sam2(
    predictor: Any,
    sample: dict[str, Any],
    temp_root: Path,
) -> dict[str, np.ndarray]:
    N = sample["images"].shape[0]
    frame_indices = sample["point_frame_indices"].cpu().numpy()
    unique = np.unique(frame_indices)
    prompt_frame = int(unique[0])
    keep = frame_indices == prompt_frame

    points = sample["point_coords"].cpu().numpy()[keep].astype(np.float32)
    labels = sample["point_labels"].cpu().numpy()[keep].astype(np.int32)
    if points.shape[0] == 0:
        raise RuntimeError("No valid prompt points for SAM2")

    with tempfile.TemporaryDirectory(dir=temp_root) as td:
        td_path = Path(td)
        _save_sampled_frames_to_jpg_dir(sample, td_path)

        inference_state = predictor.init_state(video_path=str(td_path))
        pred_logits_by_frame: dict[int, np.ndarray] = {}

        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_frame,
            obj_id=1,
            points=points,
            labels=labels,
        )
        pred_logits_by_frame[int(frame_idx)] = _extract_obj_mask(out_mask_logits, out_obj_ids, target_obj_id=1)

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prompt_frame,
            reverse=False,
        ):
            pred_logits_by_frame[int(out_frame_idx)] = _extract_obj_mask(
                out_mask_logits, out_obj_ids, target_obj_id=1
            )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prompt_frame,
            reverse=True,
        ):
            pred_logits_by_frame[int(out_frame_idx)] = _extract_obj_mask(
                out_mask_logits, out_obj_ids, target_obj_id=1
            )

        predictor.reset_state(inference_state)

    H, W = sample["H"], sample["W"]
    ordered_scores = []
    for i in range(N):
        if i in pred_logits_by_frame:
            m = pred_logits_by_frame[i]
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m).resize((W, H), Image.BILINEAR))
            s = 1.0 / (1.0 + np.exp(-np.clip(m, -20.0, 20.0)))
            ordered_scores.append(s)
        else:
            ordered_scores.append(np.zeros((H, W), dtype=np.float32))
    scores = np.stack(ordered_scores, axis=0)
    pred = scores >= 0.5
    return {"pred_binary": pred, "pred_scores": scores}


def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-frame IoU: pred and gt are single-frame [H,W]."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def iou_binary_pooled(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pooled IoU: pred and gt are [N,H,W]; pool all frames into one mask."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def evaluate_prompt_points_on_pred_mask(
    sample: dict[str, Any],
    pred_binary: np.ndarray,
) -> list[dict[str, Any]]:
    """
    For each prompted point, check whether it lands on predicted foreground.
    Coordinates are rounded to nearest pixel, then clipped to image bounds.
    """
    H, W = int(sample["H"]), int(sample["W"])
    coords = sample["point_coords"].cpu().numpy()
    frame_idxs = sample["point_frame_indices"].cpu().numpy().astype(np.int64)

    checks: list[dict[str, Any]] = []
    for i, ((x_f, y_f), frame_idx) in enumerate(zip(coords, frame_idxs)):
        x = int(np.clip(np.rint(float(x_f)), 0, W - 1))
        y = int(np.clip(np.rint(float(y_f)), 0, H - 1))
        frame_idx_i = int(frame_idx)
        on_pred_mask = bool(pred_binary[frame_idx_i, y, x])
        checks.append(
            {
                "prompt_index": i,
                "frame_index_in_sample": frame_idx_i,
                "x": x,
                "y": y,
                "on_pred_mask": on_pred_mask,
            }
        )
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _update_frame_pr_counts(
    counts: dict[tuple[str, str, float], dict[str, int]],
    strategy: str,
    model_name: str,
    pred_binary: np.ndarray,
    gt: np.ndarray,
    iou_thresholds: list[float],
) -> None:
    """
    Per-frame detection-style accounting at each IoU threshold.

    For each frame:
    - TP: GT exists, prediction exists, IoU >= threshold
    - FN: GT exists and (no prediction OR IoU < threshold)
    - FP: prediction exists and (no GT OR IoU < threshold)
    """
    gt_b = gt.astype(bool)
    pred_b = pred_binary.astype(bool)
    num_frames = int(gt_b.shape[0])

    for thr in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0

        for i in range(num_frames):
            gt_exists = bool(gt_b[i].any())
            pred_exists = bool(pred_b[i].any())

            if gt_exists and pred_exists:
                frame_iou = iou_binary(pred_b[i], gt_b[i])
                if frame_iou >= thr:
                    tp += 1
                else:
                    # Unmatched prediction and missed GT for this frame.
                    fp += 1
                    fn += 1
            elif gt_exists and not pred_exists:
                fn += 1
            elif pred_exists and not gt_exists:
                fp += 1
            # else: TN (ignored for precision/recall)

        key_strategy = (strategy, model_name, float(thr))
        key_all = ("all", model_name, float(thr))
        for key in (key_strategy, key_all):
            if key not in counts:
                counts[key] = {"tp": 0, "fp": 0, "fn": 0}
            counts[key]["tp"] += tp
            counts[key]["fp"] += fp
            counts[key]["fn"] += fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAM2 vs SamVGGT on Hypersim.")
    parser.add_argument("--dataset_root", type=str,
                    default=os.environ.get("HYPERSIM_ROOT"), required=False,
                    help="Hypersim root; defaults to $HYPERSIM_ROOT.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sam_v_ckpt", type=str, required=True)
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="",
        help="Path to SAM2 checkpoint. Required unless --skip_sam2_inference is set.",
    )
    parser.add_argument(
        "--sam2_vos_optimized",
        action="store_true",
        help="Enable SAM2 VOS-optimized torch.compile path (requires full build toolchain).",
    )
    parser.add_argument(
        "--skip_sam2_inference",
        action="store_true",
        help="Skip SAM2 model init/inference and evaluate SamVGGT only.",
    )
    parser.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_encode_chunk", type=int, default=8)
    parser.add_argument("--frame_count", type=int, default=16)
    parser.add_argument("--strategies", type=str, default="pose_near,pose_diverse")
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sam_vggt_device",
        type=str,
        default=None,
        help="Device for SamVGGT (e.g., cuda:0). Defaults to --device.",
    )
    parser.add_argument(
        "--sam2_device",
        type=str,
        default=None,
        help="Device for SAM2 (e.g., cuda:1). Defaults to --device.",
    )
    parser.add_argument(
        "--sam_vggt_data_parallel_gpus",
        type=str,
        default="",
        help="Comma-separated CUDA GPU ids for torch.nn.DataParallel on SamVGGT, e.g. '0,1'.",
    )
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16", "none"])
    parser.add_argument("--output_dir", type=str, default="baseline_compare_output")
    parser.add_argument("--scene_limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--object_limit", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of independent scene shards (for multi-GPU multi-process runs).",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="This process shard id in [0, num_shards-1].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from model.sam_vggt_model import build_sam_vggt
    build_sam2_video_predictor = None
    if not args.skip_sam2_inference:
        from sam2.build_sam import build_sam2_video_predictor

    frame_count = int(max(16, min(32, args.frame_count)))
    if not args.skip_sam2_inference and not args.sam2_ckpt:
        raise ValueError("--sam2_ckpt is required unless --skip_sam2_inference is set")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    allowed = {"pose_near", "pose_diverse"}
    bad = [s for s in strategies if s not in allowed]
    if bad:
        raise ValueError(f"Invalid strategies: {bad}. Allowed: {sorted(allowed)}")
    if frame_count % 2 != 0:
        raise ValueError("--frame_count must be even to split half positive and half negative frames.")

    default_device = torch.device(args.device)
    sam_vggt_device = torch.device(args.sam_vggt_device) if args.sam_vggt_device else default_device
    sam2_device = torch.device(args.sam2_device) if args.sam2_device else default_device

    if (sam_vggt_device.type == "cuda" or sam2_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Before any work: a run that crashes should still say what it was running.
    write_run_provenance(output_dir, config=vars(args))
    temp_root = output_dir / ".tmp_frames"
    temp_root.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.dataset_root) / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dp_gpu_ids: list[int] = []
    if args.sam_vggt_data_parallel_gpus.strip():
        dp_gpu_ids = [int(x.strip()) for x in args.sam_vggt_data_parallel_gpus.split(",") if x.strip()]
        if len(dp_gpu_ids) < 2:
            raise ValueError("--sam_vggt_data_parallel_gpus needs at least 2 GPU ids (e.g., '0,1').")
        if sam_vggt_device.type != "cuda":
            raise ValueError("DataParallel for SamVGGT requires CUDA device.")
        num_cuda = torch.cuda.device_count()
        if any(g < 0 or g >= num_cuda for g in dp_gpu_ids):
            raise ValueError(f"Invalid GPU id in {dp_gpu_ids}; available ids are 0..{num_cuda-1}.")
        sam_vggt_device = torch.device(f"cuda:{dp_gpu_ids[0]}")

    sam2_label = f"SAM2={sam2_device}" if not args.skip_sam2_inference else "SAM2=SKIPPED"
    print(
        f"[Device] SamVGGT={sam_vggt_device} "
        f"{'(DataParallel gpus=' + str(dp_gpu_ids) + ')' if dp_gpu_ids else ''} | "
        f"{sam2_label}"
    )

    print("[Init] Building SamVGGT...")
    sam_vggt = build_sam_vggt(
        device=str(sam_vggt_device),
        sam_model_type=args.sam_model_type,
        sam_encode_chunk=args.sam_encode_chunk,
    )
    load_samvggt_checkpoint(sam_vggt, args.sam_v_ckpt, sam_vggt_device)
    if dp_gpu_ids:
        sam_vggt = torch.nn.DataParallel(sam_vggt, device_ids=dp_gpu_ids, output_device=dp_gpu_ids[0])
    sam_vggt.eval()

    sam2_predictor = None
    if not args.skip_sam2_inference:
        print("[Init] Building SAM2 video predictor...")
        sam2_predictor = build_sam2_video_predictor(
            args.sam2_cfg,
            args.sam2_ckpt,
            device=str(sam2_device),
            vos_optimized=args.sam2_vos_optimized,
        )

    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if args.num_shards > 1:
        scene_dirs = [p for i, p in enumerate(scene_dirs) if i % args.num_shards == args.shard_id]
        print(
            f"[Shard] shard_id={args.shard_id}/{args.num_shards} assigned_scenes={len(scene_dirs)}"
        )
    if args.scene_limit > 0:
        scene_dirs = scene_dirs[: args.scene_limit]

    frame_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    pr_counts: dict[tuple[str, str, float], dict[str, int]] = {}
    prompt_point_rows: list[dict[str, Any]] = []
    prompt_miss_rows: list[dict[str, Any]] = []
    prompt_miss_summary_rows: list[dict[str, Any]] = []

    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene = discover_scene(scene_dir)
        if scene is None:
            skip_rows.append(
                {"scene": scene_dir.name, "object_id": "", "strategy": "", "reason": "missing_required_modalities"}
            )
            continue

        object_ids = scene["object_ids"]
        if args.object_limit > 0:
            object_ids = object_ids[: args.object_limit]

        print(
            f"[Scene {scene_idx + 1}/{len(scene_dirs)}] {scene['scene_name']} "
            f"objects={len(object_ids)} frames={len(scene['frame_stems'])}"
        )

        for obj_id in object_ids:
            for strategy in strategies:
                if strategy == "pose_near":
                    sampled = sample_pose_near_frames(scene, obj_id, frame_count)
                else:
                    sampled = sample_pose_diverse_frames(scene, obj_id, frame_count)

                if sampled is None:
                    skip_rows.append(
                        {
                            "scene": scene["scene_name"],
                            "object_id": obj_id,
                            "strategy": strategy,
                            "reason": "insufficient_frames_for_strategy",
                        }
                    )
                    continue

                seed_key = f"{scene['scene_name']}|{obj_id}|{strategy}".encode("utf-8")
                prompt_seed = int(args.seed + (zlib.adler32(seed_key) % 1000000))
                positive_indices = [
                    i for i, stem in enumerate(sampled) if int(obj_id) in scene["valid_ids_per_frame"][stem]
                ]
                if not positive_indices:
                    skip_rows.append(
                        {
                            "scene": scene["scene_name"],
                            "object_id": obj_id,
                            "strategy": strategy,
                            "reason": "no_positive_frames_in_sampled_clip",
                        }
                    )
                    continue
                rng = random.Random(prompt_seed)
                prompt_frame_index = int(rng.choice(positive_indices))
                sample = build_sample(
                    scene=scene,
                    frame_stems=sampled,
                    object_id=obj_id,
                    num_points=args.num_points,
                    prompt_seed=prompt_seed,
                    prompt_frame_index=prompt_frame_index,
                )
                if sample is None:
                    skip_rows.append(
                        {
                            "scene": scene["scene_name"],
                            "object_id": obj_id,
                            "strategy": strategy,
                            "reason": "failed_to_build_prompt_or_gt",
                        }
                    )
                    continue

                try:
                    out_vggt = infer_samvggt(sam_vggt, sample, device=sam_vggt_device, amp_dtype=args.amp_dtype)
                    out_sam2 = None
                    if not args.skip_sam2_inference:
                        out_sam2 = infer_sam2(sam2_predictor, sample, temp_root=temp_root)
                except Exception as e:
                    skip_rows.append(
                        {
                            "scene": scene["scene_name"],
                            "object_id": obj_id,
                            "strategy": strategy,
                            "reason": f"inference_error:{type(e).__name__}",
                        }
                    )
                    continue

                gt = sample["gt_binary"]
                pred_vggt = out_vggt["pred_binary"]
                prompt_checks = evaluate_prompt_points_on_pred_mask(sample, pred_vggt)
                num_prompt_points = len(prompt_checks)
                num_missed_prompt_points = 0
                for check in prompt_checks:
                    frame_stem = sample["frame_stems"][check["frame_index_in_sample"]]
                    row_common = {
                        "scene": scene["scene_name"],
                        "object_id": obj_id,
                        "strategy": strategy,
                        "num_frames": len(sample["frame_stems"]),
                        "prompt_index": check["prompt_index"],
                        "frame_index_in_sample": check["frame_index_in_sample"],
                        "frame_stem": frame_stem,
                        "x": check["x"],
                        "y": check["y"],
                        "on_pred_mask": int(check["on_pred_mask"]),
                    }
                    prompt_point_rows.append(row_common)
                    if not check["on_pred_mask"]:
                        num_missed_prompt_points += 1
                        prompt_miss_rows.append(row_common)

                prompt_miss_summary_rows.append(
                    {
                        "scene": scene["scene_name"],
                        "object_id": obj_id,
                        "strategy": strategy,
                        "num_frames": len(sample["frame_stems"]),
                        "num_prompt_points": num_prompt_points,
                        "num_missed_prompt_points": num_missed_prompt_points,
                        "has_exactly_half_prompt_points_missed": int(
                            num_prompt_points > 0 and (num_missed_prompt_points * 2 == num_prompt_points)
                        ),
                    }
                )

                ious_vggt = [iou_binary(pred_vggt[i], gt[i]) for i in range(gt.shape[0])]
                pooled_iou_vggt = iou_binary_pooled(pred_vggt, gt)
                _update_frame_pr_counts(
                    pr_counts,
                    strategy=strategy,
                    model_name="sam_vggt",
                    pred_binary=pred_vggt,
                    gt=gt,
                    iou_thresholds=IOU_THRESHOLDS,
                )
                if out_sam2 is not None:
                    pred_sam2 = out_sam2["pred_binary"]
                    ious_sam2 = [iou_binary(pred_sam2[i], gt[i]) for i in range(gt.shape[0])]
                    pooled_iou_sam2 = iou_binary_pooled(pred_sam2, gt)
                    _update_frame_pr_counts(
                        pr_counts,
                        strategy=strategy,
                        model_name="sam2",
                        pred_binary=pred_sam2,
                        gt=gt,
                        iou_thresholds=IOU_THRESHOLDS,
                    )

                for i, stem in enumerate(sample["frame_stems"]):
                    frame_rows.append(
                        {
                            "scene": scene["scene_name"],
                            "object_id": obj_id,
                            "strategy": strategy,
                            "model": "sam_vggt",
                            "frame_index_in_sample": i,
                            "frame_stem": stem,
                            "iou": ious_vggt[i],
                        }
                    )
                    if out_sam2 is not None:
                        frame_rows.append(
                            {
                                "scene": scene["scene_name"],
                                "object_id": obj_id,
                                "strategy": strategy,
                                "model": "sam2",
                                "frame_index_in_sample": i,
                                "frame_stem": stem,
                                "iou": ious_sam2[i],
                            }
                        )

                object_rows.append(
                    {
                        "scene": scene["scene_name"],
                        "object_id": obj_id,
                        "strategy": strategy,
                        "model": "sam_vggt",
                        "num_frames": len(sample["frame_stems"]),
                        "pooled_iou": pooled_iou_vggt,
                        "mean_per_frame_iou": float(np.mean(ious_vggt)),
                    }
                )
                if out_sam2 is not None:
                    object_rows.append(
                        {
                            "scene": scene["scene_name"],
                            "object_id": obj_id,
                            "strategy": strategy,
                            "model": "sam2",
                            "num_frames": len(sample["frame_stems"]),
                            "pooled_iou": pooled_iou_sam2,
                            "mean_per_frame_iou": float(np.mean(ious_sam2)),
                        }
                    )

    write_csv(
        output_dir / "frame_iou.csv",
        frame_rows,
        ["scene", "object_id", "strategy", "model", "frame_index_in_sample", "frame_stem", "iou"],
    )
    write_csv(
        output_dir / "object_summary.csv",
        object_rows,
        ["scene", "object_id", "strategy", "model", "num_frames", "pooled_iou", "mean_per_frame_iou"],
    )
    write_csv(output_dir / "skipped_items.csv", skip_rows, ["scene", "object_id", "strategy", "reason"])
    write_csv(
        output_dir / "sam_vggt_prompt_point_checks.csv",
        prompt_point_rows,
        [
            "scene",
            "object_id",
            "strategy",
            "num_frames",
            "prompt_index",
            "frame_index_in_sample",
            "frame_stem",
            "x",
            "y",
            "on_pred_mask",
        ],
    )
    write_csv(
        output_dir / "sam_vggt_prompt_point_misses.csv",
        prompt_miss_rows,
        [
            "scene",
            "object_id",
            "strategy",
            "num_frames",
            "prompt_index",
            "frame_index_in_sample",
            "frame_stem",
            "x",
            "y",
            "on_pred_mask",
        ],
    )
    write_csv(
        output_dir / "sam_vggt_prompt_point_miss_summary.csv",
        prompt_miss_summary_rows,
        [
            "scene",
            "object_id",
            "strategy",
            "num_frames",
            "num_prompt_points",
            "num_missed_prompt_points",
            "has_exactly_half_prompt_points_missed",
        ],
    )

    scene_buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    global_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in object_rows:
        key_scene = (r["scene"], r["strategy"], r["model"])
        key_global = (r["strategy"], r["model"])
        scene_buckets[key_scene].append(float(r["pooled_iou"]))
        global_buckets[key_global].append(float(r["pooled_iou"]))

    scene_rows: list[dict[str, Any]] = []
    for (scene_name, strategy, model_name), vals in sorted(scene_buckets.items()):
        scene_rows.append(
            {
                "scene": scene_name,
                "strategy": strategy,
                "model": model_name,
                "num_objects": len(vals),
                "avg_pooled_iou_over_objects": float(np.mean(vals)),
            }
        )
    write_csv(
        output_dir / "scene_summary.csv",
        scene_rows,
        ["scene", "strategy", "model", "num_objects", "avg_pooled_iou_over_objects"],
    )

    global_summary: dict[str, Any] = {"by_strategy": {}}
    for (strategy, model_name), vals in sorted(global_buckets.items()):
        by_strategy = global_summary["by_strategy"].setdefault(strategy, {})
        by_strategy[model_name] = {
            "num_objects": len(vals),
            "avg_pooled_iou_over_all_objects": float(np.mean(vals)),
        }

    pr_rows: list[dict[str, Any]] = []
    pr_summary: dict[str, Any] = {}
    for (strategy, model_name, thr), c in sorted(pr_counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        pr_rows.append(
            {
                "strategy": strategy,
                "model": model_name,
                "iou_threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            }
        )
        pr_summary.setdefault(strategy, {}).setdefault(model_name, {})[f"{thr:.1f}"] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    write_csv(
        output_dir / "frame_detection_precision_recall_iou_thresholds.csv",
        pr_rows,
        ["strategy", "model", "iou_threshold", "tp", "fp", "fn", "precision", "recall"],
    )

    global_summary["frame_detection_precision_recall_by_iou_threshold"] = pr_summary
    total_prompt_points = sum(int(r["num_prompt_points"]) for r in prompt_miss_summary_rows)
    total_prompt_misses = sum(int(r["num_missed_prompt_points"]) for r in prompt_miss_summary_rows)
    samples_with_exactly_half_missed = sum(
        int(r["has_exactly_half_prompt_points_missed"]) for r in prompt_miss_summary_rows
    )
    prompt_miss_by_strategy: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "num_prompt_points": 0,
            "num_missed_prompt_points": 0,
            "samples_with_exactly_half_prompt_points_missed": 0,
        }
    )
    for r in prompt_miss_summary_rows:
        s = str(r["strategy"])
        prompt_miss_by_strategy[s]["num_prompt_points"] += int(r["num_prompt_points"])
        prompt_miss_by_strategy[s]["num_missed_prompt_points"] += int(r["num_missed_prompt_points"])
        prompt_miss_by_strategy[s]["samples_with_exactly_half_prompt_points_missed"] += int(
            r["has_exactly_half_prompt_points_missed"]
        )

    global_summary["sam_vggt_prompt_point_coverage"] = {
        "total_samples_evaluated": len(prompt_miss_summary_rows),
        "total_prompt_points": total_prompt_points,
        "total_prompt_points_not_on_pred_mask": total_prompt_misses,
        "samples_with_exactly_half_prompt_points_not_on_pred_mask": samples_with_exactly_half_missed,
        "by_strategy": dict(prompt_miss_by_strategy),
    }

    with open(output_dir / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    print("\n[Done] Outputs:")
    print(f"- {output_dir / 'frame_iou.csv'}")
    print(f"- {output_dir / 'object_summary.csv'}")
    print(f"- {output_dir / 'scene_summary.csv'}")
    print(f"- {output_dir / 'global_summary.json'}")
    print(f"- {output_dir / 'skipped_items.csv'}")
    print(f"- {output_dir / 'frame_detection_precision_recall_iou_thresholds.csv'}")
    print(f"- {output_dir / 'sam_vggt_prompt_point_checks.csv'}")
    print(f"- {output_dir / 'sam_vggt_prompt_point_misses.csv'}")
    print(f"- {output_dir / 'sam_vggt_prompt_point_miss_summary.csv'}")

    for strategy in sorted(global_summary["by_strategy"].keys()):
        row = global_summary["by_strategy"][strategy]
        if "sam_vggt" in row and "sam2" in row:
            print(
                f"[Global][{strategy}] "
                f"sam_vggt={row['sam_vggt']['avg_pooled_iou_over_all_objects']:.4f} "
                f"sam2={row['sam2']['avg_pooled_iou_over_all_objects']:.4f}"
            )
        elif "sam_vggt" in row:
            print(
                f"[Global][{strategy}] "
                f"sam_vggt={row['sam_vggt']['avg_pooled_iou_over_all_objects']:.4f}"
            )
    if "sam_vggt_prompt_point_coverage" in global_summary:
        cov = global_summary["sam_vggt_prompt_point_coverage"]
        print(
            "[Global][sam_vggt_prompt_point_coverage] "
            f"samples={cov['total_samples_evaluated']} "
            f"prompt_points={cov['total_prompt_points']} "
            f"missed_prompt_points={cov['total_prompt_points_not_on_pred_mask']} "
            f"samples_with_exactly_half_miss={cov['samples_with_exactly_half_prompt_points_not_on_pred_mask']}"
        )

    shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
