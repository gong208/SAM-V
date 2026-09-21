#!/usr/bin/env python3
"""
Combined SamVGGT 3D Tracking Benchmark.

Evaluates SamVGGT everything mode on IGGT 3D tracking scenes with all metrics:
  - T-mIoU and T-SR (temporal tracking quality)
  - Pooled IoU (multi-frame segmentation quality)
  - Frame-level precision/recall at multiple IoU thresholds

Usage:
    python sam_vggt_3dtracking_benchmark.py \
        --sam_v_ckpt /path/to/checkpoint.pth \
        --benchmark_root $BENCHMARK_ROOT/scannetpp \
        --prompt_source grid \
        --output_dir ./3dtracking_output
"""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
SAM_HQ_ROOT = REPO_ROOT / "submodules" / "sam-hq"
VGGT_ROOT = REPO_ROOT / "submodules" / "vggt"
# The LangSplat-modified SAM fork backs the optional `sam_dense_masks_ls` prompt
# source only. It is imported lazily (see _load_langsplat_sam) so the default
# pipeline needs just the vggt and sam-hq submodules.
LS_ROOT = REPO_ROOT / "submodules" / "segment-anything-langsplat-modified"
for p in [str(SAM_HQ_ROOT), str(VGGT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from masks.automatic_mask_generator import SamVGGTAutomaticMaskGenerator
from masks.everything_mode_demo import load_checkpoint, visualize_everything
from utils.provenance import write_run_provenance
from masks.prompt_sampling import sample_prompt_points_from_mask
from model.sam_vggt_model import build_sam_vggt
from segment_anything import (
    SamAutomaticMaskGenerator as SamDenseAutomaticMaskGenerator,
    sam_model_registry,
)


def _load_langsplat_sam():
    """Import the optional LangSplat SAM fork on demand.

    Only the `sam_dense_masks_ls` prompt source needs it; the paper pipeline
    uses `sam_dense_masks` and never reaches this.
    """
    if str(LS_ROOT) not in sys.path:
        sys.path.insert(0, str(LS_ROOT))
    try:
        from segment_anything_ls import (
            SamAutomaticMaskGenerator as SamLSAutomaticMaskGenerator,
            sam_model_registry as ls_sam_model_registry,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional submodule
        raise SystemExit(
            "prompt_source 'sam_dense_masks_ls' requires the optional submodule "
            f"'segment-anything-langsplat-modified' at {LS_ROOT}. Either clone it "
            "there or use --prompt_source sam_dense_masks (the paper default)."
        ) from exc
    return SamLSAutomaticMaskGenerator, ls_sam_model_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE = (1024, 1024)
IOU_THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]

PROPOSAL_SAM_MODEL_TYPE = "vit_h"
PROPOSAL_POINTS_PER_SIDE = 32
PROPOSAL_POINTS_PER_BATCH = 64
PROPOSAL_PRED_IOU_THRESH = 0.88
PROPOSAL_STABILITY_SCORE_THRESH = 0.95
PROPOSAL_BOX_NMS_THRESH = 0.7
PROPOSAL_MIN_MASK_REGION_AREA = 0
PROPOSAL_MAX_MASKS_PER_FRAME = 64

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameRecord:
    image_path: Path
    gt_path: Path


@dataclass
class SceneInputs:
    scene_id: str
    image_tensor: torch.Tensor        # [N, 3, H, W]
    gt_instance_maps: np.ndarray       # [N, H, W] int64
    frame_names: List[str]
    image_paths: List[str]
    gt_paths: List[str]


RGB_FRAME_STEM_RE = re.compile(r"^frame_\d+$")
RGB_NUMERIC_STEM_RE = re.compile(r"^\d+$")

# ---------------------------------------------------------------------------
# Scene / frame discovery
# ---------------------------------------------------------------------------

def is_rgb_frame_path(image_path: Path) -> bool:
    if image_path.suffix.lower() != ".jpg":
        return False
    stem = image_path.stem
    return (
        RGB_FRAME_STEM_RE.fullmatch(stem) is not None
        or RGB_NUMERIC_STEM_RE.fullmatch(stem) is not None
    )


def list_scene_image_paths(image_dir: Path) -> List[Path]:
    return [p for p in sorted(image_dir.glob("*.jpg")) if is_rgb_frame_path(p)]


def gt_path_for_image(image_path: Path, gt_parent_dir: Path) -> Path:
    return gt_parent_dir / f"{image_path.stem}_label.npy"


def _is_valid_scene_dir(d: Path) -> bool:
    """
    A valid scene has <scene>/images/ containing RGB frames (frame_XXXXXX.jpg
    or XXXXX.jpg) alongside matching GT instance masks (<stem>_label.npy).
    """
    image_dir = d / "images"
    if not image_dir.is_dir():
        return False
    image_paths = list_scene_image_paths(image_dir)
    if not image_paths:
        return False
    return any(gt_path_for_image(p, image_dir).is_file() for p in image_paths)


def discover_scene_dirs(benchmark_root: Path, scene_ids: Optional[Sequence[str]]) -> List[Path]:
    if scene_ids:
        scene_dirs = [benchmark_root / sid for sid in scene_ids]
    else:
        scene_dirs = sorted(
            p for p in benchmark_root.iterdir()
            if p.is_dir() and _is_valid_scene_dir(p)
        )

    valid, missing = [], []
    for sd in scene_dirs:
        if _is_valid_scene_dir(sd):
            valid.append(sd)
        else:
            missing.append(str(sd))
    if missing:
        raise FileNotFoundError("Missing required scene directories:\n" + "\n".join(missing))
    if not valid:
        raise FileNotFoundError(f"No valid scenes found under {benchmark_root}.")
    return valid


def build_scene_frame_records(scene_dir: Path) -> List[FrameRecord]:
    """
    Build frame records for a scene.  Layout:
      <scene>/images/<stem>.jpg          -- RGB frame
      <scene>/images/<stem>_label.npy    -- GT instance mask

    where <stem> is frame_XXXXXX (scannetpp) or XXXXX (scannet).
    """
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No 'images' directory in {scene_dir}")
    image_paths = list_scene_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No RGB frames found in {image_dir}")

    records: List[FrameRecord] = []
    missing_gt: List[str] = []
    for ip in image_paths:
        gp = gt_path_for_image(ip, image_dir)
        if not gp.is_file():
            missing_gt.append(str(gp))
            continue
        records.append(FrameRecord(image_path=ip, gt_path=gp))
    if missing_gt:
        preview = "\n".join(missing_gt[:10])
        suffix = "" if len(missing_gt) <= 10 else f"\n... and {len(missing_gt) - 10} more"
        raise FileNotFoundError(f"Missing GT files:\n{preview}{suffix}")
    if not records:
        raise FileNotFoundError(f"No aligned RGB/GT frame pairs found in {scene_dir}")
    return records


# ---------------------------------------------------------------------------
# Image / GT loading
# ---------------------------------------------------------------------------

def load_rgb_tensor(image_paths: Sequence[Path], target_size: Tuple[int, int]) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    images = torch.stack(tensors, dim=0)
    if tuple(images.shape[-2:]) != target_size:
        images = F.interpolate(images, size=target_size, mode="bilinear", align_corners=False)
    return images


def resize_instance_map(instance_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    if instance_map.shape == (target_h, target_w):
        return instance_map.astype(np.int64, copy=False)
    tensor = torch.from_numpy(instance_map.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_size, mode="nearest")
    return resized[0, 0].cpu().numpy().astype(np.int64, copy=False)


def load_gt_instance_maps(gt_paths: Sequence[Path], target_size: Tuple[int, int]) -> np.ndarray:
    gt_maps: List[np.ndarray] = []
    for gt_path in gt_paths:
        gt_map = np.load(gt_path)
        if gt_map.ndim != 2:
            raise ValueError(f"Expected 2D instance map in {gt_path}, got shape {gt_map.shape}")
        gt_maps.append(resize_instance_map(gt_map, target_size))
    return np.stack(gt_maps, axis=0)


def validate_frame_geometry(image_path: Path, gt_shape: Tuple[int, int], ratio_tolerance: float = 1e-2) -> None:
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    gt_h, gt_w = gt_shape
    img_ratio = img_w / img_h
    gt_ratio = gt_w / gt_h
    if not math.isclose(img_ratio, gt_ratio, rel_tol=ratio_tolerance, abs_tol=ratio_tolerance):
        raise ValueError(f"Aspect-ratio mismatch for {image_path}: RGB {img_w}x{img_h} vs GT {gt_w}x{gt_h}")


def load_scene_inputs(scene_dir: Path, target_size: Tuple[int, int]) -> SceneInputs:
    frame_records = build_scene_frame_records(scene_dir)
    image_paths = [r.image_path for r in frame_records]
    gt_paths = [r.gt_path for r in frame_records]
    for ip, gp in zip(image_paths, gt_paths):
        gt_map = np.load(gp, mmap_mode="r")
        if gt_map.ndim != 2:
            raise ValueError(f"Expected 2D instance map in {gp}, got shape {gt_map.shape}")
        validate_frame_geometry(ip, tuple(gt_map.shape))
    image_tensor = load_rgb_tensor(image_paths, target_size=target_size)
    gt_instance_maps = load_gt_instance_maps(gt_paths, target_size=target_size)
    return SceneInputs(
        scene_id=scene_dir.name,
        image_tensor=image_tensor,
        gt_instance_maps=gt_instance_maps,
        frame_names=[p.name for p in image_paths],
        image_paths=[str(p) for p in image_paths],
        gt_paths=[str(p) for p in gt_paths],
    )


# ---------------------------------------------------------------------------
# Mask encode / decode helpers
# ---------------------------------------------------------------------------

def decode_mask(mask_data: Any) -> np.ndarray:
    if isinstance(mask_data, np.ndarray):
        return mask_data.astype(bool, copy=False)
    if torch.is_tensor(mask_data):
        return mask_data.detach().cpu().numpy().astype(bool, copy=False)
    if isinstance(mask_data, dict):
        counts = mask_data.get("counts")
        size = mask_data.get("size")
        if counts is None or size is None or len(size) != 2:
            raise ValueError(f"Unsupported RLE mask format: {mask_data}")
        height, width = int(size[0]), int(size[1])
        flat = np.zeros(height * width, dtype=np.uint8)
        idx, value = 0, 0
        for count in counts:
            next_idx = idx + int(count)
            flat[idx:next_idx] = value
            idx = next_idx
            value = 1 - value
        if idx != flat.size:
            raise ValueError(f"Malformed RLE: decoded {idx}, expected {flat.size}")
        return flat.reshape((width, height)).T.astype(bool, copy=False)
    raise TypeError(f"Unsupported mask type: {type(mask_data)!r}")


def encode_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    mask = mask.astype(bool, copy=False)
    height, width = mask.shape
    flat = mask.T.flatten()
    diff = np.diff(flat.astype(np.int8))
    change = np.concatenate([[0], np.where(diff != 0)[0] + 1, [height * width]])
    counts = np.diff(change).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": [height, width], "counts": counts}


def image_tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    return image_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)


def adapt_mask_decoder_signature_for_sam_hq(sam_model: torch.nn.Module) -> torch.nn.Module:
    """
    Make predictor calls robust when mask_decoder.forward does not accept
    HQ-specific kwargs (hq_token_only, interm_embeddings).
    """
    if getattr(sam_model.mask_decoder, "_hq_kwargs_compat_wrapped", False):
        return sam_model

    forward_sig = inspect.signature(sam_model.mask_decoder.forward)
    accepts_hq = "hq_token_only" in forward_sig.parameters
    accepts_interm = "interm_embeddings" in forward_sig.parameters
    if accepts_hq and accepts_interm:
        sam_model.mask_decoder._hq_kwargs_compat_wrapped = True
        return sam_model

    original_forward = sam_model.mask_decoder.forward

    def forward_compat(*args, **kwargs):
        kwargs.pop("hq_token_only", None)
        kwargs.pop("interm_embeddings", None)
        return original_forward(*args, **kwargs)

    sam_model.mask_decoder.forward = forward_compat  # type: ignore[assignment]
    sam_model.mask_decoder._hq_kwargs_compat_wrapped = True
    return sam_model


# ---------------------------------------------------------------------------
# Prompt group serialization
# ---------------------------------------------------------------------------

def serialize_prompt_groups(prompt_groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []
    for group in prompt_groups:
        serialized.append({
            **{k: v for k, v in group.items() if k not in ("points", "labels")},
            "points": np.asarray(group["points"], dtype=np.float32).tolist(),
            "labels": np.asarray(group["labels"], dtype=np.int64).tolist(),
        })
    return serialized


def deserialize_prompt_groups(payload: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {**g, "points": np.asarray(g["points"], dtype=np.float32),
              "labels": np.asarray(g["labels"], dtype=np.int64)}
        for g in payload
    ]


# ---------------------------------------------------------------------------
# SAM-HQ compatibility shim
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt cache + dense prompt generation
# ---------------------------------------------------------------------------

def prompt_cache_path(
    scene_output_dir: Path,
    prompt_source: str,
    prompt_sampling_method: str,
    prompt_points_per_mask: int,
) -> Path:
    cache_name = (
        f"prompt_groups_{prompt_source}_"
        f"{prompt_sampling_method}_"
        f"k{prompt_points_per_mask}_"
        f"pps{PROPOSAL_POINTS_PER_SIDE}_"
        f"piou{PROPOSAL_PRED_IOU_THRESH:.2f}_"
        f"stab{PROPOSAL_STABILITY_SCORE_THRESH:.2f}_"
        f"nms{PROPOSAL_BOX_NMS_THRESH:.2f}_"
        f"max{PROPOSAL_MAX_MASKS_PER_FRAME}.json"
    ).replace(".", "p")
    return scene_output_dir / cache_name


def build_dense_prompt_groups(
    scene_inputs: SceneInputs,
    proposal_sam_model: torch.nn.Module,
    scene_output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    if args.prompt_source == "grid":
        return None, {"prompt_source": "grid", "cached": False}

    # Key the cache on the LangSplat level too so medium/large/both don't collide.
    source_tag = args.prompt_source
    if args.prompt_source == "sam_dense_masks_ls":
        source_tag = f"{args.prompt_source}_{args.ls_level}"
    cache_file = prompt_cache_path(
        scene_output_dir, source_tag,
        args.prompt_sampling_method, args.prompt_points_per_mask,
    )
    if cache_file.is_file():
        with cache_file.open("r") as f:
            payload = json.load(f)
        return deserialize_prompt_groups(payload["prompt_groups"]), {
            **payload["summary"], "cache_path": str(cache_file), "cached": True,
        }

    prompt_views = list(range(scene_inputs.image_tensor.shape[0]))
    use_ls = args.prompt_source == "sam_dense_masks_ls"
    generator_cls = (
        _load_langsplat_sam()[0] if use_ls else SamDenseAutomaticMaskGenerator
    )
    dense_generator = generator_cls(
        model=proposal_sam_model,
        points_per_side=PROPOSAL_POINTS_PER_SIDE,
        points_per_batch=PROPOSAL_POINTS_PER_BATCH,
        pred_iou_thresh=PROPOSAL_PRED_IOU_THRESH,
        stability_score_thresh=PROPOSAL_STABILITY_SCORE_THRESH,
        box_nms_thresh=PROPOSAL_BOX_NMS_THRESH,
        min_mask_region_area=PROPOSAL_MIN_MASK_REGION_AREA,
        output_mode="binary_mask",
    )

    rng = np.random.default_rng(0)
    prompt_groups: List[Dict[str, Any]] = []
    frames_summary: List[Dict[str, Any]] = []
    next_prompt_id = 0

    for frame_idx in prompt_views:
        image_np = image_tensor_to_uint8(scene_inputs.image_tensor[frame_idx])
        if use_ls:
            # LangSplat AMG returns (anns_medium, anns_large) hierarchy levels;
            # keep the level(s) selected by --ls_level.
            anns_medium, anns_large = dense_generator.generate(image_np)
            if args.ls_level == "medium":
                dense_masks = anns_medium
            elif args.ls_level == "both":
                dense_masks = anns_medium + anns_large
            else:  # "large"
                dense_masks = anns_large
        else:
            dense_masks = dense_generator.generate(image_np, multimask_output=True)
        dense_masks.sort(
            key=lambda ann: (
                float(ann.get("predicted_iou", 0.0)),
                float(ann.get("stability_score", 0.0)),
                float(ann.get("area", 0.0)),
            ),
            reverse=True,
        )
        dense_masks = dense_masks[:PROPOSAL_MAX_MASKS_PER_FRAME]

        kept = 0
        for proposal_index, ann in enumerate(dense_masks):
            proposal_mask = decode_mask(ann["segmentation"])
            if not np.any(proposal_mask):
                continue
            points = sample_prompt_points_from_mask(
                proposal_mask,
                method=args.prompt_sampling_method,
                num_points=args.prompt_points_per_mask,
                rng=rng,
            )
            prompt_groups.append({
                "prompt_id": int(next_prompt_id),
                "frame_index": int(frame_idx),
                "points": points.astype(np.float32, copy=False),
                "labels": np.ones((len(points),), dtype=np.int64),
                "prompt_kind": args.prompt_sampling_method,
                "proposal_index": int(proposal_index),
                "proposal_area": int(ann["area"]),
                "proposal_predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "proposal_stability_score": float(ann.get("stability_score", 0.0)),
                "proposal_bbox": ann.get("bbox"),
                "proposal_rle": encode_mask_to_rle(proposal_mask),
            })
            next_prompt_id += 1
            kept += 1

        frames_summary.append({
            "frame_index": int(frame_idx),
            "frame_name": scene_inputs.frame_names[frame_idx],
            "num_dense_proposals": int(len(dense_masks)),
            "num_prompt_groups": int(kept),
        })

    summary = {
        "prompt_source": args.prompt_source,
        "prompt_sampling_method": args.prompt_sampling_method,
        "prompt_points_per_mask": int(args.prompt_points_per_mask),
        "proposal_points_per_side": PROPOSAL_POINTS_PER_SIDE,
        "proposal_points_per_batch": PROPOSAL_POINTS_PER_BATCH,
        "proposal_pred_iou_thresh": PROPOSAL_PRED_IOU_THRESH,
        "proposal_stability_score_thresh": PROPOSAL_STABILITY_SCORE_THRESH,
        "proposal_box_nms_thresh": PROPOSAL_BOX_NMS_THRESH,
        "proposal_min_mask_region_area": PROPOSAL_MIN_MASK_REGION_AREA,
        "proposal_max_masks_per_frame": PROPOSAL_MAX_MASKS_PER_FRAME,
        "num_prompt_groups": int(len(prompt_groups)),
        "frames": frames_summary,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump({"summary": summary, "prompt_groups": serialize_prompt_groups(prompt_groups)}, f, indent=2)
    return prompt_groups, {**summary, "cache_path": str(cache_file), "cached": False}


# ---------------------------------------------------------------------------
# IoU functions
# ---------------------------------------------------------------------------

def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-frame IoU on single-frame [H,W] boolean arrays."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def iou_binary_pooled(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pooled IoU on [N,H,W] arrays -- pool all frames into one."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


# ---------------------------------------------------------------------------
# GT-to-prediction mask building
# ---------------------------------------------------------------------------

def _build_pred_masks(
    annotations: Sequence[Dict[str, Any]],
    num_frames: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Return [num_pred, num_frames, H, W] boolean array from annotations."""
    num_pred = len(annotations)
    masks = np.zeros((num_pred, num_frames, H, W), dtype=bool)
    for p_idx, ann in enumerate(annotations):
        per_view = ann.get("per_view_masks", {})
        for frame_idx, mask_data in per_view.items():
            fi = int(frame_idx)
            if 0 <= fi < num_frames:
                masks[p_idx, fi] = decode_mask(mask_data)
    return masks


def _build_gt_masks(
    gt_instance_maps: np.ndarray,
    object_ids: Sequence[int],
) -> np.ndarray:
    """Return [num_gt, N, H, W] boolean array for each GT instance."""
    num_gt = len(object_ids)
    N, H, W = gt_instance_maps.shape
    masks = np.zeros((num_gt, N, H, W), dtype=bool)
    for g_idx, oid in enumerate(object_ids):
        masks[g_idx] = (gt_instance_maps == oid)
    return masks


def _compute_pairwise_pooled_iou(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
) -> np.ndarray:
    """Return [num_gt, num_pred] pooled-IoU matrix."""
    num_gt = gt_masks.shape[0]
    num_pred = pred_masks.shape[0]
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)
    for g in range(num_gt):
        for p in range(num_pred):
            iou_matrix[g, p] = iou_binary_pooled(pred_masks[p], gt_masks[g])
    return iou_matrix


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

def hungarian_match(iou_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    One-to-one bilateral matching using the Hungarian algorithm.
    Returns list of (gt_idx, pred_idx, pooled_iou).
    """
    from scipy.optimize import linear_sum_assignment
    cost = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[int, int, float]] = []
    for g, p in zip(row_ind, col_ind):
        iou_val = float(iou_matrix[g, p])
        if iou_val > 0.0:
            matches.append((int(g), int(p), iou_val))
    return matches


# ---------------------------------------------------------------------------
# Precision / recall helpers
# ---------------------------------------------------------------------------

def compute_precision_recall(
    counts: Dict[float, Dict[str, int]],
) -> Dict[float, Dict[str, float]]:
    """Compute precision and recall from TP/FP/FN counts per threshold."""
    pr: Dict[float, Dict[str, float]] = {}
    for thr, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pr[thr] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    return pr


# ---------------------------------------------------------------------------
# Combined per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    annotations: Sequence[Dict[str, Any]],
    scene_inputs: SceneInputs,
    ignore_instance_ids: Iterable[int],
) -> Dict[str, Any]:
    """
    Run Hungarian matching + compute all metrics for one scene:
    T-mIoU, T-SR, pooled IoU, and frame-level precision/recall.

    All metrics are computed over ALL GT objects (matched and unmatched).
    Unmatched GT objects contribute 0 to pooled IoU, T-mIoU, and T-SR.
    """
    ignore_ids = {int(i) for i in ignore_instance_ids}
    object_ids = sorted(
        int(oid)
        for oid in np.unique(scene_inputs.gt_instance_maps)
        if int(oid) > 0 and int(oid) not in ignore_ids
    )
    if not object_ids:
        return {
            "num_gt_objects": 0,
            "num_predictions": len(annotations),
            "object_ids": [],
            "matches": [],
            "t_miou": math.nan,
            "t_sr": math.nan,
            "t_sr_at_half": math.nan,
            "precision_recall": {},
            "per_object_metrics": [],
            "skipped": True,
            "skip_reason": "No positive GT objects remain after filtering.",
        }

    N, H, W = scene_inputs.gt_instance_maps.shape
    gt_masks = _build_gt_masks(scene_inputs.gt_instance_maps, object_ids)
    pred_masks = _build_pred_masks(annotations, N, H, W)

    if pred_masks.shape[0] == 0:
        matches: List[Tuple[int, int, float]] = []
    else:
        iou_matrix = _compute_pairwise_pooled_iou(gt_masks, pred_masks)
        matches = hungarian_match(iou_matrix)

    match_by_gt = {g: (p, iou_val) for g, p, iou_val in matches}
    frame_names = scene_inputs.frame_names
    pr_counts: Dict[float, Dict[str, int]] = {
        thr: {"tp": 0, "fp": 0, "fn": 0} for thr in IOU_THRESHOLDS
    }
    per_object_metrics: List[Dict[str, Any]] = []

    for g_idx, object_id in enumerate(object_ids):
        if g_idx in match_by_gt:
            p_idx, pooled_iou_val = match_by_gt[g_idx]
            ann = annotations[p_idx]

            ious_visible: List[float] = []
            success = True
            success_at_half = True
            per_frame_metrics: List[Dict[str, Any]] = []

            for i in range(N):
                gt_exists = bool(gt_masks[g_idx, i].any())
                pred_exists = bool(pred_masks[p_idx, i].any())

                if gt_exists or pred_exists:
                    frame_iou = iou_binary(pred_masks[p_idx, i], gt_masks[g_idx, i])
                else:
                    frame_iou = 1.0

                if gt_exists and pred_exists:
                    for thr in IOU_THRESHOLDS:
                        if frame_iou >= thr:
                            pr_counts[thr]["tp"] += 1
                        else:
                            pr_counts[thr]["fp"] += 1
                            pr_counts[thr]["fn"] += 1
                elif gt_exists and not pred_exists:
                    for thr in IOU_THRESHOLDS:
                        pr_counts[thr]["fn"] += 1
                elif pred_exists and not gt_exists:
                    for thr in IOU_THRESHOLDS:
                        pr_counts[thr]["fp"] += 1

                if gt_exists:
                    if not pred_exists:
                        success = False
                        success_at_half = False
                    elif frame_iou < 0.5:
                        success_at_half = False
                    ious_visible.append(frame_iou)
                    per_frame_metrics.append({
                        "frame_index": int(i),
                        "frame_name": frame_names[i],
                        "iou": frame_iou,
                        "gt_area": int(gt_masks[g_idx, i].sum()),
                        "pred_area": int(pred_masks[p_idx, i].sum()),
                        "pred_nonempty": pred_exists,
                    })

            t_miou_val = float(np.mean(ious_visible)) if ious_visible else 0.0
            t_sr_val = int(success)
            t_sr_at_half_val = int(success_at_half)

            per_object_metrics.append({
                "object_id": int(object_id),
                "visible_frame_indices": [fm["frame_index"] for fm in per_frame_metrics],
                "matched_prediction_index": int(p_idx),
                "source_view": int(ann.get("source_view", -1)),
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
                "pooled_iou": pooled_iou_val,
                "t_miou": t_miou_val,
                "t_sr": t_sr_val,
                "t_sr_at_half": t_sr_at_half_val,
                "frame_metrics": per_frame_metrics,
            })
        else:
            per_frame_metrics = []
            for i in range(N):
                gt_exists = bool(gt_masks[g_idx, i].any())
                if gt_exists:
                    for thr in IOU_THRESHOLDS:
                        pr_counts[thr]["fn"] += 1
                    per_frame_metrics.append({
                        "frame_index": int(i),
                        "frame_name": frame_names[i],
                        "iou": 0.0,
                        "gt_area": int(gt_masks[g_idx, i].sum()),
                        "pred_area": 0,
                        "pred_nonempty": False,
                    })

            per_object_metrics.append({
                "object_id": int(object_id),
                "visible_frame_indices": [fm["frame_index"] for fm in per_frame_metrics],
                "matched_prediction_index": None,
                "source_view": None,
                "predicted_iou": 0.0,
                "stability_score": 0.0,
                "pooled_iou": 0.0,
                "t_miou": 0.0,
                "t_sr": 0,
                "t_sr_at_half": 0,
                "frame_metrics": per_frame_metrics,
            })

    pr = compute_precision_recall(pr_counts)

    t_miou = float(np.mean([obj["t_miou"] for obj in per_object_metrics]))
    t_sr = float(np.mean([obj["t_sr"] for obj in per_object_metrics]))
    t_sr_at_half = float(np.mean([obj["t_sr_at_half"] for obj in per_object_metrics]))

    return {
        "num_gt_objects": len(object_ids),
        "num_predictions": len(annotations),
        "object_ids": object_ids,
        "matches": [(object_ids[g], p, iou) for g, p, iou in matches],
        "t_miou": t_miou,
        "t_sr": t_sr,
        "t_sr_at_half": t_sr_at_half,
        "precision_recall": {str(thr): vals for thr, vals in sorted(pr.items())},
        "per_object_metrics": per_object_metrics,
        "skipped": False,
        "skip_reason": None,
    }


# ---------------------------------------------------------------------------
# Hyperparameter grid sweep
# ---------------------------------------------------------------------------

def sanitize_run_name(
    pred_iou_thresh: float,
    stability_score_thresh: float,
    box_nms_thresh: float,
    nms_iou_type: str = "box",
) -> str:
    return (
        f"piou_{pred_iou_thresh:.3f}_stab_{stability_score_thresh:.3f}"
        f"_nms_{box_nms_thresh:.3f}_{nms_iou_type}"
    ).replace(".", "p")


def evaluate_hyperparameter_grid(
    model: torch.nn.Module,
    scene_inputs: SceneInputs,
    scene_output_dir: Path,
    args: argparse.Namespace,
    prompt_groups: Optional[List[Dict[str, Any]]],
    prompt_source_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Sweep threshold combinations. Best run selected by highest T-SR then T-mIoU."""
    combinations = list(itertools.product(
        args.pred_iou_thresh_values,
        args.stability_score_thresh_values,
        args.box_nms_thresh_values,
    ))
    if not combinations:
        raise ValueError("No hyperparameter combinations were provided for the grid search.")

    results: List[Dict[str, Any]] = []
    best_primary: Optional[Dict[str, Any]] = None
    best_by_tmiou: Optional[Dict[str, Any]] = None

    def primary_key(result: Dict[str, Any]) -> Tuple[float, float, float]:
        if bool(result.get("skipped", False)):
            return (0.0, float("-inf"), float("-inf"))
        return (1.0, float(result["t_sr"]), float(result["t_miou"]))

    def tmiou_key(result: Dict[str, Any]) -> Tuple[float, float, float]:
        if bool(result.get("skipped", False)):
            return (0.0, float("-inf"), float("-inf"))
        return (1.0, float(result["t_miou"]), float(result["t_sr"]))

    for pred_iou_thresh, stability_score_thresh, box_nms_thresh in combinations:
        run_name = sanitize_run_name(
            pred_iou_thresh, stability_score_thresh, box_nms_thresh, args.nms_iou_type,
        )
        run_dir = scene_output_dir / "grid_runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"  [{scene_inputs.scene_id}] pred_iou={pred_iou_thresh:.3f}, "
            f"stability={stability_score_thresh:.3f}, box_nms={box_nms_thresh:.3f}"
        )

        generator = SamVGGTAutomaticMaskGenerator(
            model=model,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            nms_score=args.nms_score,
            nms_iou_type=args.nms_iou_type,
            output_mode="binary_mask",
        )
        annotations = generator.generate(
            scene_inputs.image_tensor,
            prompt_frame_idx=None,
            prompt_groups=prompt_groups,
        )

        metrics = evaluate_scene(
            annotations=annotations,
            scene_inputs=scene_inputs,
            ignore_instance_ids=args.ignore_instance_ids,
        )

        run_summary = {
            "scene_id": scene_inputs.scene_id,
            "run_name": run_name,
            "pred_iou_thresh": float(pred_iou_thresh),
            "stability_score_thresh": float(stability_score_thresh),
            "box_nms_thresh": float(box_nms_thresh),
            "num_predictions": int(metrics["num_predictions"]),
            "num_gt_objects": int(metrics["num_gt_objects"]),
            "num_matches": len(metrics["matches"]),
            "t_miou": float(metrics["t_miou"]),
            "t_sr": float(metrics["t_sr"]),
            "t_sr_at_half": float(metrics["t_sr_at_half"]),
            "precision_recall": metrics["precision_recall"],
            "per_match_pooled_ious": [iou for _, _, iou in metrics["matches"]],
            "skipped": bool(metrics["skipped"]),
            "skip_reason": metrics.get("skip_reason"),
            "per_object_metrics": metrics["per_object_metrics"],
            "annotation_metadata": [
                {k: v for k, v in ann.items() if k not in ("panoramic_mask", "per_view_masks")}
                for ann in annotations
            ],
            "artifacts": {"run_dir": str(run_dir)},
            "prompt_source": args.prompt_source,
            "prompt_sampling_method": args.prompt_sampling_method,
        }
        run_result = {
            **run_summary,
            "_annotations": annotations,
            "_generator": generator,
            "_prompt_groups": prompt_groups,
            "_prompt_source_summary": prompt_source_summary,
        }
        results.append(run_summary)

        if best_primary is None or primary_key(run_summary) > primary_key(best_primary):
            best_primary = run_result
        if best_by_tmiou is None or tmiou_key(run_summary) > tmiou_key(best_by_tmiou):
            best_by_tmiou = run_result

    assert best_primary is not None
    assert best_by_tmiou is not None
    results.sort(key=primary_key, reverse=True)
    return {
        "results": results,
        "best_primary": best_primary,
        "best_by_tmiou": best_by_tmiou,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_matched_objects(
    images: torch.Tensor,
    gt_instance_maps: np.ndarray,
    annotations: Sequence[Dict[str, Any]],
    per_object_metrics: Sequence[Dict[str, Any]],
    output_path: str,
    max_objects: int,
    prompt_groups_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping matched-object visualization.")
        return

    if not per_object_metrics:
        return

    selected_objects = sorted(
        per_object_metrics,
        key=lambda item: (item["t_miou"], item["t_sr"], item["object_id"]),
    )[:max_objects]
    view_count = images.shape[0]
    fig, axes = plt.subplots(
        len(selected_objects),
        view_count,
        figsize=(3.2 * view_count, 3.2 * len(selected_objects)),
        squeeze=False,
    )

    images_np = [
        images[v].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        for v in range(view_count)
    ]

    for row_idx, obj_metrics in enumerate(selected_objects):
        object_id = obj_metrics["object_id"]
        matched_index = obj_metrics["matched_prediction_index"]
        matched_ann = None if matched_index is None else annotations[matched_index]
        matched_prompt_group = None
        if matched_ann is not None and prompt_groups_by_id is not None:
            prompt_id = matched_ann.get("prompt_id")
            if prompt_id is not None:
                matched_prompt_group = prompt_groups_by_id.get(int(prompt_id))
        frame_metrics_by_idx = {
            fm["frame_index"]: fm for fm in obj_metrics["frame_metrics"]
        }

        for view_idx in range(view_count):
            ax = axes[row_idx, view_idx]
            base = images_np[view_idx].astype(np.float32)
            gt_mask = gt_instance_maps[view_idx] == object_id
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
            if matched_ann is not None:
                pred_mask_data = matched_ann.get("per_view_masks", {}).get(view_idx)
                if pred_mask_data is not None:
                    pred_mask = decode_mask(pred_mask_data)
            proposal_mask = None
            if matched_prompt_group is not None and int(matched_prompt_group["frame_index"]) == view_idx:
                proposal_mask = decode_mask(matched_prompt_group["proposal_rle"])

            if np.any(gt_mask):
                base[gt_mask] = base[gt_mask] * 0.35 + np.array([0, 255, 0], dtype=np.float32) * 0.65
            if np.any(pred_mask):
                base[pred_mask] = base[pred_mask] * 0.35 + np.array([255, 0, 0], dtype=np.float32) * 0.65
            if proposal_mask is not None and np.any(proposal_mask):
                base[proposal_mask] = base[proposal_mask] * 0.45 + np.array([0, 128, 255], dtype=np.float32) * 0.55

            ax.imshow(base.clip(0, 255).astype(np.uint8))
            if matched_prompt_group is not None and int(matched_prompt_group["frame_index"]) == view_idx:
                for point in matched_prompt_group["points"]:
                    ax.plot(
                        float(point[0]), float(point[1]),
                        marker="*", markersize=7,
                        color="yellow", markeredgecolor="black", markeredgewidth=0.4,
                    )

            gt_visible = bool(np.any(gt_mask))
            pred_exists = bool(np.any(pred_mask))
            frame_metric = frame_metrics_by_idx.get(view_idx)

            if frame_metric is None:
                title = f"GT {object_id} | view {view_idx}\nnot visible | Pred:{'Y' if pred_exists else 'N'}"
            else:
                title = (
                    f"GT {object_id} | view {view_idx}\n"
                    f"IoU={frame_metric['iou']:.3f} "
                    f"Pred:{'Y' if pred_exists else 'N'}"
                )

            if gt_visible and not pred_exists:
                border_color = "red"
            elif not gt_visible and pred_exists:
                border_color = "orange"
            elif gt_visible and pred_exists:
                border_color = "lime"
            else:
                border_color = None

            if border_color is not None:
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            else:
                ax.axis("off")

            ax.set_title(title, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Matched-object visualization saved to {output_path}")


def visualize_prompt_sources(
    images: torch.Tensor,
    prompt_groups: Sequence[Dict[str, Any]],
    output_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping prompt-source visualization.")
        return

    if not prompt_groups:
        return

    view_count = images.shape[0]
    fig, axes = plt.subplots(1, view_count, figsize=(3.5 * view_count, 3.5), squeeze=False)
    axes = axes[0]
    images_np = [
        images[v].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        for v in range(view_count)
    ]

    for view_idx in range(view_count):
        ax = axes[view_idx]
        base = images_np[view_idx].astype(np.float32)
        groups_for_view = [g for g in prompt_groups if int(g["frame_index"]) == view_idx]
        for group in groups_for_view:
            proposal_mask = decode_mask(group["proposal_rle"])
            if np.any(proposal_mask):
                base[proposal_mask] = base[proposal_mask] * 0.55 + np.array([0, 128, 255], dtype=np.float32) * 0.45
        ax.imshow(base.clip(0, 255).astype(np.uint8))
        for group in groups_for_view:
            for point in group["points"]:
                ax.plot(
                    float(point[0]), float(point[1]),
                    marker="*", markersize=4.5,
                    color="yellow", markeredgecolor="black", markeredgewidth=0.3,
                )
        ax.set_title(f"View {view_idx} ({len(groups_for_view)} prompts)", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Prompt-source visualization saved to {output_path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def summarize_run_result(run_result: Dict[str, Any], include_details: bool) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "scene_id": run_result["scene_id"],
        "run_name": run_result["run_name"],
        "pred_iou_thresh": float(run_result["pred_iou_thresh"]),
        "stability_score_thresh": float(run_result["stability_score_thresh"]),
        "box_nms_thresh": float(run_result["box_nms_thresh"]),
        "num_predictions": int(run_result["num_predictions"]),
        "num_gt_objects": int(run_result["num_gt_objects"]),
        "num_matches": int(run_result["num_matches"]),
        "t_miou": float(run_result["t_miou"]),
        "t_sr": float(run_result["t_sr"]),
        "t_sr_at_half": float(run_result["t_sr_at_half"]),
        "per_match_pooled_ious": run_result.get("per_match_pooled_ious", []),
        "precision_recall": run_result["precision_recall"],
        "skipped": bool(run_result.get("skipped", False)),
        "skip_reason": run_result.get("skip_reason"),
        "prompt_source": run_result.get("prompt_source"),
        "prompt_sampling_method": run_result.get("prompt_sampling_method"),
        "artifacts": dict(run_result.get("artifacts", {})),
    }
    if include_details:
        summary["per_object_metrics"] = run_result["per_object_metrics"]
        summary["annotation_metadata"] = run_result.get("annotation_metadata", [])
    return summary


def write_best_run_artifacts(
    scene_inputs: SceneInputs,
    scene_output_dir: Path,
    best_result: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, str]:
    best_dir = scene_output_dir / "best_run"
    best_dir.mkdir(parents=True, exist_ok=True)

    everything_vis_path = best_dir / "everything_mode.png"
    matched_vis_path = best_dir / "matched_objects.png"
    prompt_sources_vis_path = best_dir / "prompt_sources.png"
    annotations_meta_path = best_dir / "annotations_metadata.json"
    annotations_with_masks_path = best_dir / "annotations_with_masks.json"
    per_object_metrics_path = best_dir / "per_object_metrics.json"
    prompt_groups_path = best_dir / "prompt_groups.json"

    annotations = best_result["_annotations"]
    generator = best_result["_generator"]
    prompt_groups = best_result.get("_prompt_groups") or []
    prompt_groups_by_id = {
        int(group["prompt_id"]): group
        for group in prompt_groups
        if "prompt_id" in group
    }
    num_views = int(scene_inputs.image_tensor.shape[0])

    visualize_everything(
        scene_inputs.image_tensor,
        annotations,
        str(everything_vis_path),
        prompted_points=generator.prompted_points,
        points_after_iou=generator.points_after_iou,
        points_after_stability=generator.points_after_stability,
        max_views=num_views,
    )
    visualize_matched_objects(
        images=scene_inputs.image_tensor,
        gt_instance_maps=scene_inputs.gt_instance_maps,
        annotations=annotations,
        per_object_metrics=best_result["per_object_metrics"],
        output_path=str(matched_vis_path),
        max_objects=args.max_vis_objects,
        prompt_groups_by_id=prompt_groups_by_id,
    )
    if args.prompt_source != "grid" and prompt_groups:
        visualize_prompt_sources(
            images=scene_inputs.image_tensor,
            prompt_groups=prompt_groups,
            output_path=str(prompt_sources_vis_path),
        )

    annotation_meta = [
        {k: v for k, v in ann.items() if k not in ("panoramic_mask", "per_view_masks")}
        for ann in annotations
    ]
    annotations_with_masks = []
    for ann in annotations:
        serialized = {k: v for k, v in ann.items() if k not in ("panoramic_mask", "per_view_masks")}
        serialized["panoramic_mask"] = encode_mask_to_rle(decode_mask(ann["panoramic_mask"]))
        serialized["per_view_masks"] = {
            str(view_idx): encode_mask_to_rle(decode_mask(mask))
            for view_idx, mask in ann.get("per_view_masks", {}).items()
        }
        annotations_with_masks.append(serialized)
    with annotations_meta_path.open("w") as f:
        json.dump(annotation_meta, f, indent=2)
    with annotations_with_masks_path.open("w") as f:
        json.dump(annotations_with_masks, f, indent=2)
    with per_object_metrics_path.open("w") as f:
        json.dump(best_result["per_object_metrics"], f, indent=2)
    with prompt_groups_path.open("w") as f:
        json.dump(serialize_prompt_groups(prompt_groups), f, indent=2)

    artifacts = {
        "everything_mode_visualization": str(everything_vis_path),
        "matched_objects_visualization": str(matched_vis_path),
        "annotations_metadata": str(annotations_meta_path),
        "annotations_with_masks": str(annotations_with_masks_path),
        "per_object_metrics": str(per_object_metrics_path),
        "prompt_groups": str(prompt_groups_path),
    }
    if args.prompt_source != "grid" and prompt_groups:
        artifacts["prompt_sources_visualization"] = str(prompt_sources_vis_path)
    return artifacts


def build_scene_summary(
    scene_inputs: SceneInputs,
    best_result: Dict[str, Any],
    best_by_tmiou: Dict[str, Any],
    all_results: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    prompt_source_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "scene_id": scene_inputs.scene_id,
        "frame_names": scene_inputs.frame_names,
        "image_paths": scene_inputs.image_paths,
        "gt_paths": scene_inputs.gt_paths,
        "num_frames": len(scene_inputs.frame_names),
        "target_size": list(scene_inputs.image_tensor.shape[-2:]),
        "ignore_instance_ids": [int(i) for i in args.ignore_instance_ids],
        "matching_policy": "hungarian_bilateral_one_to_one_by_pooled_iou",
        "selection_policy": "highest_t_sr_then_t_miou",
        "points_per_side": int(args.points_per_side),
        "points_per_batch": int(args.points_per_batch),
        "nms_score": args.nms_score,
        "nms_iou_type": args.nms_iou_type,
        "prompt_source": args.prompt_source,
        "prompt_sampling_method": args.prompt_sampling_method,
        "prompt_points_per_mask": int(args.prompt_points_per_mask),
        "prompt_source_summary": prompt_source_summary,
        "hyperparameter_grid": {
            "pred_iou_thresh_values": [float(v) for v in args.pred_iou_thresh_values],
            "stability_score_thresh_values": [float(v) for v in args.stability_score_thresh_values],
            "box_nms_thresh_values": [float(v) for v in args.box_nms_thresh_values],
        },
        "best_run": summarize_run_result(best_result, include_details=True),
        "best_by_tmiou": summarize_run_result(best_by_tmiou, include_details=True),
        "all_runs": [summarize_run_result(r, include_details=False) for r in all_results],
    }


def write_grid_search_csv(results: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """Write per-scene grid search CSV with all metrics."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {
            "scene_id": r["scene_id"],
            "run_name": r["run_name"],
            "pred_iou_thresh": r["pred_iou_thresh"],
            "stability_score_thresh": r["stability_score_thresh"],
            "box_nms_thresh": r["box_nms_thresh"],
            "num_predictions": r["num_predictions"],
            "num_gt_objects": r["num_gt_objects"],
            "num_matches": r["num_matches"],
            "t_miou": r["t_miou"],
            "t_sr": r["t_sr"],
            "t_sr_at_half": r["t_sr_at_half"],
            "prompt_source": r.get("prompt_source", ""),
            "prompt_sampling_method": r.get("prompt_sampling_method", ""),
            "skipped": r["skipped"],
            "skip_reason": r.get("skip_reason", ""),
        }
        for thr_str, vals in sorted(r.get("precision_recall", {}).items()):
            row[f"precision@{thr_str}"] = vals["precision"]
            row[f"recall@{thr_str}"] = vals["recall"]
            row[f"tp@{thr_str}"] = vals["tp"]
            row[f"fp@{thr_str}"] = vals["fp"]
            row[f"fn@{thr_str}"] = vals["fn"]
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        write_csv(output_path, rows, fieldnames)


def run_scene_benchmark(
    model: torch.nn.Module,
    proposal_sam_model: Optional[torch.nn.Module],
    scene_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run full benchmark pipeline for a single scene."""
    scene_output_dir = output_dir / scene_dir.name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    scene_inputs = load_scene_inputs(scene_dir, tuple(args.target_size))
    print(f"  Loaded {len(scene_inputs.frame_names)} frames at {tuple(args.target_size)}")

    prompt_groups, prompt_source_summary = build_dense_prompt_groups(
        scene_inputs=scene_inputs,
        proposal_sam_model=proposal_sam_model,
        scene_output_dir=scene_output_dir,
        args=args,
    )
    if prompt_groups is not None:
        print(f"  Built {len(prompt_groups)} prompt groups from SAM dense masks")
    else:
        print(f"  Using grid prompts ({args.points_per_side}x{args.points_per_side})")

    sweep = evaluate_hyperparameter_grid(
        model=model,
        scene_inputs=scene_inputs,
        scene_output_dir=scene_output_dir,
        args=args,
        prompt_groups=prompt_groups,
        prompt_source_summary=prompt_source_summary,
    )

    best_result = sweep["best_primary"]
    artifacts = write_best_run_artifacts(
        scene_inputs=scene_inputs,
        scene_output_dir=scene_output_dir,
        best_result=best_result,
        args=args,
    )

    write_grid_search_csv(sweep["results"], scene_output_dir / "grid_search_results.csv")

    summary = build_scene_summary(
        scene_inputs=scene_inputs,
        best_result=best_result,
        best_by_tmiou=sweep["best_by_tmiou"],
        all_results=sweep["results"],
        args=args,
        prompt_source_summary=prompt_source_summary,
    )
    summary["best_run"]["artifacts"].update(artifacts)

    with (scene_output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (scene_output_dir / "grid_search_results.json").open("w") as f:
        json.dump(summary["all_runs"], f, indent=2)

    _match_ious = best_result.get("per_match_pooled_ious", [])
    _micro_piou = sum(_match_ious) / best_result["num_gt_objects"] if best_result["num_gt_objects"] > 0 else math.nan
    print(
        f"  Best: T-SR={best_result['t_sr']:.4f}, T-SR@0.5={best_result['t_sr_at_half']:.4f}, "
        f"T-mIoU={best_result['t_miou']:.4f}, "
        f"pooled_iou={_micro_piou:.4f}, "
        f"matches={best_result['num_matches']}/{best_result['num_gt_objects']} GT, "
        f"{best_result['num_predictions']} preds"
    )
    for thr_str in sorted(best_result.get("precision_recall", {}).keys()):
        vals = best_result["precision_recall"][thr_str]
        print(f"    IoU>={thr_str}: P={vals['precision']:.3f} R={vals['recall']:.3f}")

    _num_gt = int(best_result["num_gt_objects"])
    return {
        "scene_id": scene_dir.name,
        "best_t_sr": float(best_result["t_sr"]),
        "best_t_sr_at_half": float(best_result["t_sr_at_half"]),
        "best_t_miou": float(best_result["t_miou"]),
        "mean_pooled_iou": float(sum(_match_ious) / _num_gt) if _num_gt > 0 else math.nan,
        "best_run_name": best_result["run_name"],
        "num_gt_objects": _num_gt,
        "num_matches": int(best_result["num_matches"]),
        "per_match_pooled_ious": best_result.get("per_match_pooled_ious", []),
        "precision_recall": best_result.get("precision_recall", {}),
        "summary_path": str(scene_output_dir / "summary.json"),
        "per_object_metrics": best_result["per_object_metrics"],
    }


def write_dataset_summary(
    dataset_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Aggregate scene results into a dataset-level summary."""
    all_obj_mious: List[float] = []
    all_obj_srs: List[float] = []
    all_obj_srs_at_half: List[float] = []
    all_match_ious: List[float] = []
    total_gt_objects = 0

    pr_by_threshold: Dict[str, Dict[str, Any]] = {}

    for item in dataset_results:
        total_gt_objects += item.get("num_gt_objects", 0)
        all_match_ious.extend(item.get("per_match_pooled_ious", []))
        for obj in item.get("per_object_metrics", []):
            all_obj_mious.append(float(obj["t_miou"]))
            all_obj_srs.append(float(obj["t_sr"]))
            all_obj_srs_at_half.append(float(obj["t_sr_at_half"]))
        for thr_str, vals in item.get("precision_recall", {}).items():
            if thr_str not in pr_by_threshold:
                pr_by_threshold[thr_str] = {"tp": 0, "fp": 0, "fn": 0}
            pr_by_threshold[thr_str]["tp"] += vals.get("tp", 0)
            pr_by_threshold[thr_str]["fp"] += vals.get("fp", 0)
            pr_by_threshold[thr_str]["fn"] += vals.get("fn", 0)

    mean_pr: Dict[str, Dict[str, Any]] = {}
    for thr_str, agg in sorted(pr_by_threshold.items()):
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        mean_pr[thr_str] = {
            "mean_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "mean_recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
        }

    mean_pooled_iou = float(sum(all_match_ious) / total_gt_objects) if total_gt_objects > 0 else math.nan

    scene_results_compact = []
    for item in dataset_results:
        scene_results_compact.append({
            k: v for k, v in item.items() if k != "per_object_metrics"
        })

    payload: Dict[str, Any] = {
        "num_scenes": len(dataset_results),
        "mean_best_t_sr": float(np.mean(all_obj_srs)) if all_obj_srs else math.nan,
        "mean_best_t_sr_at_half": float(np.mean(all_obj_srs_at_half)) if all_obj_srs_at_half else math.nan,
        "mean_best_t_miou": float(np.mean(all_obj_mious)) if all_obj_mious else math.nan,
        "mean_pooled_iou": mean_pooled_iou,
        "total_gt_objects": total_gt_objects,
        "total_matched_objects": len(all_match_ious),
        "mean_precision_recall_by_threshold": mean_pr,
        "scene_results": scene_results_compact,
    }

    with (output_dir / "dataset_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    pr_csv_rows = []
    for thr_str, vals in sorted(mean_pr.items()):
        pr_csv_rows.append({
            "iou_threshold": thr_str,
            "mean_precision": vals["mean_precision"],
            "mean_recall": vals["mean_recall"],
            "total_tp": vals["total_tp"],
            "total_fp": vals["total_fp"],
            "total_fn": vals["total_fn"],
        })
    if pr_csv_rows:
        write_csv(
            output_dir / "precision_recall_summary.csv",
            pr_csv_rows,
            ["iou_threshold", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combined SamVGGT 3D Tracking Benchmark. "
            "Evaluates T-mIoU, T-SR, pooled IoU, and precision/recall."
        ),
    )
    parser.add_argument(
        "--benchmark_root", type=str,
        required=True,
        help="Root containing IGGT/ScanNet++ scenes.",
    )
    parser.add_argument("--scene_ids", nargs="*", default=None,
        help="Optional scene IDs to evaluate. Default: discover all valid scenes.")
    parser.add_argument("--sam_v_ckpt", type=str, required=True,
        help="Path to the trained SamVGGT checkpoint (.pth).")
    parser.add_argument("--output_dir", type=str, default="./3dtracking_benchmark_output",
        help="Directory where benchmark outputs will be written.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--target_size", nargs=2, type=int, default=(1024, 1024),
        metavar=("HEIGHT", "WIDTH"),
        help="Resize RGB images and GT maps to this resolution.",
    )

    parser.add_argument(
        "--prompt_source", type=str, default="sam_dense_masks",
        choices=["grid", "sam_dense_masks", "sam_dense_masks_ls"],
        help="Source of prompts for SamVGGT decoding. 'sam_dense_masks' uses the "
             "sam-hq AMG (all multimask levels flattened); 'sam_dense_masks_ls' uses "
             "the LangSplat-modified AMG (level selected by --ls_level).",
    )
    parser.add_argument(
        "--ls_level", type=str, default="large",
        choices=["medium", "large", "both"],
        help="Which LangSplat AMG hierarchy level to keep when "
             "--prompt_source sam_dense_masks_ls (medium=part, large=whole, "
             "both=medium+large). Ignored for other prompt sources.",
    )
    parser.add_argument(
        "--prompt_sampling_method", type=str, default="pole_plus_diverse",
        choices=["centroid", "snapped_centroid", "random_interior",
                 "pole_plus_random", "pole_plus_diverse"],
        help="How to sample prompt points from each 2D proposal mask.",
    )
    parser.add_argument("--prompt_points_per_mask", type=int, default=3,
        help="Number of positive point prompts to sample from each proposal mask.")

    parser.add_argument("--points_per_side", type=int, default=16)
    parser.add_argument("--points_per_batch", type=int, default=8)

    parser.add_argument(
        "--proposal_sam_checkpoint", type=str,
        default=str(REPO_ROOT / "submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth"),
        help="Checkpoint used for dense per-frame SAM proposals.",
    )

    parser.add_argument("--pred_iou_thresh_values", nargs="+", type=float, default=[0.40],
        help="Grid-search values for predicted IoU filtering.")
    parser.add_argument("--stability_score_thresh_values", nargs="+", type=float, default=[0.2],
        help="Grid-search values for stability filtering.")
    parser.add_argument("--box_nms_thresh_values", nargs="+", type=float, default=[0.95],
        help="Grid-search values for panoramic box NMS.")

    parser.add_argument("--nms_score", type=str, default="iou_preds",
        choices=["stability_score", "iou_preds"],
        help="MaskData key used to rank masks during panoramic NMS in the SamVGGT "
             "generator. 'iou_preds' (default, recommended; the IoU-prediction "
             "head, supervised by the IoU loss in the ScanNet++ v2 finetune) or "
             "'stability_score' (derived from the mask logits).")

    parser.add_argument("--nms_iou_type", type=str, default="mask",
        choices=["box", "mask"],
        help="Overlap metric for panoramic NMS dedup. 'mask' (default) uses "
             "exact mask-IoU on the panoramic masks; 'box' uses fast box-IoU "
             "on the panoramic bounding boxes.")

    parser.add_argument("--ignore_instance_ids", nargs="*", type=int, default=[],
        help="GT instance IDs to ignore during evaluation.")
    parser.add_argument("--max_vis_objects", type=int, default=16,
        help="Maximum number of GT objects to show in the benchmark visualization.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Before any work: a run that crashes should still say what it was running.
    write_run_provenance(output_dir, config=vars(args))

    scene_dirs = discover_scene_dirs(benchmark_root, args.scene_ids)
    print(f"Discovered {len(scene_dirs)} scene(s) to evaluate.")

    device = torch.device(args.device)
    print("Building SamVGGT model ...")
    model = build_sam_vggt(device=str(device))
    print(f"Loading checkpoint: {args.sam_v_ckpt}")
    load_checkpoint(model, args.sam_v_ckpt, device)

    proposal_sam_model = None
    if args.prompt_source in ("sam_dense_masks", "sam_dense_masks_ls"):
        registry = (
            _load_langsplat_sam()[1]
            if args.prompt_source == "sam_dense_masks_ls"
            else sam_model_registry
        )
        print(
            f"Building proposal SAM model ({PROPOSAL_SAM_MODEL_TYPE}, "
            f"{args.prompt_source}) from {args.proposal_sam_checkpoint} ..."
        )
        proposal_sam_model = registry[PROPOSAL_SAM_MODEL_TYPE](
            checkpoint=args.proposal_sam_checkpoint
        )
        proposal_sam_model.to(device=device).eval()

    dataset_results: List[Dict[str, Any]] = []

    for scene_dir in scene_dirs:
        print(f"\n=== Scene: {scene_dir.name} ===")
        result = run_scene_benchmark(
            model=model,
            proposal_sam_model=proposal_sam_model,
            scene_dir=scene_dir,
            output_dir=output_dir,
            args=args,
        )
        dataset_results.append(result)

    write_dataset_summary(dataset_results, output_dir)

    all_obj_mious: List[float] = []
    all_obj_srs: List[float] = []
    all_obj_srs_at_half: List[float] = []
    all_match_ious: List[float] = []
    total_gt = 0
    for item in dataset_results:
        total_gt += item.get("num_gt_objects", 0)
        all_match_ious.extend(item.get("per_match_pooled_ious", []))
        for obj in item.get("per_object_metrics", []):
            all_obj_mious.append(float(obj["t_miou"]))
            all_obj_srs.append(float(obj["t_sr"]))
            all_obj_srs_at_half.append(float(obj["t_sr_at_half"]))

    mean_pooled_iou = float(sum(all_match_ious) / total_gt) if total_gt > 0 else math.nan
    mean_tsr = float(np.mean(all_obj_srs)) if all_obj_srs else math.nan
    mean_tsr_at_half = float(np.mean(all_obj_srs_at_half)) if all_obj_srs_at_half else math.nan
    mean_tmiou = float(np.mean(all_obj_mious)) if all_obj_mious else math.nan

    print(f"\n{'='*60}")
    print(f"Dataset summary ({len(dataset_results)} scenes, {total_gt} GT objects)")
    print(f"  T-SR  (micro-avg over objects):      {mean_tsr:.4f}")
    print(f"  T-SR@0.5 (micro-avg over objects):   {mean_tsr_at_half:.4f}")
    print(f"  T-mIoU (micro-avg over objects):     {mean_tmiou:.4f}")
    print(f"  Pooled IoU (micro-avg all objects):   {mean_pooled_iou:.4f}  ({len(all_match_ious)} matched / {total_gt} GT)")
    print(f"{'='*60}")

    print(f"\nDataset summary written to {output_dir / 'dataset_summary.json'}")
    print(f"Precision/recall summary written to {output_dir / 'precision_recall_summary.csv'}")


if __name__ == "__main__":
    main()
