#!/usr/bin/env python3
"""
Demo script for Multi-View Everything Mode with SamVGGT.

Loads a set of multi-view images from a directory (or a Hypersim scene),
runs the automatic mask generator, and saves a visualization grid showing
all detected objects across views.

Usage examples:

    # From a directory of images (sorted alphabetically as views)
    python everything_mode_demo.py \
        --image_dir /path/to/scene_views/ \
        --sam_v_ckpt /path/to/checkpoint.pth \
        --output_dir ./everything_output

    # From a Hypersim scene directory with explicit frame stems
    python everything_mode_demo.py \
        --image_dir $HYPERSIM_ROOT/ai_001_001/color/ \
        --sam_v_ckpt /path/to/checkpoint.pth \
        --frame_count 16 \
        --output_dir ./everything_output
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
SAM_HQ_ROOT = REPO_ROOT / "submodules" / "sam-hq"
VGGT_ROOT = REPO_ROOT / "submodules" / "vggt"
for p in [str(SAM_HQ_ROOT), str(VGGT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.checkpoint import (
    SLIM_CHECKPOINT_FORMAT,
    is_partial_checkpoint,
    load_partial_checkpoint,
)
from model.sam_vggt_model import build_sam_vggt
from masks.automatic_mask_generator import SamVGGTAutomaticMaskGenerator


# ======================================================================
# Image loading
# ======================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_images_from_dir(
    image_dir: str,
    frame_count: int | None = None,
    target_size: tuple[int, int] = (1024, 1024),
) -> torch.Tensor:
    """
    Load images from a directory and return a ``[N, 3, H, W]`` tensor
    in [0, 255] float range, resized to *target_size*.
    """
    folder = Path(image_dir)
    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if frame_count is not None:
        step = max(1, len(paths) // frame_count)
        paths = paths[::step][:frame_count]

    tensors: List[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32)                    # [H, W, 3]
        t = torch.from_numpy(arr).permute(2, 0, 1)               # [3, H, W]
        tensors.append(t)

    images = torch.stack(tensors, dim=0)                          # [N, 3, H, W]
    if images.shape[-2:] != target_size:
        images = F.interpolate(
            images, size=target_size, mode="bilinear", align_corners=False,
        )
    return images


# ======================================================================
# Checkpoint loading
# ======================================================================

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if is_partial_checkpoint(ckpt):
        # Released checkpoint, or one this repo's trainer wrote -- either way the
        # frozen weights are absent by design and verified on load.
        load_partial_checkpoint(model, ckpt)
        model.to(device).eval()
        return
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()


# ======================================================================
# Visualization
# ======================================================================

def random_color(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(50, 255, size=3).astype(np.uint8)


def _scatter_points_on_ax(ax, prompted_points: list[dict], view_idx: int, marker_size: float = 6):
    """Draw prompted points for a given view on a matplotlib axis."""
    for pp in prompted_points:
        if pp["view"] != view_idx:
            continue
        x, y = pp["point"]
        ax.plot(x, y, marker=".", markersize=marker_size, color="lime",
                markeredgecolor="black", markeredgewidth=0.3)


def visualize_everything(
    images: torch.Tensor,
    annotations: list[dict],
    output_path: str,
    prompted_points: list[dict] | None = None,
    points_after_iou: list[dict] | None = None,
    points_after_stability: list[dict] | None = None,
    max_views: int = 8,
) -> None:
    """
    Save a visualization grid: one column per view.

    Rows:
        0 – original images with prompted points overlaid
        1 – prompts that survived predicted-IoU filtering
        2 – prompts that survived stability filtering
        3 – single overlay of every predicted object, each with its own color
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping visualization.")
        return

    if prompted_points is None:
        prompted_points = []
    if points_after_iou is None:
        points_after_iou = []
    if points_after_stability is None:
        points_after_stability = []

    N = images.shape[0]
    n_views = min(N, max_views)
    H, W = images.shape[-2:]

    imgs_np = []
    for v in range(n_views):
        arr = images[v].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        imgs_np.append(arr)

    n_objects = len(annotations)
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_views, figsize=(3 * n_views, 3 * n_rows))
    if n_rows == 1 and n_views == 1:
        axes = np.array([[axes]])
    elif getattr(axes, "ndim", 0) == 1:
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    # Row 0: original images + prompted points
    pts_per_view = {}
    for pp in prompted_points:
        pts_per_view.setdefault(pp["view"], []).append(pp)
    for v in range(n_views):
        axes[0, v].imshow(imgs_np[v])
        n_pts_v = len(pts_per_view.get(v, []))
        _scatter_points_on_ax(axes[0, v], prompted_points, v)
        axes[0, v].set_title(f"View {v} ({n_pts_v} pts)", fontsize=8)
        axes[0, v].axis("off")

    # Row 1: prompts kept after IoU filtering
    kept_per_view = {}
    for pp in points_after_iou:
        kept_per_view.setdefault(pp["view"], []).append(pp)
    for v in range(n_views):
        axes[1, v].imshow(imgs_np[v])
        n_kept_v = len(kept_per_view.get(v, []))
        _scatter_points_on_ax(axes[1, v], points_after_iou, v, marker_size=5)
        axes[1, v].set_title(f"After IoU ({n_kept_v} pts)", fontsize=8)
        axes[1, v].axis("off")

    # Row 2: prompts kept after stability filtering
    stab_per_view = {}
    for pp in points_after_stability:
        stab_per_view.setdefault(pp["view"], []).append(pp)
    for v in range(n_views):
        axes[2, v].imshow(imgs_np[v])
        n_stab_v = len(stab_per_view.get(v, []))
        _scatter_points_on_ax(axes[2, v], points_after_stability, v, marker_size=5)
        axes[2, v].set_title(f"After stability ({n_stab_v} pts)", fontsize=8)
        axes[2, v].axis("off")

    # Row 3: overlay of all masks + points of surviving masks
    # Use a per-pixel winner-takes-color rule (highest predicted_iou),
    # so overlapping masks remain visually distinct instead of repeatedly
    # blending into muddy colors.
    overlays = [img.copy().astype(np.float32) for img in imgs_np]
    for v in range(n_views):
        best_score = np.full((H, W), -np.inf, dtype=np.float32)
        color_canvas = np.zeros((H, W, 3), dtype=np.float32)
        covered = np.zeros((H, W), dtype=bool)

        for obj_idx, ann in enumerate(annotations):
            if v not in ann["per_view_masks"]:
                continue
            mask = ann["per_view_masks"][v]
            if isinstance(mask, dict):
                from segment_anything.utils.amg import rle_to_mask
                mask = rle_to_mask(mask)
            mask = mask.astype(bool)

            score = float(ann.get("predicted_iou", 0.0))
            update = mask & (score > best_score)
            if not np.any(update):
                continue

            color = random_color(obj_idx + 42).astype(np.float32)
            best_score[update] = score
            color_canvas[update] = color
            covered |= mask

        overlays[v][covered] = overlays[v][covered] * 0.35 + color_canvas[covered] * 0.65

    for v in range(n_views):
        axes[3, v].imshow(overlays[v].clip(0, 255).astype(np.uint8))
        axes[3, v].set_title(f"All masks ({n_objects})", fontsize=8)
        axes[3, v].axis("off")
    # mark surviving prompt points on the overlay row
    for ann in annotations:
        sv = ann["source_view"]
        if sv < n_views:
            point_coords = ann.get("point_coords", [])
            if not point_coords:
                continue
            if isinstance(point_coords[0], (int, float)):
                point_list = [point_coords]
            else:
                point_list = point_coords
            for x, y in point_list:
                axes[3, sv].plot(x, y, marker="*", markersize=8, color="red",
                                 markeredgecolor="white", markeredgewidth=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-View Everything Mode demo for SamVGGT",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing multi-view images of a single scene.",
    )
    parser.add_argument(
        "--sam_v_ckpt", type=str, required=True,
        help="Path to the trained SamVGGT checkpoint (.pth).",
    )
    parser.add_argument("--frame_count", type=int, default=16)
    parser.add_argument(
        "--prompt_frame_idx",
        type=int,
        default=None,
        help=(
            "Optional source-view index to prompt only one frame. "
            "Default: prompt on every loaded frame."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./everything_output")

    # Generator parameters
    parser.add_argument("--points_per_side", type=int, default=16,
                        help="Grid resolution per frame (default 16 -> 256 pts/frame)")
    parser.add_argument("--points_per_batch", type=int, default=8)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.40)
    parser.add_argument("--stability_score_thresh", type=float, default=0.80)
    parser.add_argument("--box_nms_thresh", type=float, default=0.7)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Load images ----
    print(f"Loading images from {args.image_dir} ...")
    images = load_images_from_dir(
        args.image_dir,
        frame_count=args.frame_count,
    )
    N = images.shape[0]
    print(f"  Loaded {N} views at {images.shape[-2:]}")

    # ---- Build model ----
    print("Building SamVGGT model ...")
    model = build_sam_vggt(device=str(device))
    print(f"Loading checkpoint: {args.sam_v_ckpt}")
    load_checkpoint(model, args.sam_v_ckpt, device)

    # ---- Create generator ----
    generator = SamVGGTAutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
    )

    # ---- Generate ----
    print("Running everything mode ...")
    annotations = generator.generate(
        images,
        prompt_frame_idx=args.prompt_frame_idx,
    )
    print(f"  Found {len(annotations)} objects across {N} views")

    # ---- Summary ----
    for i, ann in enumerate(annotations):
        views = sorted(ann["per_view_masks"].keys())
        print(
            f"  Object {i}: IoU={ann['predicted_iou']:.3f}, "
            f"stability={ann['stability_score']:.3f}, "
            f"area={ann['area']}, "
            f"views={views}, "
            f"source=view {ann['source_view']}"
        )

    # ---- Visualize ----
    vis_path = os.path.join(args.output_dir, "everything_mode.png")
    visualize_everything(images, annotations, vis_path,
                         prompted_points=generator.prompted_points,
                         points_after_iou=generator.points_after_iou,
                         points_after_stability=generator.points_after_stability)

    # ---- Save annotations (metadata only, no large masks) ----
    import json

    meta = []
    for ann in annotations:
        meta.append({
            k: v for k, v in ann.items()
            if k not in ("panoramic_mask", "per_view_masks")
        })
    meta_path = os.path.join(args.output_dir, "annotations.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Annotation metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
