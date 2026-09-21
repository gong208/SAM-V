#!/usr/bin/env python3
"""
Run SAM2 and SamVGGT inference on custom color images with hardcoded point prompts.

No ground-truth masks are needed. The script loads images from a directory,
builds a sample dict with user-specified point prompts, runs both models,
and saves overlay visualizations.

Usage:
    python infer_custom_images.py \
        --image_dir /path/to/images \
        --sam_v_ckpt /path/to/sam_vggt.pth \
        --sam2_ckpt /path/to/sam2.pt

    # SamVGGT only (no SAM2 checkpoint or inference):
    python infer_custom_images.py \
        --image_dir /path/to/images \
        --sam_v_ckpt /path/to/sam_vggt.pth \
        --no_sam2
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
SAM2_ROOT = REPO_ROOT / "sam2"
SAM_HQ_ROOT = REPO_ROOT / "submodules" / "sam-hq"
VGGT_ROOT = REPO_ROOT / "submodules" / "vggt"
for p in (SAM2_ROOT, SAM_HQ_ROOT, VGGT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.compare_baseline_sam2 import (
    TARGET_SIZE,
    infer_sam2,
    infer_samvggt,
    load_samvggt_checkpoint,
    set_seed,
)

# ────────────────────────────────────────────────────────────────────────────
# HARDCODED PROMPTS
#
# Edit this section to specify your point prompts.
#   frame_index: which image (0-based index in sorted order) the point is on
#   x, y:        pixel coordinates in the *original* image (will be rescaled
#                to TARGET_SIZE automatically)
#   label:       1 = positive (foreground), 0 = negative (background)
#
# A 5x5 grid of points is automatically generated around each prompt using
# GRID_SPACING (in original-image pixels). All grid points inherit the same
# frame_index and label as the seed prompt.
# ────────────────────────────────────────────────────────────────────────────
HARDCODED_PROMPTS: list[dict] = [
    {"frame_index": 0, "x": 688, "y": 135, "label": 1},
]

GRID_SPACING = 1  # pixels in original image coordinates

RGB_FRAME_STEM_RE = re.compile(r"^frame_\d+$")
RGB_NUMERIC_STEM_RE = re.compile(r"^\d+$")


def _is_canonical_rgb_frame(image_path: Path) -> bool:
    if image_path.suffix.lower() != ".jpg":
        return False
    stem = image_path.stem
    return (
        RGB_FRAME_STEM_RE.fullmatch(stem) is not None
        or RGB_NUMERIC_STEM_RE.fullmatch(stem) is not None
    )


def _expand_prompts_with_grid(
    prompts: list[dict],
    spacing: int,
) -> list[dict]:
    """For each prompt, generate a 5x5 grid of points centred on it."""
    expanded: list[dict] = []
    offsets = [-2, -1, 0, 1, 2]
    for p in prompts:
        cx, cy = p["x"], p["y"]
        for dy in offsets:
            for dx in offsets:
                expanded.append({
                    "frame_index": p["frame_index"],
                    "x": cx + dx * spacing,
                    "y": cy + dy * spacing,
                    "label": p["label"],
                })
    return expanded


def _load_images_and_gt_from_scene_images_dir(
    image_dir: Path,
) -> tuple[list[str], list[np.ndarray], list[str]]:
    """
    Discover canonical scene frames and GT masks in <scene_id>/images:
      - RGB: frame_XXXXXX.jpg or XXXXXX.jpg
      - GT:  frame_XXXXXX_label.npy or XXXXXX_label.npy
    """
    image_paths = [
        p for p in sorted(image_dir.glob("*.jpg"))
        if _is_canonical_rgb_frame(p)
    ]
    if not image_paths:
        raise FileNotFoundError(
            f"No canonical RGB frames found in {image_dir}. "
            "Expected files like frame_000000.jpg or 000000.jpg."
        )

    gt_paths: list[Path] = []
    missing_gt: list[str] = []
    for image_path in image_paths:
        gt_path = image_dir / f"{image_path.stem}_label.npy"
        if not gt_path.is_file():
            missing_gt.append(str(gt_path))
        gt_paths.append(gt_path)
    if missing_gt:
        preview = "\n".join(missing_gt[:10])
        suffix = "" if len(missing_gt) <= 10 else f"\n... and {len(missing_gt) - 10} more"
        raise FileNotFoundError(
            "Missing GT mask files for scene images:\n"
            f"{preview}{suffix}"
        )

    stems = [p.stem for p in image_paths]
    arrays = [np.array(Image.open(p).convert("RGB"), dtype=np.float32) for p in image_paths]
    return stems, arrays, [str(p) for p in gt_paths]


def build_sample_from_image_arrays(
    stems: list[str],
    arrays: list[np.ndarray],
    prompts: list[dict],
    gt_mask_paths: list[str] | None = None,
) -> dict:
    """Build a SamVGGT/SAM2 sample from RGB arrays and point prompts.

    Prompt coordinates are expressed in each source image's original pixel
    coordinates and are scaled to TARGET_SIZE per prompted frame.
    """
    if not arrays:
        raise ValueError("At least one image is required.")
    if len(stems) != len(arrays):
        raise ValueError("stems and arrays must have the same length.")
    if not prompts:
        raise ValueError("At least one point prompt is required.")

    images: list[torch.Tensor] = []
    original_sizes: list[tuple[int, int]] = []
    for i, arr in enumerate(arrays):
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Image {i} must be an RGB array with shape [H,W,3].")
        orig_h, orig_w = arr.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Image {i} has invalid size: {orig_w}x{orig_h}.")
        original_sizes.append((orig_h, orig_w))

        t = torch.from_numpy(arr.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)
        if t.shape[-2:] != TARGET_SIZE:
            t = F.interpolate(t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
        images.append(t.squeeze(0))

    H, W = TARGET_SIZE
    coords = []
    labels = []
    frame_indices = []
    for idx, p in enumerate(prompts):
        try:
            frame_index = int(p["frame_index"])
            x = float(p["x"])
            y = float(p["y"])
            label = int(p["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Prompt {idx} must contain frame_index, x, y, and label.") from exc

        if frame_index < 0 or frame_index >= len(arrays):
            raise ValueError(f"Prompt {idx} frame_index out of range: {frame_index}.")
        if label not in (0, 1):
            raise ValueError(f"Prompt {idx} label must be 0 or 1.")

        orig_h, orig_w = original_sizes[frame_index]
        if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
            raise ValueError(
                f"Prompt {idx} coordinate ({x:.1f}, {y:.1f}) is outside "
                f"frame {frame_index} ({orig_w}x{orig_h})."
            )

        coords.append([x * (W / orig_w), y * (H / orig_h)])
        labels.append(label)
        frame_indices.append(frame_index)

    return {
        "images": torch.stack(images, dim=0).float(),
        "frame_stems": stems,
        "gt_mask_paths": gt_mask_paths or [],
        "point_coords": torch.tensor(coords, dtype=torch.float32),
        "point_labels": torch.tensor(labels, dtype=torch.long),
        "point_frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        "original_sizes": original_sizes,
        "H": H,
        "W": W,
    }


def build_custom_sample(
    image_dir: Path,
    prompts: list[dict],
) -> dict:
    """Build a sample dict matching the format expected by infer_samvggt / infer_sam2."""
    stems, arrays, gt_mask_paths = _load_images_and_gt_from_scene_images_dir(image_dir)
    return build_sample_from_image_arrays(stems, arrays, prompts, gt_mask_paths=gt_mask_paths)


def _grid_shape(n: int, max_cols: int = 4) -> tuple[int, int]:
    cols = min(max_cols, max(1, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _to_uint8(images_nchw: torch.Tensor) -> np.ndarray:
    imgs = images_nchw.detach().cpu()
    if imgs.dtype.is_floating_point:
        imgs = imgs.clamp(0, 255)
    return imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)


def _draw_points(ax, coords_xy: np.ndarray, labels: np.ndarray) -> None:
    pos = labels == 1
    neg = labels == 0
    if pos.any():
        ax.scatter(coords_xy[pos, 0], coords_xy[pos, 1],
                   c="yellow", marker="*", s=100, edgecolors="black", linewidths=0.7)
    if neg.any():
        ax.scatter(coords_xy[neg, 0], coords_xy[neg, 1],
                   c="red", marker="*", s=100, edgecolors="white", linewidths=0.7)


def save_overlay(
    sample: dict,
    pred_sam2: np.ndarray | None,
    pred_samvggt: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    imgs = _to_uint8(sample["images"])
    N = imgs.shape[0]
    stems = sample["frame_stems"]
    use_sam2 = pred_sam2 is not None

    coords = sample["point_coords"].detach().cpu().numpy()
    labels = sample["point_labels"].detach().cpu().numpy().astype(int)
    frame_idx = sample["point_frame_indices"].detach().cpu().numpy().astype(int)

    frame_rows, ncols = _grid_shape(N, max_cols=4)
    row_blocks = 3 if use_sam2 else 2  # original | [SAM2] | SamVGGT
    total_rows = frame_rows * row_blocks
    fig, axes = plt.subplots(total_rows, ncols, figsize=(4 * ncols, 3.5 * total_rows))

    if total_rows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    green = np.array([0.0, 1.0, 0.0, 0.5], dtype=np.float32)
    blue = np.array([30 / 255.0, 144 / 255.0, 255 / 255.0, 0.5], dtype=np.float32)

    for i in range(N):
        fr = i // ncols
        fc = i % ncols
        idx = frame_idx == i

        ax_orig = axes[fr, fc]
        ax_orig.imshow(imgs[i])
        if idx.any():
            _draw_points(ax_orig, coords[idx], labels[idx])
        ax_orig.set_title(f"frame {i} ({stems[i]})")
        ax_orig.axis("off")

        if use_sam2:
            ax_s2 = axes[frame_rows + fr, fc]
            ax_s2.imshow(imgs[i])
            mask_s2 = pred_sam2[i].astype(np.float32)[..., None] * green[None, None, :]
            ax_s2.imshow(mask_s2)
            if idx.any():
                _draw_points(ax_s2, coords[idx], labels[idx])
            ax_s2.set_title(f"SAM2 | frame {i}")
            ax_s2.axis("off")

        sv_row0 = 2 * frame_rows if use_sam2 else frame_rows
        ax_sv = axes[sv_row0 + fr, fc]
        ax_sv.imshow(imgs[i])
        mask_sv = pred_samvggt[i].astype(np.float32)[..., None] * blue[None, None, :]
        ax_sv.imshow(mask_sv)
        if idx.any():
            _draw_points(ax_sv, coords[idx], labels[idx])
        ax_sv.set_title(f"SamVGGT | frame {i}")
        ax_sv.axis("off")

    for i in range(N, frame_rows * ncols):
        fr = i // ncols
        fc = i % ncols
        axes[fr, fc].axis("off")
        if use_sam2:
            axes[frame_rows + fr, fc].axis("off")
        axes[(2 * frame_rows if use_sam2 else frame_rows) + fr, fc].axis("off")

    if use_sam2:
        fig.suptitle("SAM2 (green) vs SamVGGT (blue)", fontsize=13, y=0.995)
    else:
        fig.suptitle("SamVGGT (blue)", fontsize=13, y=0.995)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer SAM2 and SamVGGT on custom images with hardcoded point prompts.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help=(
            "Directory containing scene frames and GT masks. "
            "Expected <stem>.jpg and <stem>_label.npy where <stem> is frame_<digits> or <digits>."
        ),
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default=None,
        help="Benchmark root used with --scene_id to resolve <benchmark_root>/<scene_id>/images.",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene ID under benchmark root. If provided, image_dir resolves to <benchmark_root>/<scene_id>/images.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16", "none"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sam_v_ckpt", type=str, required=True)
    parser.add_argument(
        "--no_sam2",
        action="store_true",
        help="Do not load or run SAM2; only SamVGGT (--sam2_ckpt not required).",
    )
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default=None,
                        help="SAM2 checkpoint path (required unless --no_sam2).")
    parser.add_argument("--sam_model_type", type=str, default="vit_h",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_encode_chunk", type=int, default=8)
    parser.add_argument("--sam2_vos_optimized", action="store_true")

    parser.add_argument("--output_dir", type=str, default="custom_infer_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_sam2 and not args.sam2_ckpt:
        raise SystemExit("error: --sam2_ckpt is required unless --no_sam2 is set.")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if args.scene_id:
        if not args.benchmark_root:
            raise SystemExit("error: --scene_id requires --benchmark_root.")
        image_dir = Path(args.benchmark_root) / args.scene_id / "images"
    elif args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        raise SystemExit("error: provide either --scene_id (with --benchmark_root) or --image_dir.")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # expanded_prompts = _expand_prompts_with_grid(HARDCODED_PROMPTS, GRID_SPACING)
    expanded_prompts = HARDCODED_PROMPTS
    # print(f"Seed prompts: {len(HARDCODED_PROMPTS)}, "
    #       f"expanded to {len(expanded_prompts)} with 5x5 grid (spacing={GRID_SPACING}px)")
    sample = build_custom_sample(image_dir, expanded_prompts)
    N = sample["images"].shape[0]
    prompt_frame = int(sample["point_frame_indices"][0].item())
    print(f"Loaded {N} images from {image_dir}")
    print(f"Discovered {len(sample['gt_mask_paths'])} GT masks in the same directory.")
    print(f"Resolution resized to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Prompt points: {len(expanded_prompts)} on frame {prompt_frame} ({sample['frame_stems'][prompt_frame]})")

    from model.sam_vggt_model import build_sam_vggt

    print("[Init] Building SamVGGT...")
    sam_vggt = build_sam_vggt(
        device=str(device),
        sam_model_type=args.sam_model_type,
        sam_encode_chunk=args.sam_encode_chunk,
    )
    load_samvggt_checkpoint(sam_vggt, args.sam_v_ckpt, device)
    sam_vggt.eval()

    sam2_predictor = None
    if not args.no_sam2:
        from sam2.build_sam import build_sam2_video_predictor

        print("[Init] Building SAM2 video predictor...")
        sam2_predictor = build_sam2_video_predictor(
            args.sam2_cfg,
            args.sam2_ckpt,
            device=str(device),
            vos_optimized=args.sam2_vos_optimized,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_root = out_dir / ".tmp_frames"
    temp_root.mkdir(parents=True, exist_ok=True)

    import time

    print("[Infer] Running SamVGGT...")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    samvggt_out = infer_samvggt(sam_vggt, sample, device=device, amp_dtype=args.amp_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    print(f"SamVGGT inference time: {t1 - t0:.3f}s")
    pred_sam2: np.ndarray | None = None
    if sam2_predictor is not None:
        print("[Infer] Running SAM2...")
        sam2_out = infer_sam2(sam2_predictor, sample, temp_root=temp_root)
        pred_sam2 = sam2_out["pred_binary"]
    else:
        print("[Infer] Skipping SAM2 (--no_sam2).")

    pred_samvggt = samvggt_out["pred_binary"]
    samvggt_logits = samvggt_out.get("logits")

    print(f"SamVGGT predicted foreground pixels per frame: "
          f"{[int(pred_samvggt[i].sum()) for i in range(N)]}")
    if pred_sam2 is not None:
        print(f"SAM2    predicted foreground pixels per frame: "
              f"{[int(pred_sam2[i].sum()) for i in range(N)]}")
    if samvggt_logits is not None:
        print(f"SamVGGT logit min: {float(samvggt_logits.min()):.6f}, "
              f"max: {float(samvggt_logits.max()):.6f}")

    overlay_path = out_dir / "overlay.png"
    save_overlay(sample, pred_sam2, pred_samvggt, overlay_path)
    print(f"Saved overlay: {overlay_path}")

    import shutil
    shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
