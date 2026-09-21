from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from benchmarks.compare_baseline_sam2 import TARGET_SIZE


SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MASK_RGBA = (0, 144, 255, 115)
POS_POINT = (255, 220, 40)
NEG_POINT = (235, 60, 60)


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


@dataclass(frozen=True)
class UploadedImage:
    filename: str
    stem: str
    array: np.ndarray
    width: int
    height: int


def safe_stem(filename: str, fallback: str) -> str:
    stem = Path(filename).stem or fallback
    stem = SAFE_STEM_RE.sub("_", stem).strip("._")
    return stem or fallback


def decode_image_bytes(filename: str, data: bytes) -> UploadedImage:
    if not data:
        raise ValueError(f"{filename or 'upload'} is empty.")
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"{filename or 'upload'} is not a readable image.") from exc

    array = np.array(image, dtype=np.float32)
    stem = safe_stem(filename, "frame")
    return UploadedImage(
        filename=filename or f"{stem}.png",
        stem=stem,
        array=array,
        width=image.width,
        height=image.height,
    )


def build_uploaded_sample(images: list[UploadedImage], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    stems = [image.stem for image in images]
    arrays = [image.array for image in images]
    sample = build_sample_from_image_arrays(stems, arrays, prompts)
    sample["upload_filenames"] = [image.filename for image in images]
    sample["upload_sizes"] = [(image.height, image.width) for image in images]
    return sample


def cleanup_output_dirs(output_root: Path, ttl_seconds: int, now: float | None = None) -> int:
    """Remove old per-request output directories.

    Only immediate child directories are removed; files such as runtime/.gitignore
    are left untouched. A ttl_seconds value <= 0 disables cleanup.
    """
    if ttl_seconds <= 0 or not output_root.exists():
        return 0

    cutoff = (time.time() if now is None else now) - ttl_seconds
    removed = 0
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime >= cutoff:
            continue
        try:
            shutil.rmtree(child)
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _sample_images_uint8(sample: dict[str, Any]) -> np.ndarray:
    images = sample["images"].detach().cpu().clamp(0, 255)
    return images.permute(0, 2, 3, 1).numpy().astype(np.uint8)


def _draw_points(draw: ImageDraw.ImageDraw, points: list[tuple[float, float, int]], radius: int = 8) -> None:
    for x, y, label in points:
        fill = POS_POINT if label == 1 else NEG_POINT
        outline = (20, 20, 20) if label == 1 else (255, 255, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        if label == 0:
            draw.line((x - radius + 3, y, x + radius - 3, y), fill=outline, width=2)
            draw.line((x, y - radius + 3, x, y + radius - 3), fill=outline, width=2)


def render_outputs(
    sample: dict[str, Any],
    pred_binary: np.ndarray,
    output_dir: Path,
    url_prefix: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _sample_images_uint8(sample)
    coords = sample["point_coords"].detach().cpu().numpy()
    labels = sample["point_labels"].detach().cpu().numpy().astype(int)
    frame_indices = sample["point_frame_indices"].detach().cpu().numpy().astype(int)

    frames: list[dict[str, Any]] = []
    for index, image_array in enumerate(images):
        mask = pred_binary[index].astype(bool)
        base = Image.fromarray(image_array, mode="RGB").convert("RGBA")
        mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        mask_pixels = np.zeros((base.height, base.width, 4), dtype=np.uint8)
        mask_pixels[mask] = MASK_RGBA
        mask_layer = Image.fromarray(mask_pixels, mode="RGBA")
        overlay = Image.alpha_composite(base, mask_layer)

        frame_points = [
            (float(coords[i, 0]), float(coords[i, 1]), int(labels[i]))
            for i in range(len(labels))
            if int(frame_indices[i]) == index
        ]
        _draw_points(ImageDraw.Draw(overlay), frame_points)

        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        overlay_name = f"frame_{index:03d}_overlay.png"
        mask_name = f"frame_{index:03d}_mask.png"
        overlay.save(output_dir / overlay_name)
        mask_image.save(output_dir / mask_name)

        filename = sample.get("upload_filenames", sample["frame_stems"])[index]
        orig_h, orig_w = sample.get("upload_sizes", sample.get("original_sizes"))[index]
        frames.append({
            "index": index,
            "filename": filename,
            "width": int(orig_w),
            "height": int(orig_h),
            "overlay_url": f"{url_prefix}/{overlay_name}",
            "mask_url": f"{url_prefix}/{mask_name}",
            "foreground_pixels": int(mask.sum()),
        })
    return frames


def mock_predict(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    """Small deterministic mask generator for web/API testing without model weights."""
    h, w = int(sample["H"]), int(sample["W"])
    n = sample["images"].shape[0]
    pred = np.zeros((n, h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    coords = sample["point_coords"].detach().cpu().numpy()
    labels = sample["point_labels"].detach().cpu().numpy().astype(int)
    frame_indices = sample["point_frame_indices"].detach().cpu().numpy().astype(int)
    radius = max(24, min(h, w) // 12)
    for (x, y), label, frame_index in zip(coords, labels, frame_indices):
        disk = (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= radius ** 2
        if label == 1:
            pred[int(frame_index)] |= disk
        else:
            pred[int(frame_index)] &= ~disk
    scores = pred.astype(np.float32)
    return {"pred_binary": pred, "pred_scores": scores}


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
