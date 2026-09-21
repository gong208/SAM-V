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
from PIL import Image, ImageDraw

from demos.infer_custom_images import build_sample_from_image_arrays


SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MASK_RGBA = (0, 144, 255, 115)
POS_POINT = (255, 220, 40)
NEG_POINT = (235, 60, 60)


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
