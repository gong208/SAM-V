#!/usr/bin/env python3
"""
Interchange format between baseline inference (stage B, baseline env) and
scoring (stage C, project env).

One ``tracks.json`` per scene::

    {"scene_id": "0a76e06478",
     "frame_names": ["frame_000000.jpg", ...],
     "target_size": [1024, 1024],
     "method": "panst3r_v2",
     "config": {...},
     "infer_seconds": {"prepare": 0.0, "infer": 12.3, "assemble": 0.4},
     "tracks": [{"score": 0.91,
                 "per_view_masks": {"0": {"size": [1024, 1024], "counts": [...]}}}]}

The RLE is exactly what ``encode_mask_to_rle`` in the frozen benchmark module
emits, so ``decode_mask`` consumes it unchanged.  Frames on which a track is
absent are omitted, matching SAM-V's own convention.

This module is import-safe in either environment: the RLE helpers are imported
from the frozen benchmark module when its heavy dependencies are available
(project env) and fall back to local, byte-identical copies otherwise, so a
baseline env can write ``tracks.json`` without pulling in sam-hq/vggt.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

INFER_SECONDS_KEYS = ("prepare", "infer", "assemble")


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------
# Mirrors is_rgb_frame_path / list_scene_image_paths in the frozen benchmark
# module (sam_vggt_3dtracking_benchmark.py:94-95,101-112).  It is duplicated
# rather than imported because a baseline env cannot import that module (it
# pulls in sam-hq / vggt / segment_anything).  Any divergence is caught by the
# frame_names assert in score_baseline.py, which compares against the frozen
# module's own listing.
_RGB_FRAME_STEM_RE = re.compile(r"^frame_\d+$")
_RGB_NUMERIC_STEM_RE = re.compile(r"^\d+$")


def is_rgb_frame_path(image_path: Path) -> bool:
    if image_path.suffix.lower() != ".jpg":
        return False
    stem = image_path.stem
    return (
        _RGB_FRAME_STEM_RE.fullmatch(stem) is not None
        or _RGB_NUMERIC_STEM_RE.fullmatch(stem) is not None
    )


def list_scene_frame_paths(scene_dir: Path) -> List[Path]:
    """RGB frames of one benchmark scene, in the order the scorer expects."""
    image_dir = Path(scene_dir) / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No 'images' directory in {scene_dir}")
    paths = [p for p in sorted(image_dir.glob("*.jpg")) if is_rgb_frame_path(p)]
    if not paths:
        raise FileNotFoundError(f"No RGB frames found in {image_dir}")
    return paths


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------
# Preferred: the frozen implementations, so encode/decode cannot drift.
try:  # pragma: no cover - depends on which env we are in
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.sam_vggt_3dtracking_benchmark import (  # type: ignore
        decode_mask,
        encode_mask_to_rle,
    )
    _RLE_SOURCE = "frozen"
except Exception:  # baseline env without sam-hq / vggt / torch-cuda stack
    _RLE_SOURCE = "local"

    def decode_mask(mask_data: Any) -> np.ndarray:
        """Local copy of the frozen ``decode_mask`` (dict/ndarray inputs only)."""
        if isinstance(mask_data, np.ndarray):
            return mask_data.astype(bool, copy=False)
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
        """Local copy of the frozen ``encode_mask_to_rle``."""
        mask = mask.astype(bool, copy=False)
        height, width = mask.shape
        flat = mask.T.flatten()
        diff = np.diff(flat.astype(np.int8))
        change = np.concatenate([[0], np.where(diff != 0)[0] + 1, [height * width]])
        counts = np.diff(change).tolist()
        if flat[0]:
            counts = [0] + counts
        return {"size": [height, width], "counts": counts}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def total_infer_seconds(d: Optional[Mapping[str, Any]]) -> float:
    """Sum the three-key ``infer_seconds`` breakdown, tolerating missing keys."""
    if not d:
        return 0.0
    return float(sum(float(d.get(k, 0.0) or 0.0) for k in INFER_SECONDS_KEYS))


def make_infer_seconds(
    prepare: float = 0.0,
    infer: float = 0.0,
    assemble: float = 0.0,
) -> Dict[str, float]:
    """Every method writes all three keys, so the sum is well-defined."""
    return {"prepare": float(prepare), "infer": float(infer), "assemble": float(assemble)}


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def make_track(
    per_view_masks: Mapping[Any, Any],
    score: float = 0.0,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Build one track entry.  ``per_view_masks`` maps frame index -> boolean mask
    (ndarray) or an already-encoded RLE dict; empty masks are dropped.
    """
    encoded: Dict[str, Any] = {}
    for frame_idx, mask in per_view_masks.items():
        if isinstance(mask, dict):
            encoded[str(int(frame_idx))] = mask
            continue
        arr = np.asarray(mask).astype(bool, copy=False)
        if not arr.any():
            continue
        encoded[str(int(frame_idx))] = encode_mask_to_rle(arr)
    track: Dict[str, Any] = {"score": float(score), "per_view_masks": encoded}
    track.update(extra)
    return track


def write_tracks(
    path: Path,
    scene_id: str,
    frame_names: Sequence[str],
    tracks: Sequence[Mapping[str, Any]],
    method: str,
    config: Optional[Mapping[str, Any]] = None,
    target_size: Sequence[int] = (1024, 1024),
    infer_seconds: Optional[Mapping[str, Any]] = None,
) -> Path:
    payload = {
        "scene_id": str(scene_id),
        "frame_names": [str(n) for n in frame_names],
        "target_size": [int(target_size[0]), int(target_size[1])],
        "method": str(method),
        "config": dict(config or {}),
        "infer_seconds": dict(infer_seconds or make_infer_seconds()),
        "tracks": list(tracks),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f)
    return path


def read_tracks(path: Path) -> Dict[str, Any]:
    with Path(path).open() as f:
        payload = json.load(f)
    for key in ("scene_id", "frame_names", "tracks"):
        if key not in payload:
            raise ValueError(f"{path}: tracks.json missing required key {key!r}")
    return payload


def tracks_to_annotations(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a ``tracks.json`` payload into the annotation dicts that
    ``evaluate_scene`` consumes.  Only ``per_view_masks`` is load-bearing; the
    remaining keys are read defensively by the frozen scorer.
    """
    annotations: List[Dict[str, Any]] = []
    for track in payload["tracks"]:
        per_view = {int(k): v for k, v in track.get("per_view_masks", {}).items()}
        annotations.append({
            "per_view_masks": per_view,
            "predicted_iou": float(track.get("score", 0.0)),
            "stability_score": float(track.get("stability_score", 0.0)),
            "source_view": int(track.get("source_view", -1)),
        })
    return annotations
