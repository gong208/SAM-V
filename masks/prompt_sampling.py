from __future__ import annotations

from typing import Optional

import numpy as np


def _foreground_points_xy(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask.astype(bool, copy=False))
    if xs.size == 0:
        raise ValueError("Cannot sample prompt points from an empty mask.")
    return np.stack([xs, ys], axis=1).astype(np.float32, copy=False)


def _pad_points(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) >= num_points:
        return points[:num_points].astype(np.float32, copy=False)
    if len(points) == 0:
        raise ValueError("Cannot pad an empty point set.")
    padded = [points.astype(np.float32, copy=False)]
    while sum(len(chunk) for chunk in padded) < num_points:
        padded.append(points[-1:].astype(np.float32, copy=False))
    return np.concatenate(padded, axis=0)[:num_points]


def _round_and_clip_point(point_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    x = float(np.clip(np.round(point_xy[0]), 0, w - 1))
    y = float(np.clip(np.round(point_xy[1]), 0, h - 1))
    return np.array([x, y], dtype=np.float32)


def _nearest_foreground_point(point_xy: np.ndarray, fg_xy: np.ndarray) -> np.ndarray:
    d2 = np.sum((fg_xy - point_xy[None, :]) ** 2, axis=1)
    return fg_xy[int(np.argmin(d2))].astype(np.float32, copy=False)


def _random_unique_points(
    fg_xy: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(fg_xy) <= num_points:
        return _pad_points(fg_xy, num_points)
    idx = rng.choice(len(fg_xy), size=num_points, replace=False)
    return fg_xy[idx].astype(np.float32, copy=False)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "prompt_sampling requires opencv-python for pole-based point selection."
        ) from exc
    mask_u8 = mask.astype(np.uint8, copy=False)
    return cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32, copy=False)


def _pole_point(mask: np.ndarray, fg_xy: np.ndarray) -> np.ndarray:
    dt = _distance_transform(mask)
    if np.max(dt) <= 0.0:
        return fg_xy[0].astype(np.float32, copy=False)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return np.array([x, y], dtype=np.float32)


def _greedy_diverse_points(
    mask: np.ndarray,
    fg_xy: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    seed_point: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(fg_xy) <= num_points:
        return _pad_points(fg_xy, num_points)

    dt = _distance_transform(mask)
    dt_max = float(dt.max())
    if dt_max > 0.0:
        keep = dt[fg_xy[:, 1].astype(np.int64), fg_xy[:, 0].astype(np.int64)] >= max(1.0, 0.35 * dt_max)
        candidates = fg_xy[keep]
        if len(candidates) == 0:
            candidates = fg_xy
    else:
        candidates = fg_xy

    if seed_point is None:
        seed_point = candidates[rng.integers(len(candidates))].astype(np.float32, copy=False)

    selected = [seed_point.astype(np.float32, copy=False)]
    while len(selected) < num_points:
        selected_arr = np.stack(selected, axis=0)
        d2 = ((candidates[:, None, :] - selected_arr[None, :, :]) ** 2).sum(axis=2)
        min_d2 = d2.min(axis=1)
        candidate_dt = dt[candidates[:, 1].astype(np.int64), candidates[:, 0].astype(np.int64)]
        score = min_d2 + 0.5 * (candidate_dt ** 2)
        next_idx = int(np.argmax(score))
        next_point = candidates[next_idx].astype(np.float32, copy=False)
        if np.any(np.all(np.isclose(selected_arr, next_point[None, :], atol=0.5), axis=1)):
            # Fallback if the best candidate duplicates an existing point.
            remaining = []
            for candidate in candidates:
                if not np.any(np.all(np.isclose(selected_arr, candidate[None, :], atol=0.5), axis=1)):
                    remaining.append(candidate)
            if not remaining:
                break
            next_point = np.asarray(remaining[0], dtype=np.float32)
        selected.append(next_point)
    return _pad_points(np.stack(selected, axis=0), num_points)


def sample_prompt_points_from_mask(
    mask: np.ndarray,
    method: str = "pole_plus_diverse",
    num_points: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample prompt points from a binary proposal mask.

    Returns an array of shape [num_points, 2] in (x, y) image coordinates.
    """
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    rng = rng or np.random.default_rng(0)
    mask_bool = mask.astype(bool, copy=False)
    fg_xy = _foreground_points_xy(mask_bool)

    centroid_xy = fg_xy.mean(axis=0)
    centroid_point = _round_and_clip_point(centroid_xy, mask_bool.shape)
    snapped_centroid = _nearest_foreground_point(centroid_xy, fg_xy)

    if method == "centroid":
        return _pad_points(np.repeat(centroid_point[None, :], num_points, axis=0), num_points)
    if method == "snapped_centroid":
        return _pad_points(np.repeat(snapped_centroid[None, :], num_points, axis=0), num_points)
    if method == "random_interior":
        return _random_unique_points(fg_xy, num_points, rng)
    if method == "pole_plus_random":
        pole = _pole_point(mask_bool, fg_xy)
        if num_points == 1:
            return pole[None, :]
        remaining = fg_xy
        random_points = _random_unique_points(remaining, max(1, num_points - 1), rng)
        return _pad_points(np.concatenate([pole[None, :], random_points], axis=0), num_points)
    if method == "pole_plus_diverse":
        pole = _pole_point(mask_bool, fg_xy)
        return _greedy_diverse_points(mask_bool, fg_xy, num_points, rng, seed_point=pole)

    raise ValueError(
        f"Unknown prompt sampling method '{method}'. "
        "Use centroid, snapped_centroid, random_interior, "
        "pole_plus_random, or pole_plus_diverse."
    )
