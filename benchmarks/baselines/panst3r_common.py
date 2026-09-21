#!/usr/bin/env python3
"""
Shared PanSt3R-env helpers: frame loading and the inverse of MUSt3R's
crop/resize transform.

Kept in one place because both of these are places where a silent geometric
error would survive every other check in the pipeline.  The scorer's
``frame_names`` assert catches a wrong *frame list*; nothing downstream catches
a wrongly *inverted* crop, which would just quietly halve every IoU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_scene_frames(
    scene_dir: Path,
    image_size: int,
    patch_size: int,
    verbose: bool = False,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Load a benchmark scene's RGB frames exactly the way the upstream demo does
    (``tools/demo_panst3r.py:load_images``), additionally keeping the native
    size and MUSt3R's ``to_orig`` transform per frame so masks can be mapped
    back.
    """
    import PIL.Image
    from must3r.datasets import ImgNorm
    from must3r.tools.image import get_resize_function

    from benchmarks.baselines.track_io import list_scene_frame_paths

    paths = list_scene_frame_paths(scene_dir)
    views: List[Dict[str, Any]] = []
    for path in paths:
        rgb_image = PIL.Image.open(path).convert("RGB")
        rgb_image.load()
        W, H = rgb_image.size
        resize_func, _to_rescaled, to_orig = get_resize_function(image_size, patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))
        views.append({
            "img": rgb_tensor,
            "true_shape": np.int32([rgb_tensor.shape[-2], rgb_tensor.shape[-1]]),
            "path": str(path),
            "name": path.name,
            "native_hw": (int(H), int(W)),
            "to_orig": np.asarray(to_orig, dtype=np.float64),
        })
        if verbose:
            print(f"  {path.name}: {W}x{H} -> {rgb_tensor.shape[-1]}x{rgb_tensor.shape[-2]}")
    return paths, views


def crop_box_from_to_orig(
    to_orig: np.ndarray,
    proc_hw: Sequence[int],
    native_hw: Sequence[int],
) -> Dict[str, int]:
    """
    Recover the CenterCrop box that MUSt3R applied before resizing.

    ``get_resize_function`` (must3r/tools/image.py:53-93) returns
    ``to_orig = to_orig_crop @ to_orig_resize``, i.e.::

        [[crop_W/target_W, 0,              (W-crop_W)/2],
         [0,               crop_H/target_H, (H-crop_H)/2],
         [0,               0,               1          ]]

    so the crop size and offset read straight off the matrix.  The identity
    case (native resolution already in the table) falls out correctly too.
    """
    proc_h, proc_w = int(proc_hw[0]), int(proc_hw[1])
    native_h, native_w = int(native_hw[0]), int(native_hw[1])

    crop_w = int(round(float(to_orig[0, 0]) * proc_w))
    crop_h = int(round(float(to_orig[1, 1]) * proc_h))
    left = int(round(float(to_orig[0, 2])))
    top = int(round(float(to_orig[1, 2])))

    if np.allclose(to_orig, np.eye(3)):
        # get_resize_function short-circuits to the identity when the native
        # resolution is already one of its supported ones: no crop, no resize.
        crop_h, crop_w, top, left = native_h, native_w, 0, 0

    if not (0 < crop_h <= native_h and 0 < crop_w <= native_w):
        raise ValueError(
            f"recovered crop {crop_h}x{crop_w} inconsistent with native {native_h}x{native_w} "
            f"(to_orig=\n{to_orig})"
        )
    if top < 0 or left < 0 or top + crop_h > native_h or left + crop_w > native_w:
        raise ValueError(
            f"recovered crop box top={top} left={left} {crop_h}x{crop_w} does not fit "
            f"native {native_h}x{native_w}"
        )
    return {
        "crop_h": crop_h, "crop_w": crop_w, "top": top, "left": left,
        "is_trivial": bool(crop_h == native_h and crop_w == native_w and top == 0 and left == 0),
    }


def masks_to_target(
    masks_proc: np.ndarray,
    to_orig: np.ndarray,
    native_hw: Sequence[int],
    target_size: Sequence[int] = (1024, 1024),
) -> np.ndarray:
    """
    Map per-view masks from PanSt3R's processed resolution to the benchmark's
    1024x1024 evaluation grid.

    ``masks_proc`` is ``[K, H_proc, W_proc]`` boolean/float.  The path is
    processed -> (undo resize) -> crop box -> (undo crop) -> native -> target,
    with **bilinear + >0.5**, matching the archived SAM2 baseline
    (``benchmarks/archive/evaluate_scannetpp_sam2.py:120``) and SAM-V's own
    ``_build_output`` upsampling.

    When the crop is trivial -- which is the case for *both* benchmark splits,
    since 920x690 and 640x480 are both 4:3 and MUSt3R's 512 table maps 4:3 to
    (384, 512) exactly -- the two resizes compose into one, and a single
    interpolation is used instead of two so no extra resampling blur is
    introduced.
    """
    masks_proc = np.asarray(masks_proc)
    if masks_proc.ndim != 3:
        raise ValueError(f"expected [K,H,W], got {masks_proc.shape}")
    if masks_proc.shape[0] == 0:
        return np.zeros((0, int(target_size[0]), int(target_size[1])), dtype=bool)

    proc_h, proc_w = masks_proc.shape[-2:]
    native_h, native_w = int(native_hw[0]), int(native_hw[1])
    target = (int(target_size[0]), int(target_size[1]))
    box = crop_box_from_to_orig(to_orig, (proc_h, proc_w), (native_h, native_w))

    t = torch.from_numpy(masks_proc.astype(np.float32, copy=False)).unsqueeze(0)

    if box["is_trivial"]:
        out = F.interpolate(t, size=target, mode="bilinear", align_corners=False)
        return (out[0] > 0.5).numpy()

    # Undo the resize onto the crop box, then paste back into the native frame.
    # Pixels outside the crop were never seen by the model, so they stay empty.
    up = F.interpolate(t, size=(box["crop_h"], box["crop_w"]), mode="bilinear", align_corners=False)
    canvas = torch.zeros((1, masks_proc.shape[0], native_h, native_w), dtype=torch.float32)
    canvas[:, :, box["top"]:box["top"] + box["crop_h"], box["left"]:box["left"] + box["crop_w"]] = up
    out = F.interpolate(canvas, size=target, mode="bilinear", align_corners=False)
    return (out[0] > 0.5).numpy()
