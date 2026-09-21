import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from skimage import io
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hypersim helpers
# ---------------------------------------------------------------------------

def process_single_file(args, area_ratio_threshold=0.0025):
    """
    Process a single instance mask file. Module-level for pickling in ProcessPoolExecutor.
    """
    scene_path, fname = args
    inst_dir = os.path.join(scene_path, "instance")
    out_dir = os.path.join(scene_path, "valid_ids")

    mask_path = os.path.join(inst_dir, fname)
    mask = io.imread(mask_path)

    if mask.ndim > 2:
        mask = mask[:, :, 0]

    H, W = mask.shape
    total_pixels = H * W
    min_pixels = int(area_ratio_threshold * total_pixels)

    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64))

    ids = torch.unique(mask_t)
    ids = ids[ids != 0]  # remove background

    valid_ids = []
    id_areas = []  # Track all IDs and their areas

    for idv in ids.tolist():
        area = torch.sum(mask_t == idv).item()
        id_areas.append((idv, area))
        if area >= min_pixels:
            valid_ids.append(idv)

    # If no IDs meet the threshold, use the one with the largest area
    if len(valid_ids) == 0 and len(id_areas) > 0:
        largest_id = max(id_areas, key=lambda x: x[1])[0]
        valid_ids.append(largest_id)

    # save per-image valid instance ids
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, fname.replace(".png", ".json"))
    with open(save_path, "w") as f:
        json.dump({"ids": valid_ids}, f)

    return valid_ids


def _process_hypersim_task(task):
    """Module-level wrapper for ProcessPoolExecutor (Hypersim)."""
    torch.set_num_threads(1)
    scene_path, fname, area_ratio_threshold = task
    return process_single_file(
        (scene_path, fname), area_ratio_threshold=area_ratio_threshold
    )


def compute_valid_ids_per_image(
    scene_path, area_ratio_threshold=0.0025, num_workers=0
):
    """
    Only keep instance IDs whose area >= threshold * image_area.
    If no IDs meet the threshold, include the ID with the largest area.
    """
    inst_dir = os.path.join(scene_path, "instance")
    if not os.path.isdir(inst_dir):
        return
    valid_exts = {".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF"}
    files = sorted(
        fname
        for fname in os.listdir(inst_dir)
        if os.path.splitext(fname)[1] in valid_exts and os.path.isfile(os.path.join(inst_dir, fname))
    )

    tasks = [(scene_path, fname, area_ratio_threshold) for fname in files]
    desc = f"Scene {os.path.basename(scene_path)}"
    scene_union_ids = set()

    if num_workers and num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for frame_valid_ids in tqdm(
                executor.map(_process_hypersim_task, tasks),
                total=len(tasks),
                desc=desc,
            ):
                scene_union_ids.update(frame_valid_ids)
    else:
        for task in tqdm(tasks, total=len(tasks), desc=desc):
            frame_valid_ids = _process_hypersim_task(task)
            scene_union_ids.update(frame_valid_ids)

    # Save union of valid object IDs across all frames in this scene.
    union_save_path = os.path.join(scene_path, "object_union.json")
    union_payload = {"ids": sorted(scene_union_ids)}
    with open(union_save_path, "w") as f:
        json.dump(union_payload, f)


def preprocess_dataset(root_dir, num_workers=0):
    scenes = sorted(os.listdir(root_dir))
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        print(f"Processing scene: {scene_path}")
        if os.path.isdir(scene_path):
            compute_valid_ids_per_image(scene_path, num_workers=num_workers)


# ---------------------------------------------------------------------------
# ScanNet++ helpers
# ---------------------------------------------------------------------------

def _compute_valid_ids_from_mask(mask, area_ratio_threshold=0.0025):
    """Shared logic: given a 2-D int mask, return list of valid instance IDs."""
    H, W = mask.shape
    total_pixels = H * W
    min_pixels = int(area_ratio_threshold * total_pixels)

    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64))
    ids = torch.unique(mask_t)
    ids = ids[ids != 0]

    valid_ids = []
    id_areas = []
    for idv in ids.tolist():
        area = torch.sum(mask_t == idv).item()
        id_areas.append((idv, area))
        if area >= min_pixels:
            valid_ids.append(idv)

    if len(valid_ids) == 0 and len(id_areas) > 0:
        largest_id = max(id_areas, key=lambda x: x[1])[0]
        valid_ids.append(largest_id)

    return valid_ids


def _all_nonzero_ids_from_mask(mask):
    """Return sorted list of all non-zero instance IDs present in *mask*."""
    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64))
    ids = torch.unique(mask_t)
    ids = ids[ids != 0]
    return sorted(int(v) for v in ids.tolist())


def _process_one_scannetpp_frame(task):
    """Process one ScanNet++ frame. Module-level for ProcessPoolExecutor."""
    torch.set_num_threads(1)
    scene_path, fname, area_ratio_threshold = task
    inst_dir = os.path.join(scene_path, "refined_ins_ids")
    valid_ids_dir = os.path.join(scene_path, "valid_ids")
    presence_dir = os.path.join(scene_path, "instance_presence")

    mask_path = os.path.join(inst_dir, fname)
    mask = io.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[:, :, 0]

    frame_stem = os.path.splitext(fname)[0]

    frame_valid = _compute_valid_ids_from_mask(mask, area_ratio_threshold)
    with open(os.path.join(valid_ids_dir, f"{frame_stem}.json"), "w") as f:
        json.dump({"ids": frame_valid}, f)

    all_ids = _all_nonzero_ids_from_mask(mask)
    with open(os.path.join(presence_dir, f"{frame_stem}.json"), "w") as f:
        json.dump({"ids": all_ids}, f)

    return frame_valid


def process_scannetpp_scene(
    scene_path, area_ratio_threshold=0.0025, num_workers=0
):
    """
    For one ScanNet++ scene, generate:
      - valid_ids/frame_XXXXXX.json
      - instance_presence/frame_XXXXXX.json
      - object_union.json
      - pose/frame_XXXXXX.txt  (extracted from scene_iphone_metadata.npz)
    """
    inst_dir = os.path.join(scene_path, "refined_ins_ids")
    if not os.path.isdir(inst_dir):
        print(f"  [SKIP] No refined_ins_ids/ in {scene_path}")
        return

    # Collect frame_*.png masks (skip .npy files and DSC files)
    mask_files = sorted(
        f for f in os.listdir(inst_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    if not mask_files:
        print(f"  [SKIP] No frame_*.png masks in {inst_dir}")
        return

    valid_ids_dir = os.path.join(scene_path, "valid_ids")
    presence_dir = os.path.join(scene_path, "instance_presence")
    os.makedirs(valid_ids_dir, exist_ok=True)
    os.makedirs(presence_dir, exist_ok=True)

    tasks = [
        (scene_path, fname, area_ratio_threshold) for fname in mask_files
    ]
    desc = f"Scene {os.path.basename(scene_path)}"
    scene_union_ids = set()

    if num_workers and num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for frame_valid in tqdm(
                executor.map(_process_one_scannetpp_frame, tasks),
                total=len(tasks),
                desc=desc,
            ):
                scene_union_ids.update(frame_valid)
    else:
        for task in tqdm(tasks, total=len(tasks), desc=desc):
            frame_valid = _process_one_scannetpp_frame(task)
            scene_union_ids.update(frame_valid)

    # object_union.json
    with open(os.path.join(scene_path, "object_union.json"), "w") as f:
        json.dump({"ids": sorted(scene_union_ids)}, f)

    # Extract per-frame pose files from scene_iphone_metadata.npz
    _extract_poses_scannetpp(scene_path)


def _extract_poses_scannetpp(scene_path):
    """
    Read scene_iphone_metadata.npz and write individual pose .txt files
    into pose/ directory (same 4x4 format as Hypersim).
    """
    npz_path = os.path.join(scene_path, "scene_iphone_metadata.npz")
    if not os.path.isfile(npz_path):
        print(f"  [WARN] No scene_iphone_metadata.npz in {scene_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]          # (N,) string array, e.g. "frame_000010.jpg"
    trajectories = data["trajectories"]  # (N, 4, 4)

    pose_dir = os.path.join(scene_path, "pose")
    os.makedirs(pose_dir, exist_ok=True)

    for i, img_name in enumerate(images):
        img_name = str(img_name)
        if not img_name.startswith("frame_"):
            continue
        frame_stem = os.path.splitext(img_name)[0]  # frame_000010
        pose_path = os.path.join(pose_dir, f"{frame_stem}.txt")
        np.savetxt(pose_path, trajectories[i], fmt="%.6f")


def read_scene_list(path):
    """Read a text file with one scene ID per line, return list of IDs."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_scannetpp_dataset(
    root_dir, scene_list_path, area_ratio_threshold=0.0025, num_workers=0
):
    """
    Preprocess all ScanNet++ scenes listed in *scene_list_path*.
    Generates valid_ids/, instance_presence/, object_union.json, pose/ per scene.
    """
    scene_ids = read_scene_list(scene_list_path)
    print(
        f"[ScanNet++] Processing {len(scene_ids)} scenes from {scene_list_path} "
        f"(num_workers={num_workers})"
    )
    for scene_id in scene_ids:
        scene_path = os.path.join(root_dir, scene_id)
        if not os.path.isdir(scene_path):
            print(f"  [SKIP] Scene dir not found: {scene_path}")
            continue
        process_scannetpp_scene(
            scene_path,
            area_ratio_threshold=area_ratio_threshold,
            num_workers=num_workers,
        )
    print(f"[ScanNet++] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute valid instance IDs for Hypersim and/or ScanNet++."
    )
    parser.add_argument(
        "--hypersim_roots",
        type=str,
        nargs="*",
        default=None,
        help="Hypersim split roots, e.g. $HYPERSIM_ROOT/train $HYPERSIM_ROOT/val",
    )
    parser.add_argument(
        "--scannetpp_root",
        type=str,
        default=None,
        help=(
            "ScanNet++ split root: directory that directly contains scene folders "
            "(e.g. .../processed_scannetpp_v2/train for nvs_sem_train.txt)."
        ),
    )
    parser.add_argument(
        "--scannetpp_scene_lists",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to scene-list .txt files for ScanNet++",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Process workers per scene for frame-level parallelism. "
            "0 runs sequentially. Recommended: 4-8 for ScanNet++."
        ),
    )
    args = parser.parse_args()

    if args.num_workers > 0:
        torch.set_num_threads(1)

    if args.hypersim_roots:
        for root in args.hypersim_roots:
            print(f"[Hypersim] Processing {root}")
            preprocess_dataset(root, num_workers=args.num_workers)

    if args.scannetpp_root and args.scannetpp_scene_lists:
        for scene_list in args.scannetpp_scene_lists:
            preprocess_scannetpp_dataset(
                args.scannetpp_root,
                scene_list,
                num_workers=args.num_workers,
            )
