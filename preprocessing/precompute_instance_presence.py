#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage import io
from tqdm import tqdm


VALID_EXTS = {".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF"}


def _process_one_frame(task):
    scene_path, filename, output_dir = task
    instance_dir = os.path.join(scene_path, "instance")
    mask_path = os.path.join(instance_dir, filename)

    mask = io.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[:, :, 0]

    ids = np.unique(mask).astype(np.int64)
    ids = ids[ids != 0]  # remove background
    ids_list = [int(v) for v in ids.tolist()]

    frame_stem = os.path.splitext(filename)[0]
    save_path = os.path.join(output_dir, f"{frame_stem}.json")
    with open(save_path, "w") as f:
        json.dump({"ids": ids_list}, f)

    return frame_stem, len(ids_list)


def compute_instance_presence_for_scene(scene_path, output_subdir="instance_presence", num_workers=0):
    instance_dir = os.path.join(scene_path, "instance")
    if not os.path.isdir(instance_dir):
        return 0

    files = sorted(
        f
        for f in os.listdir(instance_dir)
        if os.path.splitext(f)[1] in VALID_EXTS and os.path.isfile(os.path.join(instance_dir, f))
    )
    if len(files) == 0:
        return 0

    output_dir = os.path.join(scene_path, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(scene_path, fname, output_dir) for fname in files]
    desc = f"{os.path.basename(scene_path)}"

    if num_workers and num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(_process_one_frame, tasks), total=len(tasks), desc=desc))
    else:
        for task in tqdm(tasks, total=len(tasks), desc=desc):
            _process_one_frame(task)

    return len(files)


def preprocess_dataset(root_dir, output_subdir="instance_presence", num_workers=0):
    scene_names = sorted(os.listdir(root_dir))
    total_frames = 0
    total_scenes = 0
    for scene_name in scene_names:
        scene_path = os.path.join(root_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue
        total_scenes += 1
        total_frames += compute_instance_presence_for_scene(
            scene_path, output_subdir=output_subdir, num_workers=num_workers
        )
    print(
        f"[Done] root={root_dir} scenes={total_scenes} frames={total_frames} "
        f"output_subdir={output_subdir}"
    )


# ---------------------------------------------------------------------------
# ScanNet++ helpers
# ---------------------------------------------------------------------------

def _process_one_frame_scannetpp(task):
    """Process one ScanNet++ frame from refined_ins_ids/*.png."""
    scene_path, filename, output_dir = task
    instance_dir = os.path.join(scene_path, "refined_ins_ids")
    mask_path = os.path.join(instance_dir, filename)

    mask = io.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[:, :, 0]

    ids = np.unique(mask).astype(np.int64)
    ids = ids[ids != 0]
    ids_list = [int(v) for v in ids.tolist()]

    frame_stem = os.path.splitext(filename)[0]
    save_path = os.path.join(output_dir, f"{frame_stem}.json")
    with open(save_path, "w") as f:
        json.dump({"ids": ids_list}, f)

    return frame_stem, len(ids_list)


def compute_instance_presence_for_scene_scannetpp(
    scene_path, output_subdir="instance_presence", num_workers=0,
):
    """Compute instance_presence for a single ScanNet++ scene."""
    inst_dir = os.path.join(scene_path, "refined_ins_ids")
    if not os.path.isdir(inst_dir):
        return 0

    files = sorted(
        f for f in os.listdir(inst_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    if not files:
        return 0

    output_dir = os.path.join(scene_path, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(scene_path, fname, output_dir) for fname in files]
    desc = f"{os.path.basename(scene_path)}"

    if num_workers and num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(_process_one_frame_scannetpp, tasks),
                      total=len(tasks), desc=desc))
    else:
        for task in tqdm(tasks, total=len(tasks), desc=desc):
            _process_one_frame_scannetpp(task)

    return len(files)


def preprocess_scannetpp_dataset(
    root_dir, scene_list_path, output_subdir="instance_presence", num_workers=0,
):
    """Run instance-presence precomputation for ScanNet++ scenes in a split list."""
    from valid_instance_id import read_scene_list

    scene_ids = read_scene_list(scene_list_path)
    total_frames = 0
    total_scenes = 0
    for scene_id in scene_ids:
        scene_path = os.path.join(root_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        total_scenes += 1
        total_frames += compute_instance_presence_for_scene_scannetpp(
            scene_path, output_subdir=output_subdir, num_workers=num_workers
        )
    print(
        f"[Done ScanNet++] root={root_dir} scenes={total_scenes} "
        f"frames={total_frames} output_subdir={output_subdir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute per-frame instance presence (unique non-zero instance IDs)."
    )
    parser.add_argument(
        "--roots",
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
        "--output_subdir",
        type=str,
        default="instance_presence",
        help="Per-scene output subdirectory name.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Process workers per scene. 0 runs sequentially.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.roots:
        for root in args.roots:
            preprocess_dataset(
                root_dir=root,
                output_subdir=args.output_subdir,
                num_workers=args.num_workers,
            )
    if args.scannetpp_root and args.scannetpp_scene_lists:
        for scene_list in args.scannetpp_scene_lists:
            preprocess_scannetpp_dataset(
                root_dir=args.scannetpp_root,
                scene_list_path=scene_list,
                output_subdir=args.output_subdir,
                num_workers=args.num_workers,
            )
