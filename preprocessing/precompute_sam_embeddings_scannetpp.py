#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry


def get_rank_world_local() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def read_scene_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    # Ensure writable contiguous array to avoid torch warning on read-only buffers.
    return np.array(img, copy=True)


def prepare_sam_input_batch(
    image_paths: List[str],
    sam_model,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns tensor [B, 3, 1024, 1024] ready for SAM image encoder.
    """
    prepared = []
    for path in image_paths:
        image_np = load_image_rgb(path)
        x = torch.as_tensor(image_np, device=device, dtype=torch.float32)
        x = x.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1,3,H,W]
        # Match training preprocessing exactly: bilinear resize to 1024x1024.
        x = F.interpolate(
            x,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        x = sam_model.preprocess(x)  # [1,3,1024,1024]
        prepared.append(x)
    return torch.cat(prepared, dim=0)


def save_embedding(path: str, embedding: torch.Tensor, save_dtype: str) -> None:
    if save_dtype == "fp16":
        payload = embedding.detach().cpu().to(torch.float16).contiguous()
    elif save_dtype == "fp32":
        payload = embedding.detach().cpu().to(torch.float32).contiguous()
    else:
        raise ValueError(f"Unsupported save dtype: {save_dtype}")
    torch.save(payload, path)


def process_scannetpp_scene(
    scene_path: str,
    sam_model,
    device: torch.device,
    batch_size: int,
    output_subdir: str,
    overwrite: bool,
    save_dtype: str,
) -> tuple[int, int]:
    """
    Returns (num_saved, num_skipped_existing).
    """
    image_dir = os.path.join(scene_path, "images")
    if not os.path.isdir(image_dir):
        return 0, 0

    frame_files = sorted(
        f for f in os.listdir(image_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    if len(frame_files) == 0:
        return 0, 0

    embed_dir = os.path.join(scene_path, output_subdir)
    os.makedirs(embed_dir, exist_ok=True)

    saved = 0
    skipped = 0

    # Filter files if overwrite=False.
    pending_files = []
    for fname in frame_files:
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(embed_dir, f"{stem}.pt")
        if (not overwrite) and os.path.isfile(out_path):
            skipped += 1
            continue
        pending_files.append(fname)

    for file_chunk in chunked(pending_files, batch_size):
        image_paths = [os.path.join(image_dir, f) for f in file_chunk]
        input_batch = prepare_sam_input_batch(
            image_paths=image_paths,
            sam_model=sam_model,
            device=device,
        )
        with torch.no_grad():
            feats, _ = sam_model.image_encoder(input_batch)  # [B,256,64,64]

        for i, fname in enumerate(file_chunk):
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(embed_dir, f"{stem}.pt")
            save_embedding(out_path, feats[i], save_dtype=save_dtype)
            saved += 1

        del input_batch
        del feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "output_subdir": output_subdir,
        "save_dtype": save_dtype,
        "batch_size": batch_size,
        "num_total_frames": len(frame_files),
        "num_saved_this_run": saved,
        "num_skipped_existing": skipped,
    }
    with open(os.path.join(embed_dir, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return saved, skipped


def process_split(
    split_name: str,
    split_root: str,
    scene_list_path: str,
    sam_model,
    device: torch.device,
    batch_size: int,
    output_subdir: str,
    overwrite: bool,
    save_dtype: str,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[int, int, int]:
    scene_ids = read_scene_list(scene_list_path)
    scene_ids = scene_ids[rank::world_size]
    saved_total = 0
    skipped_total = 0
    seen_scenes = 0

    print(
        f"[ScanNet++:{split_name}][rank {rank}/{world_size}] "
        f"root={split_root} scenes_assigned={len(scene_ids)} output_subdir={output_subdir}"
    )
    for scene_id in tqdm(scene_ids, desc=f"{split_name} scenes r{rank}", position=rank % 8):
        scene_path = os.path.join(split_root, scene_id)
        if not os.path.isdir(scene_path):
            continue
        seen_scenes += 1
        saved, skipped = process_scannetpp_scene(
            scene_path=scene_path,
            sam_model=sam_model,
            device=device,
            batch_size=batch_size,
            output_subdir=output_subdir,
            overwrite=overwrite,
            save_dtype=save_dtype,
        )
        saved_total += saved
        skipped_total += skipped

    print(
        f"[Done {split_name}][rank {rank}] scenes_found={seen_scenes} "
        f"saved={saved_total} skipped_existing={skipped_total}"
    )
    return seen_scenes, saved_total, skipped_total


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute SAM image embeddings for ScanNet++ split scenes. "
            "Saves per-frame embeddings as .pt under each scene."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root containing train/, val/, and splits/.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Path to train scene-list txt. Defaults to <dataset_root>/splits/nvs_sem_train.txt.",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="Path to val scene-list txt. Defaults to <dataset_root>/splits/nvs_sem_val.txt.",
    )
    parser.add_argument(
        "--train_small_split",
        type=str,
        default=None,
        help="Path to small train scene-list txt. Defaults to <dataset_root>/splits/nvs_sem_train_small.txt.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "val"],
        help="Which splits to process. Any of: train val train_small",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help="SAM model type key for sam_model_registry.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parent.parent
                    / "submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth"),
        help="Path to SAM checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SAM image encoder, e.g. cuda or cuda:0 or cpu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images encoded together by SAM image encoder.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="sam_embeddings",
        help="Per-scene output subdirectory to store frame embeddings.",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Saved tensor dtype.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame embeddings if present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    rank, world_size, local_rank = get_rank_world_local()

    split_to_path = {
        "train": args.train_split or os.path.join(args.dataset_root, "splits", "nvs_sem_train.txt"),
        "val": args.val_split or os.path.join(args.dataset_root, "splits", "nvs_sem_val.txt"),
        "train_small": args.train_small_split or os.path.join(args.dataset_root, "splits", "nvs_sem_train_small.txt"),
    }
    split_to_root = {
        "train": os.path.join(args.dataset_root, "train"),
        "val": os.path.join(args.dataset_root, "val"),
        # train_small scene IDs still point to directories under train/
        "train_small": os.path.join(args.dataset_root, "train"),
    }

    requested_splits = []
    for s in args.splits:
        if s not in ("train", "val", "train_small"):
            raise ValueError(f"Unsupported split: {s}. Use train/val/train_small.")
        requested_splits.append(s)

    if args.device.startswith("cuda") and world_size > 1:
        # In torchrun multi-process mode, pin one process to one GPU.
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    print(
        f"[Init][rank {rank}/{world_size}] Loading SAM ({args.sam_model_type}) "
        f"checkpoint={args.sam_checkpoint} on device={device}"
    )
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False

    scenes_total = 0
    saved_total = 0
    skipped_total = 0
    for split in requested_splits:
        split_root = split_to_root[split]
        split_list = split_to_path[split]
        seen_scenes, saved, skipped = process_split(
            split_name=split,
            split_root=split_root,
            scene_list_path=split_list,
            sam_model=sam,
            device=device,
            batch_size=args.batch_size,
            output_subdir=args.output_subdir,
            overwrite=args.overwrite,
            save_dtype=args.save_dtype,
            rank=rank,
            world_size=world_size,
        )
        scenes_total += seen_scenes
        saved_total += saved
        skipped_total += skipped

    print(
        f"[Done][rank {rank}] splits={requested_splits} scenes_found={scenes_total} "
        f"saved={saved_total} skipped_existing={skipped_total} "
        f"output_subdir={args.output_subdir}"
    )


if __name__ == "__main__":
    main()
