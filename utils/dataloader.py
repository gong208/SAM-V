# Copyright by HQ-SAM team
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
#
# Derived from HQ-SAM train/utils/dataloader.py (https://github.com/SysCV/sam-hq).
# Modified for SAM-V: multi-scene ScanNet++ and Hypersim datasets, the
# object-union frame samplers, and the rank-0 index broadcast. See NOTICE.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from functools import partial
from skimage import io
import os
from glob import glob
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def _rank0_broadcast(build_fn):
    """Build an index on rank 0 only, then broadcast it to every rank.

    The dataset/sampler construction walks the dataset root (often NFS) and opens one
    small JSON/txt file per frame. Across the full ScanNet++ set that is ~724k
    per-file round-trips, and previously every rank repeated the whole walk,
    which dominated launch time (~30-45 min). Here only rank 0 touches NFS;
    the resulting (picklable) Python object is broadcast to the other ranks
    over the existing process group, so they skip the filesystem entirely.

    Falls back to a plain local call when not running distributed.
    """
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return build_fn()

    if dist.get_rank() == 0:
        obj_list = [build_fn()]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size  # [H, W]

    def __call__(self, sample):
        image = sample["image"]             # float32 OK
        label = sample["label"]             # int32 or int64 FAILS in interpolate

        # --- Resize image (float32, bilinear) ---
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # --- Resize mask (convert to float → resize → cast back to int) ---
        label_float = label.float().unsqueeze(0)   # [1, 1, H, W]

        label_resized = F.interpolate(
            label_float,
            size=self.size,
            mode="nearest"
        ).squeeze(0)

        # convert back to integer instance IDs
        label = label_resized.to(torch.int64)

        sample["image"] = image
        sample["label"] = label
        sample["shape"] = torch.tensor(self.size)

        return sample



class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='nearest'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}



import os
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
def _apply_index_payload(obj, payload):
    """Populate ``items``/``scenes``/``scene_infos`` from a broadcast payload.

    ``valid_ids`` are carried as plain Python lists in the payload (cheaper to
    pickle/broadcast than tensors) and converted back to tensors here, on every
    rank, matching what ``__getitem__`` expects.
    """
    items, scenes, scene_infos = payload
    for entry in items:
        entry["valid_ids"] = torch.tensor(entry["valid_ids"], dtype=torch.int64)
    obj.items = items
    obj.scenes = scenes
    obj.scene_infos = scene_infos


class MultiSceneImageDataset(Dataset):
    """
    Dataset that returns ONE image + mask + per-image valid instance IDs.
    """
    def __init__(self, root_dir, transform=None, sam_embedding_subdir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sam_embedding_subdir = sam_embedding_subdir

        _apply_index_payload(self, _rank0_broadcast(self._scan_index))
        print(f"[Dataset] Loaded {len(self.items)} images across {len(self.scenes)} scenes.")

    def _scan_index(self):
        items = []        # list of dicts
        scenes = {}       # scene_id -> list of indices in items
        scene_infos = {}  # scene_id -> scene metadata for object-wise sampling

        root_dir = self.root_dir
        sam_embedding_subdir = self.sam_embedding_subdir
        scene_dirs = sorted(os.listdir(root_dir))
        scene_idx = 0

        for scene_name in scene_dirs:
            scene_path = os.path.join(root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            color_dir = os.path.join(scene_path, "color")
            inst_dir  = os.path.join(scene_path, "instance")
            valid_dir = os.path.join(scene_path, "valid_ids")   # NEW

            color_files = sorted(os.listdir(color_dir))
            inst_files  = sorted(os.listdir(inst_dir))
            valid_files = sorted(os.listdir(valid_dir))

            assert len(color_files) == len(inst_files) == len(valid_files), \
                f"Scene {scene_name} mismatch between color, instance, valid_ids."

            scenes[scene_idx] = []
            valid_ids_per_frame = {}
            frame_stems = []

            for i, (im_name, gt_name, valid_name) in enumerate(zip(color_files, inst_files, valid_files)):
                im_path    = os.path.join(color_dir, im_name)
                gt_path    = os.path.join(inst_dir, gt_name)
                valid_path = os.path.join(valid_dir, valid_name)
                frame_stem = os.path.splitext(im_name)[0]

                # load precomputed valid instance IDs
                with open(valid_path, "r") as f:
                    valid_ids = [int(v) for v in json.load(f)["ids"]]

                items.append({
                    "image_path": im_path,
                    "label_path": gt_path,
                    "valid_ids": valid_ids,   # list; -> tensor in _apply_index_payload
                    "scene_id": scene_idx,
                    "frame_stem": frame_stem,
                    "sam_embedding_path": (
                        os.path.join(scene_path, sam_embedding_subdir, f"{frame_stem}.pt")
                        if sam_embedding_subdir is not None
                        else None
                    ),
                })

                scenes[scene_idx].append(len(items) - 1)
                valid_ids_per_frame[frame_stem] = valid_ids
                frame_stems.append(frame_stem)

            object_union_path = os.path.join(scene_path, "object_union.json")
            object_union_ids = []
            if os.path.isfile(object_union_path):
                try:
                    with open(object_union_path, "r") as f:
                        object_union_ids = [int(v) for v in json.load(f).get("ids", [])]
                except Exception:
                    object_union_ids = []
            if len(object_union_ids) == 0:
                union_set = set()
                for ids_this_frame in valid_ids_per_frame.values():
                    union_set.update(ids_this_frame)
                object_union_ids = sorted(union_set)

            scene_infos[scene_idx] = {
                "scene_name": scene_name,
                "scene_path": scene_path,
                "pose_dir": os.path.join(scene_path, "pose"),
                "instance_presence_dir": os.path.join(scene_path, "instance_presence"),
                "frame_stems": frame_stems,
                "valid_ids_per_frame": valid_ids_per_frame,
                "object_union_ids": object_union_ids,
            }

            scene_idx += 1

        return items, scenes, scene_infos

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sampled_object_id = -1
        if isinstance(index, tuple):
            index, sampled_object_id = index
        entry = self.items[index]

        # load image
        im = io.imread(entry["image_path"])
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, axis=2)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)

        # load instance mask
        mask = io.imread(entry["label_path"])
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.int32).unsqueeze(0)

        sample = {
            "image": im,
            "image_path": entry["image_path"],
            "label_path": entry["label_path"],
            "label": mask,
            "scene_id": entry["scene_id"],
            "index": index,
            "valid_ids": entry["valid_ids"],  # NEW
            "frame_stem": entry["frame_stem"],
            "sampled_object_id": int(sampled_object_id),
        }
        sam_embedding_path = entry.get("sam_embedding_path", None)
        if sam_embedding_path is not None and os.path.isfile(sam_embedding_path):
            payload = torch.load(sam_embedding_path, map_location="cpu")
            if torch.is_tensor(payload):
                sample["sam_embedding"] = payload

        if self.transform:
            sample = self.transform(sample)

        return sample


class ScanNetPPMultiSceneImageDataset(Dataset):
    """
    Dataset for ScanNet++ scenes.  Same interface (.items, .scenes, .scene_infos)
    as MultiSceneImageDataset so that existing samplers work unchanged.

    Differences from Hypersim:
      - images live in  images/          (only frame_*.jpg)
      - masks  live in  refined_ins_ids/ (frame_*.png, 16-bit)
      - valid_ids/, instance_presence/, pose/, object_union.json are generated
        by the offline preprocessing scripts.

    Layout: ``root_dir`` should be the split directory that directly contains scene
    folders (e.g. ``.../processed_scannetpp_v2/train`` or ``.../val``), not the
    dataset root above ``train``/``val``.
    """

    def __init__(self, root_dir, scene_list=None, transform=None, sam_embedding_subdir=None):
        """
        Args:
            root_dir:   directory containing one subdirectory per scene ID
                        (typically the ``train`` or ``val`` folder of the processed ScanNet++ root).
            scene_list: list of scene IDs (strings) to include. If ``None``, every
                        subdirectory under ``root_dir`` is used (the train/val folder
                        is already the split, so no separate split file is needed).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sam_embedding_subdir = sam_embedding_subdir
        self.scene_list = scene_list

        _apply_index_payload(self, _rank0_broadcast(self._scan_index))
        print(
            f"[ScanNetPP Dataset] Loaded {len(self.items)} images "
            f"across {len(self.scenes)} scenes."
        )

    def _scan_index(self):
        root_dir = self.root_dir
        sam_embedding_subdir = self.sam_embedding_subdir
        scene_list = self.scene_list
        if scene_list is None:
            scene_list = sorted(os.listdir(root_dir))

        items = []
        scenes = {}
        scene_infos = {}

        scene_idx = 0
        for scene_name in scene_list:
            scene_path = os.path.join(root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            color_dir = os.path.join(scene_path, "images")
            inst_dir  = os.path.join(scene_path, "refined_ins_ids")
            valid_dir = os.path.join(scene_path, "valid_ids")

            if not os.path.isdir(color_dir) or not os.path.isdir(inst_dir):
                continue
            if not os.path.isdir(valid_dir):
                continue

            # Only keep frame_*.jpg images (not DSC*)
            color_files = sorted(
                f for f in os.listdir(color_dir)
                if f.startswith("frame_") and f.endswith(".jpg")
            )
            # Build a set of available mask stems for fast lookup
            mask_stems = {
                os.path.splitext(f)[0]
                for f in os.listdir(inst_dir)
                if f.startswith("frame_") and f.endswith(".png")
            }
            valid_stems = {
                os.path.splitext(f)[0]
                for f in os.listdir(valid_dir)
                if f.endswith(".json")
            }

            scenes[scene_idx] = []
            valid_ids_per_frame = {}
            frame_stems = []

            for im_name in color_files:
                frame_stem = os.path.splitext(im_name)[0]  # frame_000010
                if frame_stem not in mask_stems:
                    continue
                if frame_stem not in valid_stems:
                    continue

                im_path    = os.path.join(color_dir, im_name)
                gt_path    = os.path.join(inst_dir, f"{frame_stem}.png")
                valid_path = os.path.join(valid_dir, f"{frame_stem}.json")

                with open(valid_path, "r") as f:
                    valid_ids = [int(v) for v in json.load(f)["ids"]]

                items.append({
                    "image_path": im_path,
                    "label_path": gt_path,
                    "valid_ids": valid_ids,   # list; -> tensor in _apply_index_payload
                    "scene_id": scene_idx,
                    "frame_stem": frame_stem,
                    "sam_embedding_path": (
                        os.path.join(scene_path, sam_embedding_subdir, f"{frame_stem}.pt")
                        if sam_embedding_subdir is not None
                        else None
                    ),
                })

                scenes[scene_idx].append(len(items) - 1)
                valid_ids_per_frame[frame_stem] = valid_ids
                frame_stems.append(frame_stem)

            if not frame_stems:
                del scenes[scene_idx]
                continue

            object_union_path = os.path.join(scene_path, "object_union.json")
            object_union_ids = []
            if os.path.isfile(object_union_path):
                try:
                    with open(object_union_path, "r") as f:
                        object_union_ids = [int(v) for v in json.load(f).get("ids", [])]
                except Exception:
                    object_union_ids = []
            if len(object_union_ids) == 0:
                union_set = set()
                for ids_this_frame in valid_ids_per_frame.values():
                    union_set.update(ids_this_frame)
                object_union_ids = sorted(union_set)

            scene_infos[scene_idx] = {
                "scene_name": scene_name,
                "scene_path": scene_path,
                "pose_dir": os.path.join(scene_path, "pose"),
                "instance_presence_dir": os.path.join(scene_path, "instance_presence"),
                "frame_stems": frame_stems,
                "valid_ids_per_frame": valid_ids_per_frame,
                "object_union_ids": object_union_ids,
            }

            scene_idx += 1

        return items, scenes, scene_infos

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sampled_object_id = -1
        if isinstance(index, tuple):
            index, sampled_object_id = index
        entry = self.items[index]

        im = io.imread(entry["image_path"])
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, axis=2)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)

        mask = io.imread(entry["label_path"])
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.int32).unsqueeze(0)

        sample = {
            "image": im,
            "image_path": entry["image_path"],
            "label_path": entry["label_path"],
            "label": mask,
            "scene_id": entry["scene_id"],
            "index": index,
            "valid_ids": entry["valid_ids"],
            "frame_stem": entry["frame_stem"],
            "sampled_object_id": int(sampled_object_id),
        }
        sam_embedding_path = entry.get("sam_embedding_path", None)
        if sam_embedding_path is not None and os.path.isfile(sam_embedding_path):
            payload = torch.load(sam_embedding_path, map_location="cpu")
            if torch.is_tensor(payload):
                sample["sam_embedding"] = payload

        if self.transform:
            sample = self.transform(sample)

        return sample


class MergedMultiSceneDataset(Dataset):
    """
    Thin wrapper that merges multiple datasets that share the
    MultiSceneImageDataset interface (.items, .scenes, .scene_infos).

    Scene IDs and item indices are remapped so the combined dataset has
    unique, contiguous scene IDs starting from 0.
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.items = []
        self.scenes = {}
        self.scene_infos = {}

        # Boundaries: for each sub-dataset, record the item-index offset so
        # __getitem__ can delegate to the right sub-dataset.
        self._ds_offsets = []  # (item_offset, scene_offset, dataset)

        item_offset = 0
        scene_offset = 0

        for ds in self.datasets:
            self._ds_offsets.append((item_offset, scene_offset, ds))

            old_to_new_scene = {}
            for old_sid in sorted(ds.scenes.keys()):
                new_sid = scene_offset + old_sid
                old_to_new_scene[old_sid] = new_sid

                self.scenes[new_sid] = [
                    idx + item_offset for idx in ds.scenes[old_sid]
                ]

                info = ds.scene_infos[old_sid].copy()
                self.scene_infos[new_sid] = info

            for item in ds.items:
                new_item = item.copy()
                new_item["scene_id"] = old_to_new_scene[item["scene_id"]]
                self.items.append(new_item)

            item_offset += len(ds.items)
            scene_offset += len(ds.scenes)

        print(
            f"[MergedDataset] {len(self.items)} total images, "
            f"{len(self.scenes)} total scenes from {len(self.datasets)} datasets."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sampled_object_id = -1
        if isinstance(index, tuple):
            index, sampled_object_id = index

        # Find which sub-dataset owns this index
        for item_off, _scene_off, ds in reversed(self._ds_offsets):
            if index >= item_off:
                local_idx = index - item_off
                if sampled_object_id >= 0:
                    sample = ds.__getitem__((local_idx, sampled_object_id))
                else:
                    sample = ds.__getitem__(local_idx)
                # Remap scene_id to the merged namespace
                sample["scene_id"] = self.items[index]["scene_id"]
                sample["index"] = index
                return sample

        raise IndexError(f"Index {index} out of range for MergedMultiSceneDataset")


def read_scene_list(path):
    """Read a text file with one scene ID per line, return list of IDs."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


from torch.utils.data import Sampler

class SceneBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_frames=8, distributed=False,
                 rank=0, world_size=1, shuffle=True, frame_sampling="consecutive"):
        """
        Args:
            frame_sampling: "consecutive" = 4 consecutive frames (non-overlapping windows),
                           "random" = 4 randomly selected frames from the same scene.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.frame_sampling = frame_sampling
        self.epoch = 0

        # Precompute scene groups: for each scene, list of valid frame index lists
        # Each element is (scene_indices, list of groups) where each group is indices into that scene
        self.scene_group_templates = []
        for scene_id, indices in dataset.scenes.items():
            if len(indices) < num_frames:
                continue
            idxs = list(indices)
            groups_this_scene = []

            if frame_sampling == "consecutive":
                for i in range(0, len(idxs) - num_frames + 1, num_frames):
                    g = idxs[i:i + num_frames]
                    if self._group_has_valid_ids(g):
                        groups_this_scene.append(g)
            else:  # "random" - precompute same number of groups as consecutive, placeholders
                n_consecutive = len(list(range(0, len(idxs) - num_frames + 1, num_frames)))
                for _ in range(n_consecutive):
                    # Placeholder; actual random selection happens in __iter__
                    groups_this_scene.append(idxs)  # store full idxs for random sampling

            if groups_this_scene:
                self.scene_group_templates.append((scene_id, idxs, groups_this_scene))

        self._build_groups()

    def _group_has_valid_ids(self, g):
        ids = [self.dataset.items[j]["valid_ids"] for j in g]
        union = torch.unique(torch.cat(ids, dim=0)) if any(t.numel() > 0 for t in ids) else torch.empty(0, dtype=torch.int64)
        return union.numel() > 0

    def _build_groups(self):
        """Build self.groups from templates. For random mode, sample fresh each time."""
        self.groups = []
        rng = np.random.default_rng(self.epoch)
        for scene_id, idxs, templates in self.scene_group_templates:
            for t in templates:
                if self.frame_sampling == "consecutive":
                    g = t
                else:
                    for _ in range(50):  # retry up to 50 times to find valid group
                        g = sorted(rng.choice(len(idxs), size=self.num_frames, replace=False).tolist())
                        g = [idxs[i] for i in g]
                        if self._group_has_valid_ids(g):
                            break
                    else:
                        continue  # skip if no valid group found
                self.groups.append(g)

    def set_epoch(self, epoch):
        """Set epoch for shuffling (and for random frame sampling: reseed RNG)"""
        self.epoch = epoch
        if self.frame_sampling == "random":
            self._build_groups()

    def __len__(self):
        if self.distributed:
            # Each rank gets a subset of batches
            num_batches = len(self.groups) // self.batch_size
            return (num_batches + self.world_size - 1) // self.world_size
        else:
            return len(self.groups) // self.batch_size

    def __iter__(self):
        groups = self.groups.copy()
        
        # Shuffle groups if needed
        if self.shuffle:
            # Use epoch-based seed for reproducibility
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            indices = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in indices]
        
        if self.distributed:
            # Split groups across ranks. Each rank must yield the same number of
            # batches to avoid DDP hangs (all ranks must participate in every step).
            num_batches = len(groups) // self.batch_size
            batches_per_rank = (num_batches + self.world_size - 1) // self.world_size
            start_batch = self.rank * batches_per_rank

            for j in range(batches_per_rank):
                # Wrap batch index so every rank yields exactly batches_per_rank
                # (pad by repeating from start when num_batches not divisible by world_size)
                i = (start_batch + j) % num_batches if num_batches > 0 else 0
                batch_start = i * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_groups = groups[batch_start:batch_end]

                if len(batch_groups) == self.batch_size:
                    flat = [idx for g in batch_groups for idx in g]
                    yield flat
        else:
            # Original non-distributed behavior
            for i in range(0, len(groups), self.batch_size):
                batch_groups = groups[i:i + self.batch_size]

                if len(batch_groups) < self.batch_size:
                    break  # drop last incomplete batch

                # flatten indices
                flat = [idx for g in batch_groups for idx in g]

                yield flat  # DataLoader sees the whole batch


def _safe_int_sort(values):
    try:
        return sorted(values, key=lambda s: int(s))
    except Exception:
        return sorted(values)


def _load_pose_position(pose_path):
    pose = np.loadtxt(pose_path).astype(np.float32)
    return pose[:3, 3]


def _farthest_point_sample_stems(frame_stems, positions, num_samples):
    n = len(frame_stems)
    if num_samples >= n:
        return frame_stems.copy()
    selected_indices = [0]
    distances = np.linalg.norm(positions - positions[0], axis=1)
    for _ in range(num_samples - 1):
        farthest_idx = int(np.argmax(distances))
        selected_indices.append(farthest_idx)
        new_dists = np.linalg.norm(positions - positions[farthest_idx], axis=1)
        distances = np.minimum(distances, new_dists)
    return [frame_stems[i] for i in selected_indices]


class ObjectUnionSceneBatchSampler(Sampler):
    """
    Scene-level object-aware sampler:
    - each yielded group corresponds to ONE training sample in ONE scene
    - group size is num_frames_per_object
    - per scene, sample up to num_objects slots per epoch
    - each slot transitions from random same-scene frames to the existing
      pose-diverse object-union strategy as epoch increases
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_frames_per_object=8,
        num_objects=None,
        target_total_frames=100,
        pose_diverse_transition_epoch=None,
        distributed=False,
        rank=0,
        world_size=1,
        shuffle=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_frames_per_object = int(num_frames_per_object)
        if self.num_frames_per_object <= 0:
            raise ValueError("num_frames_per_object must be > 0")
        if num_objects is not None:
            self.num_objects = int(num_objects)
        elif target_total_frames is not None:
            self.num_objects = max(1, int(round(
                float(target_total_frames) / float(self.num_frames_per_object)
            )))
        else:
            # None means sample ALL valid objects per scene each epoch.
            self.num_objects = None
        if self.num_objects is not None and self.num_objects <= 0:
            raise ValueError("num_objects must be > 0")

        # One dataloader sample/group carries only per-object frames.
        self.frames_per_scene_group = self.num_frames_per_object
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
        self.pose_diverse_transition_epoch = (
            None if pose_diverse_transition_epoch is None else int(pose_diverse_transition_epoch)
        )

        self.scene_ids = []
        for scene_id, indices in dataset.scenes.items():
            if len(indices) == 0:
                continue
            object_pool = dataset.scene_infos.get(scene_id, {}).get("object_union_ids", [])
            if len(object_pool) == 0:
                continue
            self.scene_ids.append(scene_id)

        # Both caches walk NFS (one pose/.txt and one instance_presence/.json
        # per frame); build on rank 0 only and broadcast to the other ranks.
        self.scene_pose_positions = _rank0_broadcast(self._build_pose_cache)
        self.scene_instance_ids_per_frame = _rank0_broadcast(self._build_instance_presence_cache)
        self._build_groups()

    def _get_pose_diverse_probability(self):
        if self.pose_diverse_transition_epoch is None:
            return 1.0
        if self.pose_diverse_transition_epoch <= 0:
            return 1.0
        return float(np.clip(self.epoch / float(self.pose_diverse_transition_epoch), 0.0, 1.0))

    def _build_pose_cache(self):
        scene_pose_positions = {}
        for scene_id, info in self.dataset.scene_infos.items():
            pose_dir = info.get("pose_dir", None)
            stems = info.get("frame_stems", [])
            pos_map = {}
            if pose_dir is not None and os.path.isdir(pose_dir):
                for stem in stems:
                    pose_path = os.path.join(pose_dir, f"{stem}.txt")
                    if not os.path.isfile(pose_path):
                        continue
                    try:
                        pos_map[stem] = _load_pose_position(pose_path)
                    except Exception:
                        continue
            scene_pose_positions[scene_id] = pos_map
        return scene_pose_positions

    def _build_instance_presence_cache(self):
        """
        Build per-frame instance-id sets from precomputed JSON files.
        This is used for strict-negative sampling: a negative frame must not
        contain the sampled object at all, even if it is below valid_ids threshold.
        """
        scene_instance_ids_per_frame = {}
        for scene_id, info in self.dataset.scene_infos.items():
            scene_name = info.get("scene_name", f"{scene_id}")
            frame_stems = info.get("frame_stems", [])
            presence_dir = info.get("instance_presence_dir", None)
            if presence_dir is None or (not os.path.isdir(presence_dir)):
                raise FileNotFoundError(
                    f"Missing instance presence directory for scene '{scene_name}': {presence_dir}. "
                    "Please run offline preprocessing to generate per-frame files in 'instance_presence/'."
                )
            ids_per_frame = {}
            for stem in frame_stems:
                ids_set = set()
                presence_path = os.path.join(presence_dir, f"{stem}.json")
                if not os.path.isfile(presence_path):
                    raise FileNotFoundError(
                        f"Missing instance presence file: {presence_path}. "
                        "Please run offline preprocessing before training."
                    )
                with open(presence_path, "r") as f:
                    payload = json.load(f)
                ids = payload.get("ids", [])
                ids_set = set(int(v) for v in ids)
                ids_per_frame[stem] = ids_set
            scene_instance_ids_per_frame[scene_id] = ids_per_frame
        return scene_instance_ids_per_frame

    def _sample_object_ids(self, object_pool, rng):
        if len(object_pool) == 0:
            return []
        if self.num_objects is None:
            # Sample ALL objects (shuffle order for randomness)
            perm = rng.permutation(len(object_pool)).tolist()
            return [int(object_pool[i]) for i in perm]
        take = min(self.num_objects, len(object_pool))
        perm = rng.permutation(len(object_pool)).tolist()
        return [int(object_pool[i]) for i in perm[:take]]

    def _group_has_valid_ids(self, group_indices):
        ids = [self.dataset.items[j]["valid_ids"] for j in group_indices]
        union = (
            torch.unique(torch.cat(ids, dim=0))
            if any(t.numel() > 0 for t in ids)
            else torch.empty(0, dtype=torch.int64)
        )
        return union.numel() > 0

    def _sample_frames_for_object(self, scene_id, object_id, rng):
        info = self.dataset.scene_infos[scene_id]
        frame_stems = list(info["frame_stems"])
        valid_ids_per_frame = info["valid_ids_per_frame"]
        pose_pos_map = self.scene_pose_positions.get(scene_id, {})
        instance_ids_per_frame = self.scene_instance_ids_per_frame.get(scene_id, {})

        obj_frames = [s for s in frame_stems if object_id in valid_ids_per_frame.get(s, [])]
        obj_frames = _safe_int_sort(obj_frames)

        diverse_budget = self.num_frames_per_object // 2
        diverse_selected = []
        if diverse_budget > 0 and len(obj_frames) > 0:
            pose_stems = [s for s in obj_frames if s in pose_pos_map]
            if len(pose_stems) > 0:
                positions = np.stack([pose_pos_map[s] for s in pose_stems], axis=0)
                diverse_count = min(diverse_budget, len(pose_stems))
                diverse_selected = _farthest_point_sample_stems(pose_stems, positions, diverse_count)
            else:
                diverse_count = min(diverse_budget, len(obj_frames))
                choose_idx = rng.choice(len(obj_frames), size=diverse_count, replace=False).tolist()
                diverse_selected = [obj_frames[i] for i in choose_idx]

        # Strict negative: object id must be absent in raw instance mask IDs.
        negative_frames = [s for s in frame_stems if object_id not in instance_ids_per_frame.get(s, set())]
        negative_frames = _safe_int_sort(negative_frames)

        negative_count = self.num_frames_per_object - len(diverse_selected)
        negative_selected = []
        if negative_count > 0:
            # Enforce strict negatives for the non-diverse half.
            if len(negative_frames) == 0:
                return None
            replace = len(negative_frames) < negative_count
            choose_idx = rng.choice(len(negative_frames), size=negative_count, replace=replace).tolist()
            negative_selected = [negative_frames[i] for i in choose_idx]
        total_count = len(diverse_selected) + len(negative_selected)
        if total_count == 0:
            return None

        # Randomize frame ordering so positives are not always front-loaded.
        # Keep pose-diverse positives in their original relative order, but place
        # them at random positions; fill remaining positions with negatives.
        if len(diverse_selected) == 0:
            return negative_selected
        if len(negative_selected) == 0:
            return diverse_selected

        pos_slots = sorted(
            rng.choice(total_count, size=len(diverse_selected), replace=False).tolist()
        )
        neg_order = rng.permutation(len(negative_selected)).tolist()
        negative_shuffled = [negative_selected[i] for i in neg_order]

        pos_i = 0
        neg_i = 0
        pos_slot_set = set(pos_slots)
        mixed = []
        for slot_idx in range(total_count):
            if slot_idx in pos_slot_set:
                mixed.append(diverse_selected[pos_i])
                pos_i += 1
            else:
                mixed.append(negative_shuffled[neg_i])
                neg_i += 1
        return mixed

    def _sample_random_scene_group(self, scene_id, rng):
        indices = self.dataset.scenes[scene_id]
        replace = len(indices) < self.frames_per_scene_group
        for _ in range(50):
            choice = rng.choice(len(indices), size=self.frames_per_scene_group, replace=replace).tolist()
            group_indices = [int(indices[i]) for i in choice]
            if self._group_has_valid_ids(group_indices):
                return [(ds_idx, -1) for ds_idx in group_indices]
        return None

    def _sample_group_for_scene_object(self, scene_id, object_id, rng):
        info = self.dataset.scene_infos[scene_id]
        indices = self.dataset.scenes[scene_id]
        stem_to_dataset_idx = {self.dataset.items[idx]["frame_stem"]: idx for idx in indices}

        group = []
        stems_for_obj = self._sample_frames_for_object(scene_id, object_id, rng)
        if stems_for_obj is None or len(stems_for_obj) == 0:
            return None
        for stem in stems_for_obj:
            ds_idx = stem_to_dataset_idx.get(stem, None)
            if ds_idx is None:
                ridx = int(rng.choice(len(indices), size=1, replace=True).item())
                ds_idx = indices[ridx]
            group.append((int(ds_idx), int(object_id)))

        while len(group) < self.frames_per_scene_group:
            ridx = int(rng.choice(len(indices), size=1, replace=True).item())
            group.append((int(indices[ridx]), int(object_id)))

        return group[: self.frames_per_scene_group]

    def _build_groups(self):
        self.groups = []
        rng = np.random.default_rng(self.epoch)
        pose_diverse_probability = self._get_pose_diverse_probability()
        for scene_id in self.scene_ids:
            object_pool = self.dataset.scene_infos.get(scene_id, {}).get("object_union_ids", [])
            sampled_objects = self._sample_object_ids(object_pool, rng)
            for obj_id in sampled_objects:
                use_pose_diverse = rng.random() < pose_diverse_probability
                if use_pose_diverse:
                    g = self._sample_group_for_scene_object(scene_id, obj_id, rng)
                else:
                    g = self._sample_random_scene_group(scene_id, rng)
                if g is not None and len(g) == self.frames_per_scene_group:
                    self.groups.append(g)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._build_groups()

    def __len__(self):
        if self.distributed:
            num_batches = len(self.groups) // self.batch_size
            return (num_batches + self.world_size - 1) // self.world_size
        return len(self.groups) // self.batch_size

    def __iter__(self):
        groups = self.groups.copy()
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in order]

        if self.distributed:
            num_batches = len(groups) // self.batch_size
            batches_per_rank = (num_batches + self.world_size - 1) // self.world_size
            start_batch = self.rank * batches_per_rank
            for j in range(batches_per_rank):
                if num_batches == 0:
                    break
                i = (start_batch + j) % num_batches
                batch_start = i * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_groups = groups[batch_start:batch_end]
                if len(batch_groups) == self.batch_size:
                    flat = [idx for g in batch_groups for idx in g]
                    yield flat
        else:
            for i in range(0, len(groups), self.batch_size):
                batch_groups = groups[i:i + self.batch_size]
                if len(batch_groups) < self.batch_size:
                    break
                flat = [idx for g in batch_groups for idx in g]
                yield flat


def multiview_collate(batch, num_frames=8):
    """
    batch: list of samples = batch_size * num_frames
    Each sample contains:
        image: [3,H,W]
        label: [1,H,W]
        valid_ids: [K_i]
    """
    # stack images and labels
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    scene_ids = [b["scene_id"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    label_paths = [b["label_path"] for b in batch]
    sam_embeddings_flat = [b.get("sam_embedding", None) for b in batch]

    # infer batch size
    B = len(batch) // num_frames
    
    # Safety check: ensure we have complete batches
    if B == 0 or len(batch) % num_frames != 0:
        raise RuntimeError(
            f"Batch size mismatch: got {len(batch)} samples but expected a multiple of {num_frames}. "
            f"Got B={B}, which would create invalid shape [B={B}, num_frames={num_frames}, ...]"
        )

    # reshape to [B, N, ...]
    images = images.view(B, num_frames, *images.shape[1:])
    labels = labels.view(B, num_frames, *labels.shape[1:])

    # group-level scene id
    scene_ids = [scene_ids[i*num_frames] for i in range(B)]

    # Collect image paths for all frames in each group
    # image_paths_grouped: list of length B, each entry is a list of num_frames paths
    image_paths_grouped = []
    label_paths_grouped = []
    for b in range(B):
        start = b * num_frames
        end = (b + 1) * num_frames
        image_paths_grouped.append(image_paths[start:end])  # List of num_frames paths
        label_paths_grouped.append(label_paths[start:end])  # List of num_frames paths

    # ----------------------------
    # NEW: compute group-level valid ids
    # ----------------------------
    group_valid_ids = []

    for b in range(B):
        start = b * num_frames
        end   = (b + 1) * num_frames

        # collect valid_ids for this group
        ids_group = []
        for i in range(start, end):
            ids_group.append(batch[i]["valid_ids"])  # [K_i]

        # union across images (concatenate and unique)
        ids_union = torch.unique(torch.cat(ids_group, dim=0))
        group_valid_ids.append(ids_union)

    sam_embeddings_grouped = None
    if all(e is not None for e in sam_embeddings_flat):
        sam_embeddings = torch.stack(sam_embeddings_flat, dim=0)
        sam_embeddings_grouped = sam_embeddings.view(B, num_frames, *sam_embeddings.shape[1:])

    return {
        "images": images,
        "labels": labels,
        "scene_id": scene_ids,
        "image_paths": image_paths_grouped,  # list of length B, each entry is a list of num_frames paths
        "label_paths": label_paths_grouped,  # list of length B, each entry is a list of num_frames paths
        "valid_ids": group_valid_ids,   # list of length B, each entry is a tensor
        "sam_embeddings": sam_embeddings_grouped,  # Optional [B,N,256,64,64]
    }

def create_dataloader(root_dir=None, batch_size=4, num_frames=8, my_transforms=[], distributed=False, rank=0, world_size=1, shuffle=True, frame_sampling="consecutive", dataset=None) -> DataLoader:
    """
    Args:
        frame_sampling: "consecutive" = num_frames consecutive frames per scene,
                        "random" = num_frames randomly selected from same scene.
        dataset: If provided, used directly; otherwise built from *root_dir*.
    """
    if dataset is None:
        dataset = MultiSceneImageDataset(
            root_dir=root_dir,
            transform=transforms.Compose(my_transforms)
        )

    sampler = SceneBatchSampler(
        dataset, 
        batch_size=batch_size, 
        num_frames=num_frames,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        frame_sampling=frame_sampling,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,  # batch_size is handled by SceneBatchSampler
        collate_fn=partial(multiview_collate, num_frames=num_frames),
        num_workers=8,
    )
    return loader


def multiview_object_union_collate(batch, num_frames_per_object=8):
    """
    batch: list of samples = batch_size * num_frames_per_object
    Each sample in the batch corresponds to one sampled object tracklet.
    """
    total_frames = num_frames_per_object

    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    scene_ids = [b["scene_id"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    label_paths = [b["label_path"] for b in batch]
    sampled_object_ids_flat = [int(b.get("sampled_object_id", -1)) for b in batch]
    sam_embeddings_flat = [b.get("sam_embedding", None) for b in batch]

    B = len(batch) // total_frames
    if B == 0 or len(batch) % total_frames != 0:
        raise RuntimeError(
            f"Batch size mismatch: got {len(batch)} samples but expected a multiple of {total_frames}. "
            f"Got B={B}, which would create invalid shape [B={B}, total_frames={total_frames}, ...]"
        )

    images = images.view(B, total_frames, *images.shape[1:])
    labels = labels.view(B, total_frames, *labels.shape[1:])
    scene_ids = [scene_ids[i * total_frames] for i in range(B)]

    image_paths_grouped = []
    label_paths_grouped = []
    group_valid_ids = []
    group_object_ids = []

    for b in range(B):
        start = b * total_frames
        end = (b + 1) * total_frames
        image_paths_grouped.append(image_paths[start:end])
        label_paths_grouped.append(label_paths[start:end])

        ids_union = torch.unique(torch.cat([batch[i]["valid_ids"] for i in range(start, end)], dim=0))
        group_valid_ids.append(ids_union)

        # One sampled object id per group/sample.
        group_object_ids.append(torch.tensor(sampled_object_ids_flat[start], dtype=torch.int64))

    sam_embeddings_grouped = None
    if all(e is not None for e in sam_embeddings_flat):
        sam_embeddings = torch.stack(sam_embeddings_flat, dim=0)
        sam_embeddings_grouped = sam_embeddings.view(B, total_frames, *sam_embeddings.shape[1:])

    return {
        "images": images,
        "labels": labels,
        "scene_id": scene_ids,
        "image_paths": image_paths_grouped,
        "label_paths": label_paths_grouped,
        "valid_ids": group_valid_ids,
        "sampled_object_ids": group_object_ids,  # list[B], each scalar tensor
        "sam_embeddings": sam_embeddings_grouped,  # Optional [B,N,256,64,64]
    }


def create_object_union_dataloader(
    root_dir=None,
    batch_size=2,
    num_frames_per_object=8,
    num_objects=None,
    target_total_frames=None,
    pose_diverse_transition_epoch=None,
    my_transforms=[],
    distributed=False,
    rank=0,
    world_size=1,
    shuffle=True,
    dataset=None,
) -> DataLoader:
    """
    New scene dataloader that samples objects from object_union.json and then
    yields one sample at a time:
      - sample up to num_objects slots per scene per epoch
      - each sample has num_frames_per_object frames
      - optionally transition from random same-scene groups to pose-diverse
        object-union groups by epoch

    If *dataset* is provided it is used directly; otherwise a
    MultiSceneImageDataset is built from *root_dir*.
    """
    if dataset is None:
        dataset = MultiSceneImageDataset(
            root_dir=root_dir,
            transform=transforms.Compose(my_transforms)
        )

    sampler = ObjectUnionSceneBatchSampler(
        dataset,
        batch_size=batch_size,
        num_frames_per_object=num_frames_per_object,
        num_objects=num_objects,
        target_total_frames=target_total_frames,
        pose_diverse_transition_epoch=pose_diverse_transition_epoch,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(
            multiview_object_union_collate,
            num_frames_per_object=num_frames_per_object,
        ),
        num_workers=8,
    )
    return loader




