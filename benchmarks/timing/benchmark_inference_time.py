"""
Inference Time Comparison: IGGT vs SAM-VGGT vs SAM2
====================================================
Benchmarks core inference time (excluding image I/O) for three methods
on the 3D Tracking Benchmark (ScanNet / ScanNet++).

Each model can be run separately in its own conda environment, saving
partial results to JSON. A merge step combines them into one report.

IGGT:      forward pass + pose/depth decoding + KNN smoothing + DBSCAN clustering
SAM-VGGT:  SAM proposal generation (per-frame AMG + point sampling) +
           SamVGGT decode with proposal prompt_groups + NMS
SAM2:      per-frame AMG + video propagation (forward+backward) + track NMS

Usage (separate envs):
    conda activate iggt
    python benchmark_inference_time.py --model iggt --dataset both

    conda activate sam_vggt
    python benchmark_inference_time.py --model sam_vggt --dataset both

    conda activate iggt
    python benchmark_inference_time.py --model sam2 --dataset both

    python benchmark_inference_time.py --merge \\
        benchmark_results_iggt.json \\
        benchmark_results_sam_vggt.json \\
        benchmark_results_sam2.json
"""

import gc
import json
import os
import sys
import glob
import time
import shutil
import inspect
import logging
import tempfile
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — only add paths that exist; model-specific imports are deferred
# ---------------------------------------------------------------------------

FILE_DIR = Path(__file__).resolve().parent          # <repo>/benchmarks/timing
REPO_ROOT = FILE_DIR.parent.parent                  # <repo>            (masks/, model/)
IGGT_DIR = REPO_ROOT / "iggt"                       # <repo>/iggt       (utils.model)
IGGT_ROOT = IGGT_DIR / "iggt"                       # <repo>/iggt/iggt  (vendored `iggt` pkg)
# NOTE: sam-hq/ and vggt/ sat directly under the project root in the older
# nested layout this file was written for; today they are git submodules.

def _setup_sys_path(model_name: str):
    """Add only the paths needed for the selected model to avoid conflicts.

    The key issue: REPO_ROOT contains an `iggt/` subdirectory (= IGGT_DIR)
    that creates a competing namespace-package portion with a different `utils/`
    package, breaking `from iggt.utils.* import ...` for IGGT.  So we must not
    have both IGGT_DIR and REPO_ROOT on sys.path simultaneously.
    """
    paths: List[str] = []

    if model_name == "iggt":
        paths = [str(IGGT_DIR)]
    elif model_name == "sam_vggt":
        paths = [
            str(REPO_ROOT),
            str(REPO_ROOT / "submodules" / "sam-hq"),
            str(REPO_ROOT / "submodules" / "vggt"),
        ]
    elif model_name == "sam2":
        paths = [
            str(REPO_ROOT / "sam2"),
            str(IGGT_DIR),
        ]
    else:
        # merge mode — no model imports needed
        return

    for p in reversed(paths):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Configs (mirroring each evaluation script)
# ---------------------------------------------------------------------------

IGGT_IMAGE_SIZE = (504, 336)
IGGT_CLUSTERING_CONFIG = {
    "eps": 0.06,
    "min_samples": 100,
    "min_cluster_size": 500,
    "knn_k": 20,
}

SAM_VGGT_TARGET_SIZE = (1024, 1024)
SAM_VGGT_CONFIG = {
    "points_per_side": 16,
    "points_per_batch": 8,
    "pred_iou_thresh": 0.40,
    "stability_score_thresh": 0.20,
    "box_nms_thresh": 0.95,
}

PROPOSAL_SAM_MODEL_TYPE = "vit_h"
PROPOSAL_POINTS_PER_SIDE = 32
PROPOSAL_POINTS_PER_BATCH = 64
PROPOSAL_PRED_IOU_THRESH = 0.88
PROPOSAL_STABILITY_SCORE_THRESH = 0.95
PROPOSAL_BOX_NMS_THRESH = 0.7
PROPOSAL_MIN_MASK_REGION_AREA = 0
PROPOSAL_MAX_MASKS_PER_FRAME = 64

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_AMG_CONFIG = {
    "points_per_side": 32,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "box_nms_thresh": 0.7,
    "min_mask_region_area": 0,
    "multimask_output": True,
}
SAM2_MAX_MASKS_PER_FRAME = 64
SAM2_TRACK_NMS_THRESH = 0.7

# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def timed(label: str = ""):
    """Context manager that returns elapsed seconds after torch.cuda.synchronize."""
    class _Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *exc):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self._t0
            if label:
                logger.info(f"  [{label}] {self.elapsed:.3f}s")
    return _Timer()


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




# ---------------------------------------------------------------------------
# Scene discovery (shared across all methods)
# ---------------------------------------------------------------------------

def load_tracking_benchmark(scene_dir):
    img_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(img_dir):
        scene_name = os.path.basename(scene_dir)
        alt_dir = os.path.join(scene_dir, scene_name)
        if os.path.isdir(alt_dir):
            img_dir = alt_dir

    label_files = sorted(glob.glob(os.path.join(img_dir, "*_label.npy")))
    frames = [os.path.basename(lf).replace("_label.npy", "") for lf in label_files]

    image_paths = []
    frame_names = []
    for frame in frames:
        img_path = os.path.join(img_dir, f"{frame}.jpg")
        label_path = os.path.join(img_dir, f"{frame}_label.npy")
        if os.path.exists(img_path) and os.path.exists(label_path):
            image_paths.append(img_path)
            frame_names.append(frame)

    return image_paths, frame_names


def discover_scenes(datasets: List[str]) -> List[Tuple[str, str, str]]:
    """Return list of (dataset_name, scene_name, scene_dir_path)."""
    all_scenes: List[Tuple[str, str, str]] = []
    for ds_name in datasets:
        benchmark_dir = os.path.join(
            os.environ.get("BENCH", ""), ds_name
        )
        if not os.path.isdir(benchmark_dir):
            logger.warning(f"Benchmark dir not found: {benchmark_dir}, skipping {ds_name}")
            continue
        scene_names = sorted([
            d for d in os.listdir(benchmark_dir)
            if os.path.isdir(os.path.join(benchmark_dir, d))
        ])
        for sn in scene_names:
            all_scenes.append((ds_name, sn, os.path.join(benchmark_dir, sn)))
    return all_scenes


# ===================================================================
# IGGT
# ===================================================================

def _align_state_dicts(model_state_dict, ckpt_state_dict):
    """Match checkpoint keys to model keys by name+shape (no detectron2 dep)."""
    result = {}
    ckpt_remaining = set(ckpt_state_dict.keys())
    for key, weight in model_state_dict.items():
        if key in ckpt_state_dict and weight.shape == ckpt_state_dict[key].shape:
            result[key] = ckpt_state_dict[key]
            ckpt_remaining.discard(key)
        else:
            logger.debug(f"  IGGT ckpt: skipped key {key}")
    if ckpt_remaining:
        logger.debug(f"  IGGT ckpt: {len(ckpt_remaining)} unused keys")
    return result


def load_iggt_model(ckpt_path):
    from iggt.models.vggt import IGGT

    logger.info(f"Loading IGGT model from {ckpt_path} ...")
    model = IGGT()
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    state_dict = _align_state_dicts(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(DEVICE)
    logger.info("IGGT model loaded")
    return model


def _iggt_inference_core(model, images_tensor):
    """Core IGGT inference on a pre-loaded tensor. Returns label maps."""
    from iggt.utils.pose_enc import pose_encoding_to_extri_intri
    from iggt.utils.geometry import unproject_depth_map_to_point_map
    from iggt.utils.misc import cluster_features_to_masks_mv, knn_avg_features_pyg

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images_tensor)

    predictions["pose_enc"] = predictions["pose_enc"][-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images_tensor.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    part_feature = torch.from_numpy(predictions["part_feat"]).permute(0, 2, 3, 1)
    part_feature = F.normalize(part_feature, dim=3)

    spatial_knn_part_features = knn_avg_features_pyg(
        world_points, part_feature, k=IGGT_CLUSTERING_CONFIG["knn_k"]
    )

    dbscan_result = cluster_features_to_masks_mv(
        spatial_knn_part_features,
        method="dbscan",
        eps=IGGT_CLUSTERING_CONFIG["eps"],
        min_samples=IGGT_CLUSTERING_CONFIG["min_samples"],
        min_cluster_size=IGGT_CLUSTERING_CONFIG["min_cluster_size"],
        apply_colormap=True,
    )
    return dbscan_result[0]


def _load_iggt_images(image_paths, target_size):
    """Load and resize images for IGGT (bicubic resize, ToTensor)."""
    from torchvision import transforms as TF
    to_tensor = TF.ToTensor()
    images = []
    w, h = target_size
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((w, h), Image.Resampling.BICUBIC)
        images.append(to_tensor(img))
    return torch.stack(images)


def time_iggt(model, image_paths):
    """Time IGGT inference (excluding image loading)."""
    images = _load_iggt_images(image_paths, IGGT_IMAGE_SIZE).to(DEVICE)

    with timed("IGGT") as t:
        _iggt_inference_core(model, images)

    del images
    clear_gpu()
    return t.elapsed


# ===================================================================
# SAM-VGGT (with SAM proposal prompts)
# ===================================================================

def load_sam_vggt_model(ckpt_path):
    from sam_vggt_model import build_sam_vggt
    from everything_mode_demo import load_checkpoint

    logger.info(f"Loading SAM-VGGT model from {ckpt_path} ...")
    model = build_sam_vggt(
        sam_model_type="vit_h",
        device=DEVICE,
        sam_encode_chunk=0,
    )
    load_checkpoint(model, ckpt_path, torch.device(DEVICE))
    logger.info("SAM-VGGT model loaded")
    return model


def load_proposal_sam_model(ckpt_path):
    from segment_anything import sam_model_registry

    logger.info(f"Loading proposal SAM model ({PROPOSAL_SAM_MODEL_TYPE}) from {ckpt_path} ...")
    model = sam_model_registry[PROPOSAL_SAM_MODEL_TYPE](checkpoint=ckpt_path)
    model.to(device=torch.device(DEVICE)).eval()
    _adapt_mask_decoder_for_sam_hq(model)
    logger.info("Proposal SAM model loaded")
    return model


def _adapt_mask_decoder_for_sam_hq(sam_model):
    """Strip extra kwargs for SAM-HQ mask_decoder if needed."""
    if getattr(sam_model.mask_decoder, "_hq_kwargs_compat_wrapped", False):
        return
    forward_sig = inspect.signature(sam_model.mask_decoder.forward)
    if "hq_token_only" in forward_sig.parameters and "interm_embeddings" in forward_sig.parameters:
        sam_model.mask_decoder._hq_kwargs_compat_wrapped = True
        return
    original_forward = sam_model.mask_decoder.forward

    def forward_compat(*args, **kwargs):
        kwargs.pop("hq_token_only", None)
        kwargs.pop("interm_embeddings", None)
        return original_forward(*args, **kwargs)

    sam_model.mask_decoder.forward = forward_compat
    sam_model.mask_decoder._hq_kwargs_compat_wrapped = True


def _image_tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    return image_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)


def _decode_mask(mask_data) -> np.ndarray:
    if isinstance(mask_data, np.ndarray):
        return mask_data.astype(bool, copy=False)
    if torch.is_tensor(mask_data):
        return mask_data.detach().cpu().numpy().astype(bool, copy=False)
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
        return flat.reshape((width, height)).T.astype(bool, copy=False)
    raise TypeError(f"Unsupported mask type: {type(mask_data)!r}")


def _load_images_tensor(image_paths, target_size):
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    images = torch.stack(tensors, dim=0)
    if tuple(images.shape[-2:]) != tuple(target_size):
        images = F.interpolate(images, size=target_size, mode="bilinear", align_corners=False)
    return images


def _generate_proposal_prompts(
    proposal_sam_model,
    images_tensor: torch.Tensor,
    prompt_sampling_method: str,
    prompt_points_per_mask: int,
) -> List[Dict[str, Any]]:
    """Run per-frame SAM AMG and sample prompt points from each proposal."""
    from segment_anything import SamAutomaticMaskGenerator as SamDenseAMG
    from prompt_sampling import sample_prompt_points_from_mask

    dense_generator = SamDenseAMG(
        model=proposal_sam_model,
        points_per_side=PROPOSAL_POINTS_PER_SIDE,
        points_per_batch=PROPOSAL_POINTS_PER_BATCH,
        pred_iou_thresh=PROPOSAL_PRED_IOU_THRESH,
        stability_score_thresh=PROPOSAL_STABILITY_SCORE_THRESH,
        box_nms_thresh=PROPOSAL_BOX_NMS_THRESH,
        min_mask_region_area=PROPOSAL_MIN_MASK_REGION_AREA,
        output_mode="binary_mask",
    )

    rng = np.random.default_rng(0)
    prompt_groups: List[Dict[str, Any]] = []
    next_prompt_id = 0
    N = images_tensor.shape[0]

    for frame_idx in range(N):
        image_np = _image_tensor_to_uint8(images_tensor[frame_idx])
        dense_masks = dense_generator.generate(image_np, multimask_output=True)
        dense_masks.sort(
            key=lambda ann: (
                float(ann.get("predicted_iou", 0.0)),
                float(ann.get("stability_score", 0.0)),
                float(ann.get("area", 0.0)),
            ),
            reverse=True,
        )
        dense_masks = dense_masks[:PROPOSAL_MAX_MASKS_PER_FRAME]

        for ann in dense_masks:
            proposal_mask = _decode_mask(ann["segmentation"])
            if not np.any(proposal_mask):
                continue
            points = sample_prompt_points_from_mask(
                proposal_mask,
                method=prompt_sampling_method,
                num_points=prompt_points_per_mask,
                rng=rng,
            )
            prompt_groups.append({
                "prompt_id": int(next_prompt_id),
                "frame_index": int(frame_idx),
                "points": points.astype(np.float32, copy=False),
                "labels": np.ones((len(points),), dtype=np.int64),
            })
            next_prompt_id += 1

    return prompt_groups


def time_sam_vggt(
    sam_vggt_model,
    proposal_sam_model,
    image_paths,
    prompt_sampling_method: str,
    prompt_points_per_mask: int,
):
    """Time SAM-VGGT inference in two stages: proposal gen + VGGT decode."""
    from automatic_mask_generator import SamVGGTAutomaticMaskGenerator

    images = _load_images_tensor(image_paths, SAM_VGGT_TARGET_SIZE)

    # Stage 1: SAM proposal generation
    with timed("SAM-VGGT proposal") as t_proposal:
        prompt_groups = _generate_proposal_prompts(
            proposal_sam_model, images, prompt_sampling_method, prompt_points_per_mask
        )

    logger.info(f"  Generated {len(prompt_groups)} proposal prompt groups")

    # Stage 2: SAM-VGGT decode with proposal prompts
    generator = SamVGGTAutomaticMaskGenerator(
        model=sam_vggt_model,
        points_per_side=SAM_VGGT_CONFIG["points_per_side"],
        points_per_batch=SAM_VGGT_CONFIG["points_per_batch"],
        pred_iou_thresh=SAM_VGGT_CONFIG["pred_iou_thresh"],
        stability_score_thresh=SAM_VGGT_CONFIG["stability_score_thresh"],
        box_nms_thresh=SAM_VGGT_CONFIG["box_nms_thresh"],
        output_mode="binary_mask",
    )

    with timed("SAM-VGGT inference") as t_inference:
        with torch.no_grad():
            annotations = generator.generate(images, prompt_groups=prompt_groups)

    logger.info(f"  SAM-VGGT produced {len(annotations)} annotations")

    del images, prompt_groups, annotations
    clear_gpu()

    return {
        "proposal_sec": t_proposal.elapsed,
        "inference_sec": t_inference.elapsed,
        "total_sec": t_proposal.elapsed + t_inference.elapsed,
    }


# ===================================================================
# SAM2
# ===================================================================

def load_sam2_models(ckpt_path):
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    logger.info(f"Loading SAM2 from {ckpt_path} ...")
    sam2_image = build_sam2(SAM2_CFG, ckpt_path, device=DEVICE, apply_postprocessing=False)
    sam2_video = build_sam2_video_predictor(SAM2_CFG, ckpt_path, device=DEVICE)
    logger.info("SAM2 models loaded")
    return sam2_image, sam2_video


def _sam2_inference_core(sam2_image, sam2_video, image_paths, target_size):
    """Core SAM2 inference: per-frame AMG + video propagation + track NMS."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from torchvision.ops import batched_nms

    N = len(image_paths)
    H_t, W_t = target_size

    amg = SAM2AutomaticMaskGenerator(
        sam2_image,
        points_per_side=SAM2_AMG_CONFIG["points_per_side"],
        points_per_batch=SAM2_AMG_CONFIG["points_per_batch"],
        pred_iou_thresh=SAM2_AMG_CONFIG["pred_iou_thresh"],
        stability_score_thresh=SAM2_AMG_CONFIG["stability_score_thresh"],
        box_nms_thresh=SAM2_AMG_CONFIG["box_nms_thresh"],
        min_mask_region_area=SAM2_AMG_CONFIG["min_mask_region_area"],
        multimask_output=SAM2_AMG_CONFIG["multimask_output"],
        output_mode="binary_mask",
    )

    with tempfile.TemporaryDirectory(prefix="sam2_bench_") as td:
        td_path = Path(td)
        for i, p in enumerate(image_paths):
            shutil.copy(p, td_path / f"{i:05d}.jpg")

        state = sam2_video.init_state(video_path=str(td_path))
        H_v = state["video_height"]
        W_v = state["video_width"]

        all_tracks = []
        all_scores = []

        for init_f in range(N):
            img_np = np.array(Image.open(image_paths[init_f]).convert("RGB"))
            with torch.inference_mode():
                raw_masks = amg.generate(img_np)
            raw_masks.sort(
                key=lambda a: (
                    float(a.get("predicted_iou", 0.0)),
                    float(a.get("stability_score", 0.0)),
                    float(a.get("area", 0.0)),
                ),
                reverse=True,
            )
            raw_masks = raw_masks[:SAM2_MAX_MASKS_PER_FRAME]
            raw_masks = [m for m in raw_masks if m["segmentation"].any()]
            K = len(raw_masks)
            if K == 0:
                continue

            sam2_video.reset_state(state)
            for k, m in enumerate(raw_masks):
                sam2_video.add_new_mask(
                    inference_state=state,
                    frame_idx=init_f,
                    obj_id=k,
                    mask=m["segmentation"].astype(bool, copy=False),
                )

            batch_volume = np.zeros((K, N, H_v, W_v), dtype=bool)
            with torch.inference_mode():
                for f, obj_ids, vrm in sam2_video.propagate_in_video(
                    state, start_frame_idx=init_f, reverse=False,
                ):
                    m_np = (vrm[:, 0] > 0.0).detach().cpu().numpy()
                    for slot, oid in enumerate(obj_ids):
                        batch_volume[oid, f] = m_np[slot]

                if init_f > 0:
                    for f, obj_ids, vrm in sam2_video.propagate_in_video(
                        state, start_frame_idx=init_f, reverse=True,
                    ):
                        m_np = (vrm[:, 0] > 0.0).detach().cpu().numpy()
                        for slot, oid in enumerate(obj_ids):
                            batch_volume[oid, f] = m_np[slot]

            for k in range(K):
                if tuple(batch_volume[k].shape[-2:]) != (H_t, W_t):
                    t = torch.from_numpy(batch_volume[k].astype(np.float32))[None]
                    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
                    track = (t[0] > 0.5).cpu().numpy()
                else:
                    track = batch_volume[k].copy()
                all_tracks.append(track)
                all_scores.append(float(raw_masks[k].get("predicted_iou", 0.0)))

            del batch_volume
            torch.cuda.empty_cache()

        sam2_video.reset_state(state)

    if not all_tracks:
        return np.zeros((0, N, H_t, W_t), dtype=bool)

    # Track NMS
    def _panoramic_bbox(track):
        if not track.any():
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        W = track.shape[2]
        y_any = track.any(axis=(0, 2))
        y_idx = np.where(y_any)[0]
        y1, y2 = float(y_idx[0]), float(y_idx[-1] + 1)
        nonempty = np.where(track.any(axis=(1, 2)))[0]
        f_first, f_last = int(nonempty[0]), int(nonempty[-1])
        x_first = int(np.where(track[f_first].any(axis=0))[0][0])
        x_last = int(np.where(track[f_last].any(axis=0))[0][-1])
        return np.array([f_first * W + x_first, y1, f_last * W + x_last + 1, y2], dtype=np.float32)

    boxes = np.stack([_panoramic_bbox(t) for t in all_tracks])
    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    idxs_t = torch.zeros(len(all_tracks), dtype=torch.long)
    keep = batched_nms(boxes_t, scores_t, idxs_t, iou_threshold=SAM2_TRACK_NMS_THRESH).tolist()
    keep_sorted = sorted(int(i) for i in keep)

    pred_masks = np.stack([all_tracks[i] for i in keep_sorted], axis=0)
    return pred_masks


def time_sam2(sam2_image, sam2_video, image_paths):
    """Time SAM2 inference (excluding image loading to numpy)."""
    with timed("SAM2") as t:
        _sam2_inference_core(sam2_image, sam2_video, image_paths, SAM_VGGT_TARGET_SIZE)

    clear_gpu()
    return t.elapsed


# ===================================================================
# Stats and reporting
# ===================================================================

def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }


def print_summary_table(per_scene: List[Dict], aggregate: Dict):
    """Print a formatted comparison table. Handles partial data (missing models)."""
    has_iggt = any(r.get("iggt_sec") is not None for r in per_scene)
    has_vggt = any(r.get("sam_vggt_total_sec") is not None for r in per_scene)
    has_sam2 = any(r.get("sam2_sec") is not None for r in per_scene)

    cols = [f"{'Scene':<30}", f"{'Frames':>6}"]
    if has_iggt:
        cols.append(f"{'IGGT(s)':>9}")
    if has_vggt:
        cols.extend([f"{'VGGT-prop(s)':>12}", f"{'VGGT-inf(s)':>12}", f"{'VGGT-tot(s)':>12}"])
    if has_sam2:
        cols.append(f"{'SAM2(s)':>9}")

    header = " ".join(cols)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for s in per_scene:
        parts = [f"{s['scene']:<30}", f"{s['num_frames']:>6d}"]
        if has_iggt:
            v = s.get("iggt_sec")
            parts.append(f"{v:>9.2f}" if v is not None else f"{'--':>9}")
        if has_vggt:
            vp = s.get("sam_vggt_proposal_sec")
            vi = s.get("sam_vggt_inference_sec")
            vt = s.get("sam_vggt_total_sec")
            parts.append(f"{vp:>12.2f}" if vp is not None else f"{'--':>12}")
            parts.append(f"{vi:>12.2f}" if vi is not None else f"{'--':>12}")
            parts.append(f"{vt:>12.2f}" if vt is not None else f"{'--':>12}")
        if has_sam2:
            v = s.get("sam2_sec")
            parts.append(f"{v:>9.2f}" if v is not None else f"{'--':>9}")
        print(" ".join(parts))
    print(sep)

    def _fmt(key):
        a = aggregate.get(key)
        if not a:
            return "N/A"
        return (
            f"mean={a['mean']:.2f}  std={a['std']:.2f}  "
            f"min={a['min']:.2f}  max={a['max']:.2f}  total={a['total']:.2f}"
        )

    print()
    if has_iggt:
        print(f"  IGGT            : {_fmt('iggt')}")
    if has_vggt:
        print(f"  VGGT (proposal) : {_fmt('sam_vggt_proposal')}")
        print(f"  VGGT (inference): {_fmt('sam_vggt_inference')}")
        print(f"  VGGT (total)    : {_fmt('sam_vggt_total')}")
    if has_sam2:
        print(f"  SAM2            : {_fmt('sam2')}")
    print()


# ===================================================================
# Single-model benchmark runner
# ===================================================================

def run_benchmark_single_model(model_name: str, args):
    """Run benchmark for one model only, save partial JSON."""
    _setup_sys_path(model_name)

    datasets = []
    if args.dataset in ("scannet", "both"):
        datasets.append("scannet")
    if args.dataset in ("scannetpp", "both"):
        datasets.append("scannetpp")

    all_scenes = discover_scenes(datasets)
    if not all_scenes:
        logger.error("No scenes found. Check --dataset and benchmark_data paths.")
        return

    logger.info(f"Benchmarking model={model_name} on {len(all_scenes)} scenes")

    # Load only the required model(s)
    iggt_model = None
    sam_vggt_model = None
    proposal_sam_model = None
    sam2_image = None
    sam2_video = None

    if model_name == "iggt":
        iggt_model = load_iggt_model(args.iggt_ckpt)
    elif model_name == "sam_vggt":
        sam_vggt_model = load_sam_vggt_model(args.sam_v_ckpt)
        proposal_sam_model = load_proposal_sam_model(args.proposal_sam_ckpt)
    elif model_name == "sam2":
        sam2_image, sam2_video = load_sam2_models(args.sam2_ckpt)

    clear_gpu()

    # Warmup
    if args.warmup_runs > 0:
        _, warmup_name, warmup_dir = all_scenes[0]
        image_paths, _ = load_tracking_benchmark(warmup_dir)
        if image_paths:
            logger.info(f"Running {args.warmup_runs} warmup pass(es) on {warmup_name} ...")
            for wi in range(args.warmup_runs):
                logger.info(f"  Warmup {wi + 1}/{args.warmup_runs}")
                if model_name == "iggt":
                    time_iggt(iggt_model, image_paths)
                elif model_name == "sam_vggt":
                    time_sam_vggt(
                        sam_vggt_model, proposal_sam_model, image_paths,
                        args.prompt_sampling_method, args.prompt_points_per_mask,
                    )
                elif model_name == "sam2":
                    time_sam2(sam2_image, sam2_video, image_paths)
            logger.info("Warmup complete")

    # Benchmark loop
    per_scene_results: List[Dict[str, Any]] = []

    for idx, (ds_name, scene_name, scene_dir) in enumerate(all_scenes):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{idx + 1}/{len(all_scenes)}] {ds_name}/{scene_name}")
        logger.info(f"{'=' * 60}")

        image_paths, frame_names = load_tracking_benchmark(scene_dir)
        if not image_paths:
            logger.warning(f"  No data, skipping {scene_name}")
            continue

        N = len(image_paths)
        logger.info(f"  {N} frames")

        entry: Dict[str, Any] = {
            "dataset": ds_name,
            "scene": scene_name,
            "num_frames": N,
        }

        if model_name == "iggt":
            sec = time_iggt(iggt_model, image_paths)
            entry["iggt_sec"] = round(sec, 4)
            logger.info(f"  => IGGT={sec:.2f}s")

        elif model_name == "sam_vggt":
            vt = time_sam_vggt(
                sam_vggt_model, proposal_sam_model, image_paths,
                args.prompt_sampling_method, args.prompt_points_per_mask,
            )
            entry["sam_vggt_proposal_sec"] = round(vt["proposal_sec"], 4)
            entry["sam_vggt_inference_sec"] = round(vt["inference_sec"], 4)
            entry["sam_vggt_total_sec"] = round(vt["total_sec"], 4)
            logger.info(
                f"  => VGGT={vt['total_sec']:.2f}s "
                f"(prop={vt['proposal_sec']:.2f}+inf={vt['inference_sec']:.2f})"
            )

        elif model_name == "sam2":
            sec = time_sam2(sam2_image, sam2_video, image_paths)
            entry["sam2_sec"] = round(sec, 4)
            logger.info(f"  => SAM2={sec:.2f}s")

        per_scene_results.append(entry)

    # Build config for this run
    config: Dict[str, Any] = {
        "model": model_name,
        "dataset": args.dataset,
        "warmup_runs": args.warmup_runs,
        "device": DEVICE,
        "num_scenes": len(per_scene_results),
    }
    if model_name == "iggt":
        config["iggt_ckpt"] = args.iggt_ckpt
        config["iggt_config"] = IGGT_CLUSTERING_CONFIG
    elif model_name == "sam_vggt":
        config["sam_v_ckpt"] = args.sam_v_ckpt
        config["proposal_sam_ckpt"] = args.proposal_sam_ckpt
        config["prompt_sampling_method"] = args.prompt_sampling_method
        config["prompt_points_per_mask"] = args.prompt_points_per_mask
        config["sam_vggt_config"] = SAM_VGGT_CONFIG
        config["proposal_config"] = {
            "points_per_side": PROPOSAL_POINTS_PER_SIDE,
            "points_per_batch": PROPOSAL_POINTS_PER_BATCH,
            "pred_iou_thresh": PROPOSAL_PRED_IOU_THRESH,
            "stability_score_thresh": PROPOSAL_STABILITY_SCORE_THRESH,
            "box_nms_thresh": PROPOSAL_BOX_NMS_THRESH,
            "max_masks_per_frame": PROPOSAL_MAX_MASKS_PER_FRAME,
        }
    elif model_name == "sam2":
        config["sam2_ckpt"] = args.sam2_ckpt
        config["sam2_amg_config"] = SAM2_AMG_CONFIG

    # Build aggregate for this model
    agg_keys_map = {
        "iggt": [("iggt", "iggt_sec")],
        "sam_vggt": [
            ("sam_vggt_proposal", "sam_vggt_proposal_sec"),
            ("sam_vggt_inference", "sam_vggt_inference_sec"),
            ("sam_vggt_total", "sam_vggt_total_sec"),
        ],
        "sam2": [("sam2", "sam2_sec")],
    }
    aggregate = {}
    for agg_key, scene_key in agg_keys_map[model_name]:
        vals = [r[scene_key] for r in per_scene_results if scene_key in r]
        aggregate[agg_key] = compute_stats(vals)

    output = {
        "config": config,
        "per_scene": per_scene_results,
        "aggregate": aggregate,
    }

    # Default output name includes model name
    if args.output_json == "benchmark_results.json":
        out_name = f"benchmark_results_{model_name}.json"
    else:
        out_name = args.output_json

    output_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    print_summary_table(per_scene_results, aggregate)
    return output_path


# ===================================================================
# Merge mode
# ===================================================================

def merge_results(json_paths: List[str], output_path: str):
    """Merge partial per-model JSONs into one combined report."""
    all_data = []
    for jp in json_paths:
        with open(jp) as f:
            all_data.append(json.load(f))

    # Build a scene index: (dataset, scene) -> merged entry
    scene_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for data in all_data:
        for entry in data["per_scene"]:
            key = (entry["dataset"], entry["scene"])
            if key not in scene_index:
                scene_index[key] = {
                    "dataset": entry["dataset"],
                    "scene": entry["scene"],
                    "num_frames": entry["num_frames"],
                }
            scene_index[key].update({
                k: v for k, v in entry.items()
                if k not in ("dataset", "scene", "num_frames")
            })

    per_scene = list(scene_index.values())
    per_scene.sort(key=lambda e: (e["dataset"], e["scene"]))

    # Aggregate across all models
    agg_pairs = [
        ("iggt", "iggt_sec"),
        ("sam_vggt_proposal", "sam_vggt_proposal_sec"),
        ("sam_vggt_inference", "sam_vggt_inference_sec"),
        ("sam_vggt_total", "sam_vggt_total_sec"),
        ("sam2", "sam2_sec"),
    ]
    aggregate = {}
    for agg_key, scene_key in agg_pairs:
        vals = [r[scene_key] for r in per_scene if r.get(scene_key) is not None]
        if vals:
            aggregate[agg_key] = compute_stats(vals)

    # Merge configs
    merged_config = {"merged_from": json_paths}
    for data in all_data:
        cfg = data.get("config", {})
        model = cfg.get("model", "unknown")
        merged_config[f"{model}_config"] = cfg

    output = {
        "config": merged_config,
        "per_scene": per_scene,
        "aggregate": aggregate,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Merged results saved to {output_path}")

    print_summary_table(per_scene, aggregate)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference time: IGGT vs SAM-VGGT vs SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run each model separately (different conda envs):
  conda activate iggt
  python benchmark_inference_time.py --model iggt --dataset both

  conda activate sam_vggt
  python benchmark_inference_time.py --model sam_vggt --dataset both

  conda activate iggt
  python benchmark_inference_time.py --model sam2 --dataset both

  # Merge results into one report:
  python benchmark_inference_time.py --merge \\
      benchmark_results_iggt.json \\
      benchmark_results_sam_vggt.json \\
      benchmark_results_sam2.json
""",
    )

    parser.add_argument(
        "--model", choices=["iggt", "sam_vggt", "sam2"],
        help="Which model to benchmark. Run one at a time per conda env.",
    )
    parser.add_argument(
        "--merge", nargs="+", metavar="JSON_FILE",
        help="Merge multiple partial result JSONs into one combined report.",
    )
    parser.add_argument(
        "--dataset", choices=["scannet", "scannetpp", "both"], default="both",
        help="Which dataset(s) to benchmark on.",
    )
    parser.add_argument(
        "--iggt_ckpt", type=str,
        default=os.environ.get("IGGT_CHECKPOINT"),
    )
    parser.add_argument(
        "--sam_v_ckpt", type=str,
        default=os.environ.get("SAMV_CHECKPOINT")
    )
    parser.add_argument(
        "--sam2_ckpt", type=str,
        default=str(REPO_ROOT / "submodules" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"),
    )
    parser.add_argument(
        "--proposal_sam_ckpt", type=str,
        default=str(REPO_ROOT / "submodules" / "sam-hq" / "checkpoints" / "sam_vit_h_4b8939.pth"),
    )
    parser.add_argument("--prompt_points_per_mask", type=int, default=5)
    parser.add_argument(
        "--prompt_sampling_method", type=str, default="pole_plus_diverse",
        choices=["centroid", "snapped_centroid", "random_interior",
                 "pole_plus_random", "pole_plus_diverse"],
    )
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument(
        "--output_json", type=str, default="benchmark_results.json",
        help="Output JSON file name. For --model runs, auto-suffixed with model name.",
    )
    args = parser.parse_args()

    if args.merge:
        out_path = os.path.join(os.path.dirname(__file__), args.output_json)
        merge_results(args.merge, out_path)
    elif args.model:
        run_benchmark_single_model(args.model, args)
    else:
        parser.error("Specify either --model <name> to benchmark, or --merge <files> to combine results.")


if __name__ == "__main__":
    main()
