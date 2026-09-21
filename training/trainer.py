"""SAM-VGGT training entry point and engine.

Driven by a YAML config (see ``configs/``) loaded into a :class:`TrainConfig`.
Sampling mode is selected with ``--mode {object_union,random}`` (or the config's
``mode`` field). Run, e.g.::

    PYTHONPATH="$PWD" python training/trainer.py --config configs/object_union.yaml

Visualization helpers live in ``training/viz.py``; all hyperparameters live in
``training/config.py``.
"""

import os
import sys
import argparse
import time
from datetime import timedelta
from pathlib import Path

# Make the sibling modules (config.py / viz.py) importable no matter how this
# file is invoked -- by path, via `python -m`, or from another working
# directory. Relying on Python's implicit sys.path[0] only covers the
# `python training/trainer.py` case.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
import wandb

from model.sam_vggt_model import build_sam_vggt
from utils.dataloader import (
    create_dataloader,
    create_object_union_dataloader,
    MultiSceneImageDataset,
    ScanNetPPMultiSceneImageDataset,
    MergedMultiSceneDataset,
    Resize,
)
from utils.loss_mask import loss_masks, predicted_iou_loss
from utils.checkpoint import (
    TRAINABLE_ONLY_FORMAT,
    is_partial_checkpoint,
    is_slim_checkpoint,
    load_partial_checkpoint,
)
from utils.misc import (
    sample_points_for_instances,
    sample_points_for_instances_single_frame,
)
# Sibling imports (config.py / viz.py live next to this script). We deliberately
# avoid `from training.config import ...`: when the vggt submodule is on the
# PYTHONPATH it ships its own real `training` package that would shadow ours.
import viz
from config import load_config
from viz import (
    FIXED_VAL_VIZ_TARGETS,
    visualize_gt_prompt_overlay,
    visualize_pred_mask_overlay,
    viz_and_pack,
    _save_train_mismatch_visualization,
)


# ======================================================================
#  Prompt / point sampling helpers
# ======================================================================
def sample_prompt_k(cfg):
    """Draw the number of prompt points for one batch, uniform in [k_min, k_max]."""
    return int(torch.randint(cfg.prompt_k_min, cfg.prompt_k_max + 1, (1,)).item())


def single_frame_prompt_ratio(cfg, epoch_idx):
    """Per-sample probability of a single-frame prompt at this epoch.

    ``cfg.prompt_transition_epoch is None`` (the default) disables the
    curriculum: prompts are single-frame from epoch 0, which is what stage 2 and
    every current experiment use. Set it to an epoch E to ramp linearly from 0 at
    epoch 0 to 1 at epoch >= E, reproducing the stage-1 prompt curriculum.
    """
    transition_epoch = cfg.prompt_transition_epoch
    if transition_epoch is None or transition_epoch <= 0:
        return 1.0
    return float(max(0.0, min(1.0, float(epoch_idx) / float(transition_epoch))))


def sample_training_prompt_points(
    cfg,
    labels_cat,
    chosen_ids,
    *,
    k,
    num_frames,
    frame_width,
    single_frame_ratio,
):
    """Sample one training batch's prompt points under the curriculum.

    At ``single_frame_ratio == 1.0`` (curriculum off) this is exactly
    ``sample_points_for_instances_single_frame``. Below 1.0 a per-sample coin
    flip mixes in the multi-frame half-positive / half-negative prompt. The order
    of the calls matters: it is the order the stage-1 run drew them in.
    """
    if single_frame_ratio <= 0.0:
        return sample_points_for_instances(
            labels_cat, chosen_ids, k=k, num_frames=num_frames,
            frame_width=frame_width, positive_only=False,
        )
    if single_frame_ratio >= 1.0:
        return sample_points_for_instances_single_frame(
            labels_cat, chosen_ids, k=k, num_frames=num_frames, frame_width=frame_width,
        )

    multi_points, multi_labels = sample_points_for_instances(
        labels_cat, chosen_ids, k=k, num_frames=num_frames,
        frame_width=frame_width, positive_only=False,
    )
    single_points, single_labels = sample_points_for_instances_single_frame(
        labels_cat, chosen_ids, k=k, num_frames=num_frames, frame_width=frame_width,
    )
    use_single_frame = torch.rand(labels_cat.shape[0], device=labels_cat.device) < single_frame_ratio
    sampled_points = multi_points.clone()
    sampled_labels = multi_labels.clone()
    sampled_points[use_single_frame] = single_points[use_single_frame]
    sampled_labels[use_single_frame] = single_labels[use_single_frame]
    return sampled_points, sampled_labels


def resolve_chosen_id_from_sampled_or_random(sampled_object_id, valid_ids):
    sid = int(sampled_object_id.item()) if torch.is_tensor(sampled_object_id) else int(sampled_object_id)
    if sid >= 0:
        valid_id_set = {int(x) for x in valid_ids.detach().cpu().tolist()}
        if sid in valid_id_set:
            return torch.tensor([sid], dtype=torch.int64)

    rid = torch.randint(0, len(valid_ids), (1,))
    return valid_ids[rid]


# ======================================================================
#  Shared per-batch prep (used by both the train loop and evaluate_model)
# ======================================================================
def build_concat_masks(labels, chosen_ids, num_frames):
    """labels: [B,N,1,H,W], chosen_ids: [B] -> (labels_cat[B,H,W*N], mask_cat[B,H,W*N])."""
    labels_2d = labels.squeeze(2)                       # [B,N,H,W]
    chosen_ids_exp = chosen_ids[:, None, None, None]    # [B,1,1,1]
    binary_masks = (labels_2d == chosen_ids_exp).float()  # [B,N,H,W]
    labels_cat = torch.cat([labels_2d[:, i] for i in range(num_frames)], dim=2)    # [B,H,W*N]
    mask_cat = torch.cat([binary_masks[:, i] for i in range(num_frames)], dim=2)   # [B,H,W*N]
    return labels_cat, mask_cat


def global_to_per_frame_points(sampled_points, sampled_labels, B, frame_width, device):
    """Convert width-concatenated global coords -> per-frame (x_local, y) + frame index."""
    point_coords_list = []
    point_labels_list = []
    point_frame_indices_list = []
    for b in range(B):
        pts = sampled_points[b]            # [K,2]
        x_all, y_all = pts[:, 0], pts[:, 1]
        frame_idx = (x_all // frame_width).long()
        x_local = x_all % frame_width
        point_coords = torch.stack([x_local, y_all], dim=1).to(device)
        point_coords_list.append(point_coords)
        point_labels_list.append(sampled_labels[b])
        point_frame_indices_list.append(frame_idx.to(device))
    return point_coords_list, point_labels_list, point_frame_indices_list


def extract_iou_predictions(outputs, num_samples):
    """Pull a per-sample predicted-IoU vector [B] from model outputs.

    With multimask_output=False the head emits a single score per sample; this
    is robust to either [B] or [B,1] shapes.
    """
    iou_pred = outputs["iou_predictions"]
    return iou_pred.reshape(num_samples, -1)[:, 0].float()


def build_checkpoint(
    cfg,
    *,
    epoch,
    model,
    optimizer,
    scaler,
    scheduler,
    num_frames,
    loss,
    val_iou,
    val_loss,
    val_mask_loss,
    val_dice_loss,
    current_k,
    pose_diverse_ratio,
    single_frame_ratio,
    best_val_loss,
    best_iou,
    no_improve_count,
):
    """Single source of truth for checkpoint dicts (best-val / best-iou / latest)."""
    bare = model.module if hasattr(model, "module") else model
    # Store only what training actually changes. The frozen SAM encoder, prompt
    # encoder and VGGT are ~99.6% of the parameters and are reloaded from their own
    # public checkpoints by build_sam_vggt, so re-serialising them every save costs
    # 7.6 GB per file for nothing. Keyed off requires_grad rather than a module
    # list, so it stays correct if what is frozen ever changes.
    trainable_keys = sorted(n for n, prm in bare.named_parameters() if prm.requires_grad)
    full_state = bare.state_dict()
    model_state_dict = {k: full_state[k] for k in trainable_keys}
    return {
        "epoch": epoch,
        "model_state_format": TRAINABLE_ONLY_FORMAT,
        "trainable_keys": trainable_keys,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_step_applied_for_epoch": False,
        "loss": loss,
        "val_iou": val_iou,
        "val_loss": val_loss,
        "val_mask_loss": val_mask_loss,
        "val_dice_loss": val_dice_loss,
        "num_frames": num_frames,
        "prompt_k_min": cfg.prompt_k_min,
        "prompt_k_max": cfg.prompt_k_max,
        "prompt_k": current_k,
        "prompt_transition_epoch": cfg.prompt_transition_epoch,
        "prompt_single_frame_ratio": float(single_frame_ratio),
        "prompt_multi_frame_ratio": float(1.0 - single_frame_ratio),
        "pose_diverse_ratio": float(pose_diverse_ratio) if pose_diverse_ratio is not None else None,
        "random_scene_ratio": float(1.0 - pose_diverse_ratio) if pose_diverse_ratio is not None else None,
        "best_val_loss": float(best_val_loss),
        "best_iou": float(best_iou),
        "no_improve_count": int(no_improve_count),
        "patience": int(cfg.patience),
    }


# ======================================================================
#  Validation
# ======================================================================
def evaluate_model(
    model,
    val_dataloader,
    cfg,
    device="cuda",
    num_frames=4,
    num_batches=1,
    rank=0,
    world_size=1,
    val_viz_state=None,
):
    """
    Evaluate model on validation set and return metrics.
    When world_size > 1, each rank evaluates its dataloader shard and metrics
    are all_reduced so returned values are global averages.

    Visualization: logs lowest/highest-loss batches plus fixed targets to wandb
    (rank 0 only).
    """
    model.eval()
    amp = cfg.amp
    total_loss = 0.0
    total_mask_loss = 0.0
    total_dice_loss = 0.0
    total_iou = 0.0
    total_iou_pred_loss = 0.0
    num_batches_eval = 0
    if val_viz_state is None:
        val_viz_state = {}
    if "fixed_targets" not in val_viz_state:
        val_viz_state["fixed_targets"] = {
            f"fixed_{i}": {
                "scene_name": scene_name,
                "object_id": int(object_id),
                "slot": None,  # {"batch_idx": int, "batch_item_idx": int}
            }
            for i, (scene_name, object_id) in enumerate(FIXED_VAL_VIZ_TARGETS)
        }
    fixed_targets = val_viz_state["fixed_targets"]
    # Only keep local min/max for global aggregation (avoid storing all batches)
    batch_min_loss, batch_min_idx, batch_min_output = None, None, None
    batch_max_loss, batch_max_idx, batch_max_output = None, None, None
    fixed_viz_data_local = {}  # key -> viz_data discovered on this rank
    val_viz_images = {}  # batches to log (lowest/highest loss plus fixed targets)

    _use_cuda = str(device).startswith("cuda")

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            images = data["images"].to(device)       # [B, N(frames), 3, 1024, 1024]
            labels = data["labels"].to(device)       # [B, N(frames), 1, 1024, 1024]
            valid_ids_list = data["valid_ids"]       # list of B tensors
            sampled_object_ids = data.get("sampled_object_ids", None)

            B, N, _, H, W = images.shape
            assert N == num_frames

            # Instance selection: random by default, then override fixed targets.
            valid_indices = [b for b in range(B) if len(valid_ids_list[b]) > 0]
            if len(valid_indices) == 0:
                continue  # Skip batch if no valid instances
            chosen_ids = []
            for b in valid_indices:
                if sampled_object_ids is not None:
                    chosen_ids.append(
                        resolve_chosen_id_from_sampled_or_random(sampled_object_ids[b], valid_ids_list[b])
                    )
                else:
                    valid_ids = valid_ids_list[b]
                    rid = torch.randint(0, len(valid_ids), (1,))
                    chosen_ids.append(valid_ids[rid])
            chosen_ids = torch.stack(chosen_ids).to(device).squeeze(1)  # [B']

            # Map original batch index -> scene name and valid object-id set.
            scene_by_batch_item = {}
            valid_id_set_by_batch_item = {}
            for b in valid_indices:
                image_paths_b = data["image_paths"][b]
                scene_name_b = os.path.basename(os.path.dirname(os.path.dirname(image_paths_b[0])))
                scene_by_batch_item[b] = scene_name_b
                valid_id_set_by_batch_item[b] = {int(x) for x in valid_ids_list[b].detach().cpu().tolist()}

            # Force fixed scene/object targets and lock their slots on first match.
            for key, target in fixed_targets.items():
                target_scene = target["scene_name"]
                target_obj = int(target["object_id"])
                slot = target.get("slot")
                matched_batch_item_idx = None

                if slot is not None:
                    slot_batch_idx = int(slot.get("batch_idx", -1))
                    slot_item_idx = int(slot.get("batch_item_idx", -1))
                    if (
                        slot_batch_idx == batch_idx
                        and slot_item_idx in valid_indices
                        and scene_by_batch_item.get(slot_item_idx) == target_scene
                        and target_obj in valid_id_set_by_batch_item.get(slot_item_idx, set())
                    ):
                        matched_batch_item_idx = slot_item_idx
                else:
                    for b in valid_indices:
                        if scene_by_batch_item.get(b) == target_scene and target_obj in valid_id_set_by_batch_item.get(b, set()):
                            matched_batch_item_idx = b
                            target["slot"] = {"batch_idx": int(batch_idx), "batch_item_idx": int(b)}
                            break

                if matched_batch_item_idx is not None:
                    filtered_idx = valid_indices.index(matched_batch_item_idx)
                    chosen_ids[filtered_idx] = target_obj

            images = images[valid_indices]
            labels = labels[valid_indices]
            B = len(valid_indices)

            labels_cat, mask_cat = build_concat_masks(labels, chosen_ids, N)
            sampled_points, sampled_labels = sample_points_for_instances_single_frame(
                labels_cat,
                chosen_ids,
                k=3,
                num_frames=N,
                frame_width=W,
            )
            point_coords_list, point_labels_list, point_frame_indices_list = global_to_per_frame_points(
                sampled_points, sampled_labels, B, W, device
            )

            sam_pre = images.to(device)
            sam_feats_precomputed = data.get("sam_embeddings", None)
            if sam_feats_precomputed is not None:
                sam_feats_precomputed = sam_feats_precomputed.to(device=device)

            with torch.amp.autocast("cuda", enabled=amp and _use_cuda):
                outputs = model.forward(
                    sam_pre=sam_pre,
                    sam_feats_precomputed=sam_feats_precomputed,
                    point_coords_list=point_coords_list,
                    point_labels_list=point_labels_list,
                    point_frame_indices_list=point_frame_indices_list,
                    multimask_output=False,
                    visualize=False,
                )

            low_res_masks = outputs["low_res_logits"]    # [B,1,256,256*N]

            # Build target masks for loss
            target_masks = mask_cat.unsqueeze(1).to(device)   # [B,1,H,W*N]
            pred_h, pred_w = low_res_masks.shape[2], low_res_masks.shape[3]
            target_masks_low = F.interpolate(
                target_masks.float(),
                size=(pred_h, pred_w),
                mode="nearest",
            )  # [B,1,256,256*N]

            pred_masks = low_res_masks[:, 0:1]

            # Positive weight from target statistics for class balancing
            target_pos_ratio = target_masks_low.sum().item() / target_masks_low.numel()
            if target_pos_ratio > 0:
                pos_weight = min((1.0 - target_pos_ratio) / target_pos_ratio, cfg.pos_weight_cap)
            else:
                pos_weight = 10.0

            val_B = pred_masks.shape[0]
            loss_mask_per_sample, loss_dice_per_sample = loss_masks(
                pred_masks.float(),
                target_masks_low.float(),
                num_masks=float(val_B),
                pos_weight=pos_weight,
                use_focal_loss=True,
                dice_on_full_mask=True,
            )
            loss_mask = loss_mask_per_sample.mean() * cfg.mask_loss_weight
            loss_dice = loss_dice_per_sample.mean()
            loss = loss_mask + loss_dice

            # IoU: pool per sample (over frames), then average over samples in batch
            pred_probs = torch.sigmoid(pred_masks)
            pred_binary = (pred_probs > 0.5).float()
            target_binary = target_masks_low
            intersection_per_sample = (pred_binary * target_binary).flatten(2).sum(dim=2).squeeze(-1)
            union_per_sample = (pred_binary + target_binary).clamp(0, 1).flatten(2).sum(dim=2).squeeze(-1)
            iou_per_sample = intersection_per_sample / (union_per_sample + 1e-8)  # [B]
            iou = iou_per_sample.mean().item()

            # Predicted-IoU regression metric (reported, not folded into val_loss)
            iou_pred = extract_iou_predictions(outputs, val_B)
            loss_iou_pred = predicted_iou_loss(iou_pred, iou_per_sample.detach())

            total_loss += loss.item()
            total_mask_loss += loss_mask.item()
            total_dice_loss += loss_dice.item()
            total_iou += iou
            total_iou_pred_loss += loss_iou_pred.item()
            num_batches_eval += 1

            def _make_viz_data(filtered_idx: int):
                batch_item_idx_local = valid_indices[filtered_idx]
                image_paths_viz_local = data["image_paths"][batch_item_idx_local]  # list of N paths
                scene_name_viz_local = os.path.basename(os.path.dirname(os.path.dirname(image_paths_viz_local[0])))
                frame_ids_viz_local = [os.path.splitext(os.path.basename(p))[0] for p in image_paths_viz_local]
                return (
                    low_res_masks[filtered_idx : filtered_idx + 1].detach().cpu().clone(),
                    images[filtered_idx].cpu().clone(),           # [N,3,H,W]
                    mask_cat[filtered_idx].cpu().clone(),         # [H,W*N]
                    point_coords_list[filtered_idx].cpu().clone(),
                    sampled_labels[filtered_idx].cpu().clone(),
                    point_frame_indices_list[filtered_idx].cpu().clone(),
                    H, W, N,
                    float(loss_mask_per_sample[filtered_idx].item()),
                    float(loss_dice_per_sample[filtered_idx].item()),
                    float(iou_per_sample[filtered_idx].item()),
                    int(chosen_ids[filtered_idx].item()),
                    scene_name_viz_local,
                    frame_ids_viz_local,
                )

            # Use first valid sample in this batch for lowest/highest-loss tracking.
            viz_data = _make_viz_data(0)
            loss_val = loss.item()
            if batch_min_loss is None or loss_val < batch_min_loss:
                batch_min_loss, batch_min_idx, batch_min_output = loss_val, batch_idx, viz_data
            if batch_max_loss is None or loss_val > batch_max_loss:
                batch_max_loss, batch_max_idx, batch_max_output = loss_val, batch_idx, viz_data
            # Track fixed scene/object viz data on every rank.
            for key, target in fixed_targets.items():
                slot = target.get("slot")
                if slot is None or int(slot.get("batch_idx", -1)) != batch_idx:
                    continue
                batch_item_idx_slot = int(slot.get("batch_item_idx", -1))
                if batch_item_idx_slot not in valid_indices:
                    continue
                fixed_filtered_idx = valid_indices.index(batch_item_idx_slot)
                fixed_viz_data_local[key] = _make_viz_data(fixed_filtered_idx)

    # Lowest/highest-loss visualizations (rank-0-local). The cross-rank
    # dist.send/recv transfer was removed: its NCCL point-to-point comm bootstrap
    # deadlocked during validation at multi-GPU scale (one rank's send had no
    # matching recv -> store timeout -> the rest hung on the next collective).
    # Examples are now drawn from rank 0's own validation shard; the metric
    # all_reduce below still aggregates across all ranks.
    if rank == 0 and (batch_min_output is not None or batch_max_output is not None):
        if batch_min_output is not None:
            val_viz_images["lowest_loss"] = viz_and_pack(batch_min_output)
        if batch_max_output is not None:
            if batch_max_output is not batch_min_output:
                val_viz_images["highest_loss"] = viz_and_pack(batch_max_output)
            else:
                val_viz_images["highest_loss"] = val_viz_images.get("lowest_loss")

    # Fixed scene/object visualizations. Each fixed target lands on whichever
    # rank's validation shard happens to hold that scene/object, so a single rank
    # (e.g. rank 0) only ever sees a subset. Render locally on every rank, then
    # gather the rendered images onto rank 0 with all_gather_object -- a symmetric
    # collective that, unlike the previously-removed point-to-point send/recv,
    # cannot deadlock (every rank issues exactly one matching call). Rendering
    # before the gather keeps the payload small PIL images instead of raw tensors.
    fixed_viz_images_local = {
        key: viz_and_pack(vd)
        for key, vd in fixed_viz_data_local.items()
        if vd is not None
    }
    if world_size > 1 and dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, fixed_viz_images_local)
    else:
        gathered = [fixed_viz_images_local]

    # Build val_viz_images dict: merge all ranks' fixed targets (rank 0 only).
    if rank == 0:
        merged = {}
        for d in gathered:
            if not d:
                continue
            for key, imgs in d.items():
                if imgs is not None:
                    merged.setdefault(key, imgs)
        for key in sorted(fixed_targets.keys()):
            val_viz_images[key] = merged.get(key)

    # Reduce metrics across ranks when distributed
    if world_size > 1 and dist.is_initialized():
        sums_t = torch.tensor(
            [total_loss, total_mask_loss, total_dice_loss, total_iou, total_iou_pred_loss],
            device=device,
            dtype=torch.float64,
        )
        count_t = torch.tensor([num_batches_eval], device=device, dtype=torch.float64)
        dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        n = count_t.item()
        if n > 0:
            total_loss, total_mask_loss, total_dice_loss, total_iou, total_iou_pred_loss = sums_t.tolist()
            num_batches_eval = int(n)
        else:
            num_batches_eval = 0

    model.train()  # Switch back to training mode

    denom = num_batches_eval if num_batches_eval > 0 else 1
    return {
        "val_loss": total_loss / denom if num_batches_eval > 0 else 0.0,
        "val_mask_loss": total_mask_loss / denom if num_batches_eval > 0 else 0.0,
        "val_dice_loss": total_dice_loss / denom if num_batches_eval > 0 else 0.0,
        "val_iou": total_iou / denom if num_batches_eval > 0 else 0.0,
        "val_iou_pred_loss": total_iou_pred_loss / denom if num_batches_eval > 0 else 0.0,
        "val_viz_images": val_viz_images,
    }


# ======================================================================
#  Training loop
# ======================================================================
def train_sam_vggt(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    cfg,
    device,
    num_frames,
    rank=0,
    world_size=1,
    is_distributed=False,
    resume_path=None,
    finetune_from=None,
):
    epochs = cfg.epochs
    amp = cfg.amp
    patience = cfg.patience

    best_val_loss_path = os.path.join(cfg.output_dir, f"sam_vggt_best_val_loss_{cfg.ckpt_tag}.pth")
    best_iou_path = os.path.join(cfg.output_dir, f"sam_vggt_best_iou_{cfg.ckpt_tag}.pth")
    latest_path = os.path.join(cfg.output_dir, f"sam_vggt_latest_{cfg.ckpt_tag}.pth")

    # Only initialize wandb on rank 0
    if rank == 0:
        run_name = cfg.wandb_run_name or f"ddp-{world_size}gpus-{num_frames}frames-{cfg.ckpt_tag}"
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            config={
                "mode": cfg.mode,
                "epochs": epochs,
                "batch_size": cfg.batch_size,
                "number batches": len(train_dataloader),
                "num_frames": num_frames,
                "amp": amp,
                "world_size": world_size,
                "distributed": is_distributed,
                "datasets": cfg.datasets,
                "prompt_k_min": cfg.prompt_k_min,
                "prompt_k_max": cfg.prompt_k_max,
                "prompt_transition_epoch": cfg.prompt_transition_epoch,
                "iou_loss_weight": cfg.iou_loss_weight,
                "loss function": "focal loss + full mask dice loss + iou-pred mse",
            },
            name=run_name,
        )
        # Plot per-epoch metrics (validation scalars + media, epoch summaries)
        # against an explicit "epoch" x-axis so each validation key's media forms
        # one frame-per-epoch series that lines up across epochs. This only sets
        # the display x-axis; we never pass step= to wandb.log (that would clash
        # with wandb's monotonic internal step driven by per-batch training logs),
        # so logged values, the internal step counter, and all per-batch training
        # curves are unchanged.
        wandb.define_metric("epoch")
        wandb.define_metric("val_*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("epoch_loss/*", step_metric="epoch")
        os.makedirs(cfg.output_dir, exist_ok=True)

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_loss = float("inf")
    best_val_loss = float("inf")
    best_iou = 0.0
    no_improve_count = 0
    start_epoch = 0
    val_viz_state = {}  # Persists fixed batch indices and chosen_ids across validation runs

    # ------------------------------------------------------------------
    # Finetune: load model weights only (no optimizer / scheduler / epoch)
    # ------------------------------------------------------------------
    if finetune_from is not None:
        if not os.path.isfile(finetune_from):
            raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_from}")
        if rank == 0:
            print(f"Loading model weights for finetuning from: {finetune_from}")
        try:
            ckpt = torch.load(finetune_from, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(finetune_from, map_location=device)

        model_to_load = model.module if hasattr(model, "module") else model
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint missing model_state_dict: {finetune_from}")
        if is_partial_checkpoint(ckpt):
            # Trained modules only -- either a released checkpoint or one this
            # trainer wrote. The frozen SAM and VGGT weights are already in place
            # from build_sam_vggt, so this must load non-strictly;
            # load_partial_checkpoint verifies it did so fully.
            load_partial_checkpoint(model_to_load, ckpt)
            kind = ("slim release checkpoint" if is_slim_checkpoint(ckpt)
                    else "trainable-only training checkpoint")
        else:
            model_to_load.load_state_dict(ckpt["model_state_dict"], strict=True)
            kind = "full training checkpoint"
        if rank == 0:
            print(
                f"Loaded model weights from {finetune_from} ({kind}, "
                f"original epoch={ckpt.get('epoch', '?')}). Starting fresh from epoch 0."
            )

    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if rank == 0:
            print(f"Loading resume checkpoint from: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)

        model_to_load = model.module if hasattr(model, "module") else model
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint missing model_state_dict: {resume_path}")
        if is_slim_checkpoint(ckpt):
            raise SystemExit(
                f"{resume_path} is a slim release checkpoint. It carries only the "
                "trained weights, with no optimizer, scaler, scheduler or epoch "
                "state, so a run cannot be resumed from it. Use --finetune_from to "
                "start a fresh run from these weights, or resume from a full "
                "training checkpoint."
            )
        if is_partial_checkpoint(ckpt):
            load_partial_checkpoint(model_to_load, ckpt)
        else:
            model_to_load.load_state_dict(ckpt["model_state_dict"], strict=True)

        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            # Legacy checkpoints are saved before epoch-end scheduler.step().
            if not ckpt.get("scheduler_step_applied_for_epoch", False):
                scheduler.step()

        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        best_iou = float(ckpt.get("best_iou", ckpt.get("val_iou", 0.0)))
        no_improve_count = 0  # New patience on resume

        if rank == 0:
            print(
                f"Resumed at epoch={start_epoch} with best_val_loss={best_val_loss:.4f}, "
                f"best_iou={best_iou:.4f}, patience={patience}"
            )

    if is_distributed:
        dist.barrier()

    if start_epoch >= epochs:
        if rank == 0:
            print(f"Resume epoch ({start_epoch}) is >= total epochs ({epochs}); nothing to train.")
            wandb.finish()
        return

    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataloader.batch_sampler, "set_epoch"):
            train_dataloader.batch_sampler.set_epoch(epoch)
        pose_diverse_ratio = None
        if hasattr(train_dataloader.batch_sampler, "_get_pose_diverse_probability"):
            pose_diverse_ratio = train_dataloader.batch_sampler._get_pose_diverse_probability()
        # Prompts come from a single frame on the chosen mask unless the stage-1
        # curriculum is on (cfg.prompt_transition_epoch); the point count is drawn
        # fresh per batch in [prompt_k_min, prompt_k_max].
        single_frame_ratio = single_frame_prompt_ratio(cfg, epoch)
        current_k = cfg.prompt_k_min  # last value used; refreshed each batch below

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting Epoch {epoch+1}/{epochs}")
            if pose_diverse_ratio is not None:
                print(
                    f"Frame sampling mix: random_scene={1.0 - pose_diverse_ratio:.3f}, "
                    f"pose_diverse={pose_diverse_ratio:.3f}"
                )
            if single_frame_ratio >= 1.0:
                print("Prompt sampling: single_frame (all positive)")
            else:
                print(
                    f"Prompt sampling mix: single_frame={single_frame_ratio:.3f}, "
                    f"multi_frame={1.0 - single_frame_ratio:.3f} "
                    f"(curriculum -> epoch {cfg.prompt_transition_epoch})"
                )
            print(f"Prompt points: k ~ uniform[{cfg.prompt_k_min}, {cfg.prompt_k_max}] per batch")
            print(f"{'='*60}")
        running_loss = 0
        running_mask_loss = 0
        running_dice_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        should_capture_train_misseg = (epoch + 1) > 40 and ((epoch + 1) % 5 == 0)
        should_log_periodic_train_viz = ((epoch + 1) % 50 == 0)
        periodic_train_viz_logged = False
        misseg_examples_saved = 0
        misseg_save_dir = None
        if rank == 0 and should_capture_train_misseg:
            misseg_save_dir = os.path.join(cfg.output_dir, "train_misseg_viz", f"epoch_{epoch + 1:03d}")
            os.makedirs(misseg_save_dir, exist_ok=True)

        for batch_idx, data in enumerate(train_dataloader):
            images = data["images"].to(device)       # [B, N(frames), 3, 1024, 1024]
            labels = data["labels"].to(device)       # [B, N(frames), 1, 1024, 1024]
            valid_ids_list = data["valid_ids"]
            sampled_object_ids = data.get("sampled_object_ids", None)

            B, N, _, H, W = images.shape
            assert N == num_frames

            # ------------------------------------------------------------------
            #  Random instance selection PER GROUP using offline valid-IDs
            # ------------------------------------------------------------------
            if sampled_object_ids is not None:
                chosen_ids = torch.stack(
                    [
                        resolve_chosen_id_from_sampled_or_random(sampled_object_ids[b], valid_ids_list[b])
                        for b in range(B)
                    ],
                    dim=0,
                ).to(device).squeeze(1)  # [B]
            else:
                chosen_ids = []
                for b in range(B):
                    valid_ids = valid_ids_list[b]
                    if len(valid_ids) == 0:
                        continue
                    rid = torch.randint(0, len(valid_ids), (1,))
                    chosen_ids.append(valid_ids[rid])
                chosen_ids = torch.stack(chosen_ids).to(device).squeeze(1)  # [B]

            # ------------------------------------------------------------------
            #  Build masks + sample prompt points, to per-frame coords.
            #  k is drawn fresh per batch. Points are all positive from a single
            #  frame on the chosen mask (same routine as eval) unless the stage-1
            #  curriculum is active, which mixes in multi-frame pos/neg prompts.
            # ------------------------------------------------------------------
            labels_cat, mask_cat = build_concat_masks(labels, chosen_ids, N)
            current_k = sample_prompt_k(cfg)
            sampled_points, sampled_labels = sample_training_prompt_points(
                cfg,
                labels_cat,
                chosen_ids,
                k=current_k,
                num_frames=N,
                frame_width=W,
                single_frame_ratio=single_frame_ratio,
            )
            point_coords_list, point_labels_list, point_frame_indices_list = global_to_per_frame_points(
                sampled_points, sampled_labels, B, W, device
            )

            sam_pre = images.to(device)
            sam_feats_precomputed = data.get("sam_embeddings", None)
            if sam_feats_precomputed is not None:
                sam_feats_precomputed = sam_feats_precomputed.to(device=device)

            # ------------------------------------------------------------------
            #  Forward pass
            # ------------------------------------------------------------------
            try:
                with torch.amp.autocast("cuda", enabled=amp):
                    outputs = model.forward(
                        sam_pre=sam_pre,
                        sam_feats_precomputed=sam_feats_precomputed,
                        point_coords_list=point_coords_list,
                        point_labels_list=point_labels_list,
                        point_frame_indices_list=point_frame_indices_list,
                        multimask_output=False,
                        visualize=False,
                    )
            except torch.cuda.OutOfMemoryError as e:
                print(f"[Rank {rank}] OOM Error during forward pass at batch {batch_idx}, epoch {epoch+1}", flush=True)
                print(f"[Rank {rank}] Error: {e}", flush=True)
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"[Rank {rank}] GPU memory: Allocated={memory_allocated:.2f} GB, Reserved={memory_reserved:.2f} GB", flush=True)
                torch.cuda.empty_cache()
                if is_distributed:
                    try:
                        dist.barrier()
                    except Exception:
                        pass
                raise
            low_res_masks = outputs["low_res_logits"]    # [B,1,256,256*N]

            # ------------------------------------------------------------------
            #  Build target masks for loss
            # ------------------------------------------------------------------
            target_masks = mask_cat.unsqueeze(1).to(device)   # [B,1,H,W*N]
            pred_h, pred_w = low_res_masks.shape[2], low_res_masks.shape[3]
            target_masks_low = F.interpolate(
                target_masks.float(),
                size=(pred_h, pred_w),
                mode="nearest",
            )  # [B,1,256,256*N]
            pred_masks = low_res_masks[:, 0:1]

            # ------------------------------------------------------------------
            #  Loss
            # ------------------------------------------------------------------
            target_pos_ratio = target_masks_low.sum().item() / target_masks_low.numel()
            if target_pos_ratio > 0:
                pos_weight = min((1.0 - target_pos_ratio) / target_pos_ratio, cfg.pos_weight_cap)
            else:
                pos_weight = 10.0

            loss_mask_per_sample, loss_dice_per_sample = loss_masks(
                pred_masks.float(),
                target_masks_low.float(),
                num_masks=float(B),
                pos_weight=pos_weight,
                use_focal_loss=True,
                dice_on_full_mask=True,
            )
            loss_mask = loss_mask_per_sample.mean() * cfg.mask_loss_weight
            loss_dice = loss_dice_per_sample.mean()

            # Diagnostics + true IoU per sample (also the regression target below)
            pred_probs = torch.sigmoid(pred_masks)
            pred_mean = pred_probs.mean().item()
            target_mean = target_masks_low.mean().item()
            pred_logits_mean = pred_masks.mean().item()
            pred_logits_std = pred_masks.std().item()
            pred_binary = (pred_probs > 0.5).float()
            target_binary = target_masks_low
            intersection_per_sample = (pred_binary * target_binary).flatten(2).sum(dim=2).squeeze(-1)
            union_per_sample = (pred_binary + target_binary).clamp(0, 1).flatten(2).sum(dim=2).squeeze(-1)
            iou_per_sample = intersection_per_sample / (union_per_sample + 1e-8)  # [B]
            pred_max = pred_probs.max().item()
            pred_min = pred_probs.min().item()

            # Predicted-IoU regression loss (trains the decoder's IoU head)
            iou_pred = extract_iou_predictions(outputs, B)
            loss_iou = predicted_iou_loss(iou_pred, iou_per_sample.detach())

            loss = loss_mask + loss_dice + cfg.iou_loss_weight * loss_iou

            # Periodic train visualization (every 50 epochs).
            if rank == 0 and should_log_periodic_train_viz and (not periodic_train_viz_logged) and B > 0:
                sample_idx = 0
                image_paths_this = data["image_paths"][sample_idx]
                scene_name_this = os.path.basename(os.path.dirname(os.path.dirname(image_paths_this[0])))
                frame_ids_this = [os.path.splitext(os.path.basename(p))[0] for p in image_paths_this]
                gt_img = visualize_gt_prompt_overlay(
                    images[sample_idx].detach().cpu(),
                    mask_cat[sample_idx].detach().cpu(),
                    point_coords_list[sample_idx].detach().cpu(),
                    sampled_labels[sample_idx].detach().cpu(),
                    point_frame_indices_list[sample_idx].detach().cpu(),
                    N=N, W=W, scene_name=scene_name_this, frame_ids=frame_ids_this,
                    chosen_object_id=int(chosen_ids[sample_idx].item()),
                    pooled_iou=float(iou_per_sample[sample_idx].item()),
                )
                pred_img = visualize_pred_mask_overlay(
                    images[sample_idx].detach().cpu(),
                    low_res_masks[sample_idx : sample_idx + 1].detach().cpu(),
                    point_coords_list[sample_idx].detach().cpu(),
                    sampled_labels[sample_idx].detach().cpu(),
                    point_frame_indices_list[sample_idx].detach().cpu(),
                    N=N, W=W,
                    loss_mask=float(loss_mask_per_sample[sample_idx].item()),
                    loss_dice=float(loss_dice_per_sample[sample_idx].item()),
                    scene_name=scene_name_this, frame_ids=frame_ids_this,
                    chosen_object_id=int(chosen_ids[sample_idx].item()),
                    pooled_iou=float(iou_per_sample[sample_idx].item()),
                )
                wandb.log({
                    "train_periodic/sample_gt_prompt": wandb.Image(gt_img),
                    "train_periodic/sample_pred_mask": wandb.Image(pred_img),
                    "train_periodic/epoch": epoch + 1,
                    "step": batch_idx + epoch * len(train_dataloader),
                })
                periodic_train_viz_logged = True

            pred_target_diff = abs(pred_mean - target_mean)
            pred_target_ratio = pred_mean / (target_mean + 1e-8)

            if rank == 0 and should_capture_train_misseg and misseg_examples_saved < 3:
                # Save samples where >= 50% of positive prompt points are uncovered.
                pred_full_probs = torch.sigmoid(
                    F.interpolate(pred_masks.detach(), size=(H, W * N), mode="bilinear", align_corners=False)
                )
                pred_full_binary = pred_full_probs > 0.5  # [B,1,H,W*N]
                misseg_indices = []
                for sample_idx in range(B):
                    point_coords_sample = point_coords_list[sample_idx].detach().long()
                    point_frames_sample = point_frame_indices_list[sample_idx].detach().long()
                    point_labels_sample = sampled_labels[sample_idx].detach().long()
                    use_idx = point_labels_sample == 1
                    if int(use_idx.sum().item()) == 0:
                        use_idx = torch.ones_like(point_labels_sample, dtype=torch.bool)
                    if int(use_idx.sum().item()) == 0:
                        continue
                    x_local = point_coords_sample[:, 0]
                    y = point_coords_sample[:, 1]
                    x_global = (x_local + point_frames_sample * W).clamp(0, W * N - 1)
                    y = y.clamp(0, H - 1)
                    x_eval = x_global[use_idx]
                    y_eval = y[use_idx]
                    covered = pred_full_binary[sample_idx, 0, y_eval, x_eval]
                    uncovered_ratio = 1.0 - float(covered.float().mean().item())
                    if uncovered_ratio >= 0.5:
                        misseg_indices.append(sample_idx)

                misseg_wandb_log = {}
                for sample_idx in misseg_indices:
                    if misseg_examples_saved >= 3:
                        break
                    image_paths_this = data["image_paths"][sample_idx]
                    scene_name_this = os.path.basename(os.path.dirname(os.path.dirname(image_paths_this[0])))
                    frame_ids_this = [os.path.splitext(os.path.basename(p))[0] for p in image_paths_this]
                    viz_result = _save_train_mismatch_visualization(
                        save_dir=misseg_save_dir,
                        epoch_idx=epoch,
                        batch_idx=batch_idx,
                        sample_idx=sample_idx,
                        scene_name=scene_name_this,
                        frame_ids=frame_ids_this,
                        images_b=images[sample_idx].detach().cpu(),
                        mask_cat_b=mask_cat[sample_idx].detach().cpu(),
                        low_res_logits_b=low_res_masks[sample_idx : sample_idx + 1].detach().cpu(),
                        point_coords_b=point_coords_list[sample_idx].detach().cpu(),
                        point_labels_b=sampled_labels[sample_idx].detach().cpu(),
                        point_frame_indices_b=point_frame_indices_list[sample_idx].detach().cpu(),
                        num_frames=N,
                        frame_width=W,
                        chosen_object_id=int(chosen_ids[sample_idx].item()),
                        loss_mask_value=float(loss_mask_per_sample[sample_idx].item()),
                        loss_dice_value=float(loss_dice_per_sample[sample_idx].item()),
                        pooled_iou_value=float(iou_per_sample[sample_idx].item()),
                    )
                    misseg_examples_saved += 1
                    example_key = f"example_{misseg_examples_saved}"
                    misseg_wandb_log[f"train_misseg/{example_key}_gt_prompt"] = wandb.Image(viz_result["gt_img"])
                    misseg_wandb_log[f"train_misseg/{example_key}_pred_mask"] = wandb.Image(viz_result["pred_img"])
                if misseg_wandb_log:
                    misseg_wandb_log["step"] = batch_idx + epoch * len(train_dataloader)
                    wandb.log(misseg_wandb_log)

            if rank == 0:
                iou = iou_per_sample.mean().item()
                wandb.log({
                    "loss/total": loss.item(),
                    "loss/mask": loss_mask.item(),
                    "loss/dice": loss_dice.item(),
                    "loss/iou_pred": loss_iou.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "pred_mean": pred_mean,
                    "target_mean": target_mean,
                    "pred_target_diff": pred_target_diff,
                    "pred_target_ratio": pred_target_ratio,
                    "pred_logits_mean": pred_logits_mean,
                    "pred_logits_std": pred_logits_std,
                    "pred_max": pred_max,
                    "pred_min": pred_min,
                    "pos_weight": pos_weight,
                    "target_pos_ratio": target_pos_ratio,
                    "prompt_k": current_k,
                    "iou": iou,
                    "intersection": intersection_per_sample.sum().item(),
                    "union": union_per_sample.sum().item(),
                    "step": batch_idx + epoch * len(train_dataloader),
                })

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping; skip step on NaN/Inf grads (synchronized across ranks).
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            invalid_grad_local = bool(torch.isnan(grad_norm) or torch.isinf(grad_norm))
            invalid_grad_global = invalid_grad_local
            if is_distributed:
                invalid_flag = torch.tensor([1 if invalid_grad_local else 0], device=device, dtype=torch.int64)
                dist.all_reduce(invalid_flag, op=dist.ReduceOp.MAX)
                invalid_grad_global = bool(invalid_flag.item())

            if invalid_grad_global:
                if rank == 0:
                    print(
                        f"Warning: skipping optimizer step due to invalid grad norm "
                        f"(local_rank0_grad_norm={float(grad_norm):.6f})."
                    )
            else:
                scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_mask_loss += loss_mask.item()
            running_dice_loss += loss_dice.item()
            num_batches += 1

            if rank == 0 and ((batch_idx + 1) % 10 == 0 or batch_idx == 0):
                elapsed = time.time() - epoch_start_time
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                total_batches = len(train_dataloader)
                estimated_remaining = (total_batches - batch_idx - 1) / batches_per_sec if batches_per_sec > 0 else 0
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} ({100*(batch_idx+1)/total_batches:.1f}%): "
                    f"Loss={running_loss/num_batches:.4f}, "
                    f"Mask={running_mask_loss/num_batches:.4f}, "
                    f"Dice={running_dice_loss/num_batches:.4f} | "
                    f"Speed: {batches_per_sec:.2f} batches/s | "
                    f"ETA: {estimated_remaining/60:.1f} min"
                )

        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            print(
                f"\nCompleted epoch {epoch+1}/{epochs} in {epoch_time/60:.2f} minutes: "
                f"Loss={running_loss/num_batches:.4f}, "
                f"Mask={running_mask_loss/num_batches:.4f}, "
                f"Dice={running_dice_loss/num_batches:.4f}"
            )
            wandb.log({
                "epoch_loss/total": running_loss / num_batches,
                "epoch_loss/mask": running_mask_loss / num_batches,
                "epoch_loss/dice": running_dice_loss / num_batches,
                "epoch": epoch + 1,
            })
            if epoch == 0 or running_loss / num_batches < best_loss:
                best_loss = running_loss / num_batches

        current_val_loss = None
        current_iou = None
        current_val_mask_loss = None
        current_val_dice_loss = None
        should_stop_early = False

        # Evaluation (all ranks run; rank 0 logs and saves)
        if val_dataloader is not None and (epoch + 1) % cfg.eval_every_n_epochs == 0:
            if rank == 0:
                print("Running evaluation on validation set...")
            val_metrics = evaluate_model(
                model=model,
                val_dataloader=val_dataloader,
                cfg=cfg,
                device=device,
                num_frames=num_frames,
                num_batches=cfg.num_val_batches,
                rank=rank,
                world_size=world_size,
                val_viz_state=val_viz_state,
            )
            if rank == 0:
                print(
                    f"Validation - Loss={val_metrics['val_loss']:.4f}, "
                    f"Mask={val_metrics['val_mask_loss']:.4f}, "
                    f"Dice={val_metrics['val_dice_loss']:.4f}, "
                    f"IoU={val_metrics['val_iou']:.4f}, "
                    f"IoUPredLoss={val_metrics['val_iou_pred_loss']:.4f}"
                )
                wandb_log_dict = {
                    "epoch": epoch + 1,  # explicit x-axis for val_* / validation/* (see define_metric)
                    "val_loss": val_metrics["val_loss"],
                    "val_mask_loss": val_metrics["val_mask_loss"],
                    "val_dice_loss": val_metrics["val_dice_loss"],
                    "val_iou": val_metrics["val_iou"],
                    "val_iou_pred_loss": val_metrics["val_iou_pred_loss"],
                }
                viz_imgs = val_metrics.get("val_viz_images", {})
                fixed_viz_keys = tuple(f"fixed_{i}" for i in range(len(FIXED_VAL_VIZ_TARGETS)))
                for key in ("lowest_loss", "highest_loss", *fixed_viz_keys):
                    if viz_imgs.get(key):
                        imgs = viz_imgs[key]
                        if imgs.get("gt_prompt") is not None:
                            wandb_log_dict[f"validation/{key}_gt_prompt"] = wandb.Image(imgs["gt_prompt"])
                        if imgs.get("pred_mask") is not None:
                            wandb_log_dict[f"validation/{key}_pred_mask"] = wandb.Image(imgs["pred_mask"])
                wandb.log(wandb_log_dict)

            current_val_loss = val_metrics["val_loss"]
            current_iou = val_metrics["val_iou"]
            current_val_mask_loss = val_metrics["val_mask_loss"]
            current_val_dice_loss = val_metrics["val_dice_loss"]

            # Best val loss (lower better)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                if rank == 0:
                    os.makedirs(cfg.output_dir, exist_ok=True)
                    torch.save(
                        build_checkpoint(
                            cfg, epoch=epoch + 1, model=model, optimizer=optimizer, scaler=scaler,
                            scheduler=scheduler, num_frames=num_frames,
                            loss=running_loss / num_batches if num_batches > 0 else 0.0,
                            val_iou=current_iou, val_loss=current_val_loss,
                            val_mask_loss=current_val_mask_loss, val_dice_loss=current_val_dice_loss,
                            current_k=current_k,
                            pose_diverse_ratio=pose_diverse_ratio, single_frame_ratio=single_frame_ratio,
                            best_val_loss=best_val_loss,
                            best_iou=best_iou, no_improve_count=no_improve_count,
                        ),
                        best_val_loss_path,
                    )
                    print(f"Best val loss improved to {best_val_loss:.4f} at epoch {epoch + 1}, saved to {best_val_loss_path}")
            # Best IoU (higher better) - drives early stopping
            if current_iou > best_iou:
                best_iou = current_iou
                no_improve_count = 0
                if rank == 0:
                    os.makedirs(cfg.output_dir, exist_ok=True)
                    torch.save(
                        build_checkpoint(
                            cfg, epoch=epoch + 1, model=model, optimizer=optimizer, scaler=scaler,
                            scheduler=scheduler, num_frames=num_frames,
                            loss=running_loss / num_batches if num_batches > 0 else 0.0,
                            val_iou=current_iou, val_loss=current_val_loss,
                            val_mask_loss=current_val_mask_loss, val_dice_loss=current_val_dice_loss,
                            current_k=current_k,
                            pose_diverse_ratio=pose_diverse_ratio, single_frame_ratio=single_frame_ratio,
                            best_val_loss=best_val_loss,
                            best_iou=best_iou, no_improve_count=no_improve_count,
                        ),
                        best_iou_path,
                    )
                    print(f"Best IoU improved to {best_iou:.4f} at epoch {epoch + 1}, saved to {best_iou_path}")
            else:
                no_improve_count += 1
                if rank == 0:
                    print(f"IoU did not improve. No improvement count: {no_improve_count}/{patience}")
                if no_improve_count >= patience:
                    if rank == 0:
                        print(f"Early stopping triggered after {no_improve_count} evaluations without IoU improvement.")
                        print(f"Best IoU was {best_iou:.4f}")
                    should_stop_early = True
            if is_distributed:
                dist.barrier()

        if rank == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            checkpoint = build_checkpoint(
                cfg, epoch=epoch + 1, model=model, optimizer=optimizer, scaler=scaler,
                scheduler=scheduler, num_frames=num_frames,
                loss=running_loss / num_batches if num_batches > 0 else 0.0,
                val_iou=current_iou, val_loss=current_val_loss,
                val_mask_loss=current_val_mask_loss, val_dice_loss=current_val_dice_loss,
                current_k=current_k,
                pose_diverse_ratio=pose_diverse_ratio, single_frame_ratio=single_frame_ratio,
                best_val_loss=best_val_loss,
                best_iou=best_iou, no_improve_count=no_improve_count,
            )
            torch.save(checkpoint, latest_path)
            # Periodic snapshot every save_every_n_epochs epochs (-1 => never).
            if cfg.save_every_n_epochs > 0 and (epoch + 1) % cfg.save_every_n_epochs == 0:
                epoch_path = os.path.join(
                    cfg.output_dir, f"sam_vggt_epoch{epoch + 1:04d}_{cfg.ckpt_tag}.pth"
                )
                torch.save(checkpoint, epoch_path)
                print(f"Saved periodic checkpoint at epoch {epoch + 1} to {epoch_path}")

        if should_stop_early:
            break

        if scheduler is not None:
            scheduler.step()

    if rank == 0:
        wandb.finish()


# ======================================================================
#  Setup helpers
# ======================================================================
def setup_distributed():
    """Initialize DDP from env vars (torchrun) or fall back to single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Reduce allocator fragmentation (helps with reserved-but-free OOMs).
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if world_size >= 8:
            # torchrun sets LOCAL_WORLD_SIZE = ranks on THIS node; if the global
            # world is larger we're multi-node and NCCL must use a real NIC, not
            # loopback. setdefault everywhere so the launcher's -e env always wins.
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
            is_multinode = world_size > local_world_size
            os.environ.setdefault("NCCL_P2P_DISABLE", "0")
            os.environ.setdefault("NCCL_SHM_DISABLE", "0")
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            os.environ.setdefault("NCCL_NET_GDR_LEVEL", "0")
            if is_multinode:
                # Multi-node: NCCL binds the inter-node NIC supplied by the launcher
                # (NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME via -e). Do NOT force "lo".
                os.environ.setdefault("NCCL_IB_DISABLE", "1")  # ethernet NICs, no IB
            else:
                # Single-node 8-GPU: loopback + SHM/P2P, no NIC needed (unchanged).
                os.environ["NCCL_IB_DISABLE"] = "1"
                os.environ["NCCL_IB_HCA"] = ""
                os.environ["NCCL_SOCKET_IFNAME"] = "lo"
            timeout = timedelta(minutes=30)
            if rank == 0:
                mode = "multi-node" if is_multinode else "single-node SHM/P2P"
                print(
                    f"[NCCL Config] {mode}; "
                    f"IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '(default)')}",
                    flush=True,
                )
        else:
            timeout = timedelta(minutes=10)

        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        print(f"Initialized DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}")
        return rank, local_rank, world_size, True, device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running in single GPU mode: device={device}")
    return 0, 0, 1, False, device


# Dataset type -> (train_ds, val_ds) builder. Both split folders are the split
# itself (no scene_list needed); roots are <root>/<train_subdir|val_subdir>.
def _build_hypersim(train_root, val_root, transform, emb):
    return (
        MultiSceneImageDataset(root_dir=train_root, transform=transform, sam_embedding_subdir=emb),
        MultiSceneImageDataset(root_dir=val_root, transform=transform, sam_embedding_subdir=emb),
    )


def _build_scannetpp(train_root, val_root, transform, emb):
    return (
        ScanNetPPMultiSceneImageDataset(root_dir=train_root, transform=transform, sam_embedding_subdir=emb),
        ScanNetPPMultiSceneImageDataset(root_dir=val_root, transform=transform, sam_embedding_subdir=emb),
    )


DATASET_BUILDERS = {
    "hypersim": _build_hypersim,
    "scannetpp": _build_scannetpp,
}


def build_datasets(cfg, rank):
    transform = transforms.Compose([Resize(size=[1024, 1024])])
    emb = cfg.offline_sam_embedding_subdir if cfg.use_offline_sam_embeddings else None

    train_list, val_list = [], []
    for spec in cfg.datasets:
        dtype = spec["type"]
        root = spec["root"]
        train_root = os.path.join(root, spec.get("train_subdir", "train"))
        val_root = os.path.join(root, spec.get("val_subdir", "val"))
        train_ds, val_ds = DATASET_BUILDERS[dtype](train_root, val_root, transform, emb)
        train_list.append(train_ds)
        val_list.append(val_ds)
        if rank == 0:
            print(
                f"[Data] {dtype}: train={len(train_ds)} imgs/{len(train_ds.scenes)} scenes "
                f"(root={train_root})"
            )

    train_ds = train_list[0] if len(train_list) == 1 else MergedMultiSceneDataset(train_list)
    val_ds = val_list[0] if len(val_list) == 1 else MergedMultiSceneDataset(val_list)
    if rank == 0:
        print(
            f"[Data] Offline SAM embeddings: {'enabled' if emb else 'disabled'} (subdir={emb}). "
            f"Final train: {len(train_ds)} imgs / {len(train_ds.scenes)} scenes; "
            f"val: {len(val_ds)} imgs / {len(val_ds.scenes)} scenes"
        )
    return train_ds, val_ds


def build_dataloaders(cfg, train_ds, val_ds, is_distributed, rank, world_size):
    if cfg.mode == "object_union":
        train_loader = create_object_union_dataloader(
            dataset=train_ds,
            batch_size=cfg.batch_size,
            num_frames_per_object=cfg.object_union_num_frames_per_object,
            num_objects=cfg.object_union_num_objects,
            target_total_frames=cfg.object_union_target_total_frames,
            pose_diverse_transition_epoch=cfg.object_union_pose_diverse_transition_epoch,
            distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
        )
        train_num_frames = int(train_loader.batch_sampler.frames_per_scene_group)

        if cfg.use_object_union_sampling_for_val:
            val_loader = create_object_union_dataloader(
                dataset=val_ds,
                batch_size=cfg.batch_size,
                num_frames_per_object=cfg.object_union_num_frames_per_object,
                num_objects=cfg.object_union_num_objects,
                target_total_frames=cfg.object_union_target_total_frames,
                pose_diverse_transition_epoch=None,
                distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                shuffle=False,
            )
            val_num_frames = int(val_loader.batch_sampler.frames_per_scene_group)
        else:
            val_loader = create_dataloader(
                dataset=val_ds, batch_size=cfg.batch_size, num_frames=cfg.base_num_frames,
                distributed=is_distributed, rank=rank, world_size=world_size,
                shuffle=False, frame_sampling=cfg.frame_sampling,
            )
            val_num_frames = cfg.base_num_frames

        if rank == 0:
            num_obj = train_loader.batch_sampler.num_objects
            num_obj_str = "ALL" if num_obj is None else str(num_obj)
            approx = f"{num_obj * train_num_frames}" if num_obj is not None else "varies (all objects)"
            print(
                f"[Data] object-union sampling: num_objects={num_obj_str}, "
                f"num_frames_per_object={cfg.object_union_num_frames_per_object}, "
                f"frames_per_sample={train_num_frames}, approx_frames_per_scene_per_epoch={approx}, "
                f"random_scene->pose_diverse transition epoch={cfg.object_union_pose_diverse_transition_epoch}"
            )
    else:  # random
        train_loader = create_dataloader(
            dataset=train_ds, batch_size=cfg.batch_size, num_frames=cfg.base_num_frames,
            distributed=is_distributed, rank=rank, world_size=world_size,
            shuffle=True, frame_sampling=cfg.frame_sampling,
        )
        train_num_frames = cfg.base_num_frames
        val_loader = create_dataloader(
            dataset=val_ds, batch_size=cfg.batch_size, num_frames=cfg.base_num_frames,
            distributed=is_distributed, rank=rank, world_size=world_size,
            shuffle=False, frame_sampling=cfg.frame_sampling,
        )
        val_num_frames = cfg.base_num_frames
        if rank == 0:
            print(f"[Data] standard scene sampling: mode={cfg.frame_sampling}, num_frames={train_num_frames}")

    if train_num_frames != val_num_frames:
        raise ValueError(
            f"Train/val frame count mismatch: train={train_num_frames}, val={val_num_frames}. "
            "Align sampler settings."
        )
    return train_loader, val_loader, train_num_frames


def build_and_freeze_model(cfg, device, rank, world_size, is_distributed):
    if rank == 0:
        print(f"Building model on device: {device}")
    model = build_sam_vggt(device=device)

    required_components = ["sam", "vggt", "embedding_fusion_mlp", "cross_attention_fusion"]
    missing = []
    for comp in required_components:
        if not hasattr(model, comp):
            missing.append(comp)
        elif comp == "sam":
            if not (hasattr(model.sam, "image_encoder") and hasattr(model.sam, "prompt_encoder") and hasattr(model.sam, "mask_decoder")):
                missing.append(f"{comp} (missing subcomponents)")
    if missing:
        raise RuntimeError(f"[Rank {rank}] Model missing required components: {missing}")

    # Freeze encoders.
    for module, name in (
        (model.sam.image_encoder, "image encoder"),
        (model.vggt, "VGGT"),
        (model.sam.prompt_encoder, "prompt encoder"),
    ):
        for p in module.parameters():
            p.requires_grad = False
        if rank == 0:
            print(f"Frozen {name}: {sum(p.numel() for p in module.parameters()):,} parameters")

    # Train fusion modules + mask decoder (which holds the IoU-prediction head).
    for module, name in (
        (model.embedding_fusion_mlp, "embedding fusion MLP"),
        (model.cross_attention_fusion, "cross attention fusion"),
        (model.sam.mask_decoder, "mask decoder"),
    ):
        for p in module.parameters():
            p.requires_grad = True
        if rank == 0:
            print(f"Unfrozen {name}: {sum(p.numel() for p in module.parameters()):,} parameters")

    if is_distributed:
        dist.barrier()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable_elements = sum(p.numel() for p in trainable_params)
    if is_distributed:
        t = torch.tensor(num_trainable_elements, device=device, dtype=torch.long)
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        if not all(x.item() == num_trainable_elements for x in gathered):
            if rank == 0:
                for r, x in enumerate(gathered):
                    print(f"  Rank {r}: {x.item():,} trainable parameters")
            dist.barrier()
            raise RuntimeError(
                f"Rank {rank} has {num_trainable_elements:,} trainable params, inconsistent across ranks."
            )
        if rank == 0:
            print(f"All ranks have {num_trainable_elements:,} trainable parameters ({len(trainable_params)} tensors) - verified")
    elif rank == 0:
        print(f"Total trainable parameters: {num_trainable_elements:,} ({len(trainable_params)} tensors)")

    return model


def build_optimizer(model, cfg, rank):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    fusion_params = list(model.embedding_fusion_mlp.parameters()) + list(model.cross_attention_fusion.parameters())
    decoder_params = list(model.sam.mask_decoder.parameters())
    fusion_ids = {id(p) for p in fusion_params}
    decoder_ids = {id(p) for p in decoder_params}
    other_params = [p for p in trainable_params if id(p) not in fusion_ids and id(p) not in decoder_ids]

    param_groups = []
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": cfg.fusion_lr})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": cfg.decoder_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": cfg.fusion_lr})

    if rank == 0:
        print(
            f"Optimizer param groups: fusion={sum(p.numel() for p in fusion_params):,} (lr={cfg.fusion_lr}), "
            f"decoder={sum(p.numel() for p in decoder_params):,} (lr={cfg.decoder_lr}), "
            f"other={sum(p.numel() for p in other_params):,} (lr={cfg.fusion_lr})"
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler_t_max, eta_min=cfg.scheduler_eta_min
    )
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train SAM-VGGT model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (see configs/).")
    parser.add_argument("--mode", type=str, default=None, choices=["object_union", "random"],
                        help="Override sampling mode from the config.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    parser.add_argument("--finetune_from", type=str, default=None,
                        help="Override finetune checkpoint (weights only). Pass '' to train from scratch. "
                             "Ignored when --resume is set.")
    args = parser.parse_args()

    overrides = {
        "mode": args.mode,
        "resume": args.resume,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    if args.finetune_from is not None:  # allow "" to mean "from scratch"
        overrides["finetune_from"] = args.finetune_from
    cfg = load_config(args.config, overrides)

    rank, local_rank, world_size, is_distributed, device = setup_distributed()

    train_ds, val_ds = build_datasets(cfg, rank)
    train_loader, val_loader, train_num_frames = build_dataloaders(
        cfg, train_ds, val_ds, is_distributed, rank, world_size
    )

    model = build_and_freeze_model(cfg, device, rank, world_size, is_distributed)
    optimizer, scheduler = build_optimizer(model, cfg, rank)

    model = model.to(device)
    if is_distributed:
        dist.barrier()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
        )
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")
    else:
        print(f"Using single GPU: {device}")

    finetune_path = cfg.finetune_from if cfg.finetune_from else None
    if cfg.resume:
        finetune_path = None

    time_start = time.time()
    train_sam_vggt(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        num_frames=train_num_frames,
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
        resume_path=cfg.resume,
        finetune_from=finetune_path,
    )
    if rank == 0:
        print(f"Training completed in {time.time() - time_start:.1f} seconds")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
