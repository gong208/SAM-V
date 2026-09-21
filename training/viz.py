"""Visualization and rank-to-rank viz transfer helpers for SAM-VGGT training.

Pure presentation code pulled out of ``trainer.py`` so the training engine stays
readable. Nothing here touches the optimizer or model state.
"""

import io
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


# Fixed scene/object targets that are always logged to wandb each validation run.
FIXED_VAL_VIZ_TARGETS = (
    ("c4c04e6d6c", 31),
    ("7bc286c1b6", 46),
    ("09c1414f1b", 178),
    ("c4c04e6d6c", 42),
)

FIXED_HYPERSIM_VAL_VIZ_TARGETS = (
    ("ai_045_008_01", 17),
    ("ai_045_008_00", 119),
)


def _send_viz_data(viz_data, dest_rank, tag_base, device):
    """Send viz_data to dest_rank using torch.save bytes."""
    buf = io.BytesIO()
    torch.save(viz_data, buf)
    b = buf.getvalue()
    size_t = torch.tensor([len(b)], dtype=torch.long, device=device)
    data_t = torch.tensor(np.frombuffer(b, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    dist.send(size_t, dst=dest_rank, tag=tag_base)
    dist.send(data_t, dst=dest_rank, tag=tag_base + 1)


def _recv_viz_data(src_rank, tag_base, device):
    """Receive viz_data from src_rank."""
    size_t = torch.zeros(1, dtype=torch.long, device=device)
    dist.recv(size_t, src=src_rank, tag=tag_base)
    n = int(size_t.item())
    data_t = torch.zeros(n, dtype=torch.uint8, device=device)
    dist.recv(data_t, src=src_rank, tag=tag_base + 1)
    buf = io.BytesIO(data_t.cpu().numpy().tobytes())
    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(buf, map_location="cpu")


def _show_points_val(coords, labels, ax, marker_size=375):
    """Point prompts overlay (green=pos, red=neg)."""
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def _show_mask_val(mask, ax, color=None):
    """Semi-transparent mask overlay on axis."""
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    elif np.asarray(color).size == 3:
        color = np.concatenate([np.asarray(color), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def visualize_gt_prompt_overlay(
    images,
    gt_mask_cat,
    point_coords,
    point_labels,
    point_frame_indices,
    N=4,
    W=1024,
    scene_name=None,
    frame_ids=None,
    chosen_object_id=None,
    pooled_iou=None,
):
    """
    Original colored image with ground truth overlay and point prompts. One panel per frame.
    Returns PIL Image for wandb logging (no local file save).
    images: [N, 3, H, W], gt_mask_cat: [H, W*N], points in per-frame coords.
    scene_name: optional scene name (e.g. from path). frame_ids: optional list of N frame ids (e.g. ['00092','00093',...]).
    """
    imgs = images.detach().cpu()
    if imgs.dtype.is_floating_point:
        imgs = (imgs.clamp(0, 255) if imgs.max() > 1.5 else (imgs * 255).clamp(0, 255))
    imgs = imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    gt_np = gt_mask_cat.cpu().numpy() if torch.is_tensor(gt_mask_cat) else np.asarray(gt_mask_cat)
    coords_np = point_coords.cpu().numpy() if torch.is_tensor(point_coords) else np.asarray(point_coords)
    plabels_np = point_labels.cpu().numpy() if torch.is_tensor(point_labels) else np.asarray(point_labels).astype(int)
    frames_np = point_frame_indices.cpu().numpy().astype(int) if torch.is_tensor(point_frame_indices) else np.asarray(point_frame_indices).astype(int)
    H = imgs.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))
    if N == 1:
        axes = [axes]
    red_rgba = np.array([1.0, 0.0, 0.0, 0.5])
    for i in range(N):
        ax = axes[i]
        ax.imshow(imgs[i])
        gt_i = gt_np[:, i * W : (i + 1) * W]
        _show_mask_val(gt_i, ax, color=red_rgba)
        idx = frames_np == i
        if idx.any():
            _show_points_val(coords_np[idx], plabels_np[idx], ax)
        title = f"Frame {i}"
        if frame_ids is not None and i < len(frame_ids):
            title += f" ({frame_ids[i]})"
        ax.set_title(title)
        ax.axis("off")
    suptitle_parts = []
    if scene_name is not None:
        suptitle_parts.append(f"Scene: {scene_name}")
    if chosen_object_id is not None:
        suptitle_parts.append(f"obj_id={int(chosen_object_id)}")
    if pooled_iou is not None:
        suptitle_parts.append(f"pooled_iou={float(pooled_iou):.4f}")
    if suptitle_parts:
        fig.suptitle(" | ".join(suptitle_parts), fontsize=10, y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def visualize_pred_mask_overlay(
    images,
    low_res_logits,
    point_coords,
    point_labels,
    point_frame_indices,
    N=4,
    W=1024,
    loss_mask=None,
    loss_dice=None,
    scene_name=None,
    frame_ids=None,
    chosen_object_id=None,
    pooled_iou=None,
):
    """
    Original colored image with predicted mask overlay. One panel per frame.
    Returns PIL Image for wandb logging (no local file save).
    images: [N, 3, H, W], low_res_logits: [1, 1, 256, 256*N] (upsampled & thresholded).
    If loss_mask and loss_dice are provided, display them in the figure title.
    scene_name: optional scene name. frame_ids: optional list of N frame ids.
    """
    imgs = images.detach().cpu()
    if imgs.dtype.is_floating_point:
        imgs = (imgs.clamp(0, 255) if imgs.max() > 1.5 else (imgs * 255).clamp(0, 255))
    imgs = imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    pred_full = F.interpolate(low_res_logits, size=(imgs.shape[1], W * N), mode="bilinear", align_corners=False)
    pred_full = (pred_full[0, 0] > 0).float().cpu().numpy()
    coords_np = point_coords.cpu().numpy() if torch.is_tensor(point_coords) else np.asarray(point_coords)
    plabels_np = point_labels.cpu().numpy() if torch.is_tensor(point_labels) else np.asarray(point_labels).astype(int)
    frames_np = point_frame_indices.cpu().numpy().astype(int) if torch.is_tensor(point_frame_indices) else np.asarray(point_frame_indices).astype(int)
    H = imgs.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))
    if N == 1:
        axes = [axes]
    for i in range(N):
        ax = axes[i]
        ax.imshow(imgs[i])
        mask_i = pred_full[:, i * W : (i + 1) * W]
        _show_mask_val(mask_i, ax, color=np.array([30/255, 144/255, 255/255, 0.6]))
        idx = frames_np == i
        if idx.any():
            _show_points_val(coords_np[idx], plabels_np[idx], ax)
        title = f"Frame {i}"
        if frame_ids is not None and i < len(frame_ids):
            title += f" ({frame_ids[i]})"
        ax.set_title(title)
        ax.axis("off")
    suptitle_parts = []
    if scene_name is not None:
        suptitle_parts.append(f"Scene: {scene_name}")
    if chosen_object_id is not None:
        suptitle_parts.append(f"obj_id={int(chosen_object_id)}")
    if pooled_iou is not None:
        suptitle_parts.append(f"pooled_iou={float(pooled_iou):.4f}")
    if loss_mask is not None and loss_dice is not None:
        suptitle_parts.append(f"loss_mask={loss_mask:.4f}, loss_dice={loss_dice:.4f}")
    if suptitle_parts:
        fig.suptitle(" | ".join(suptitle_parts), fontsize=10, y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def visualize_prompt_points_overlay(
    images,
    point_coords,
    point_labels,
    point_frame_indices,
    N=4,
    scene_name=None,
    frame_ids=None,
    chosen_object_id=None,
):
    """
    Original colored image with prompt points only (no mask overlay).
    Returns PIL Image for local debug saving.
    """
    imgs = images.detach().cpu()
    if imgs.dtype.is_floating_point:
        imgs = (imgs.clamp(0, 255) if imgs.max() > 1.5 else (imgs * 255).clamp(0, 255))
    imgs = imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    coords_np = point_coords.cpu().numpy() if torch.is_tensor(point_coords) else np.asarray(point_coords)
    plabels_np = point_labels.cpu().numpy() if torch.is_tensor(point_labels) else np.asarray(point_labels).astype(int)
    frames_np = point_frame_indices.cpu().numpy().astype(int) if torch.is_tensor(point_frame_indices) else np.asarray(point_frame_indices).astype(int)

    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))
    if N == 1:
        axes = [axes]
    for i in range(N):
        ax = axes[i]
        ax.imshow(imgs[i])
        idx = frames_np == i
        if idx.any():
            _show_points_val(coords_np[idx], plabels_np[idx], ax)
        title = f"Frame {i}"
        if frame_ids is not None and i < len(frame_ids):
            title += f" ({frame_ids[i]})"
        ax.set_title(title)
        ax.axis("off")
    suptitle_parts = []
    if scene_name is not None:
        suptitle_parts.append(f"Scene: {scene_name}")
    if chosen_object_id is not None:
        suptitle_parts.append(f"obj_id={int(chosen_object_id)}")
    if suptitle_parts:
        fig.suptitle(" | ".join(suptitle_parts), fontsize=10, y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def viz_and_pack(viz_data):
    """Build {"gt_prompt", "pred_mask"} PIL overlays from a packed viz tuple.

    viz_data layout (see _make_viz_data in the trainer):
        (low_res_masks[1,1,256,256N], images[N,3,H,W], mask_cat[H,W*N],
         point_coords[K,2], point_labels[K], point_frame_indices[K],
         H, W, N, loss_mask, loss_dice, pooled_iou, chosen_object_id,
         scene_name, frame_ids)
    """
    lrm, imgs_b, mask_cat_b, pc, pl, pfi, h, w, n, loss_m, loss_d, pooled_iou, chosen_object_id, scene_name, frame_ids = viz_data
    gt_img = visualize_gt_prompt_overlay(
        imgs_b, mask_cat_b, pc, pl, pfi, N=n, W=w, scene_name=scene_name, frame_ids=frame_ids,
        chosen_object_id=chosen_object_id, pooled_iou=pooled_iou,
    )
    pred_img = visualize_pred_mask_overlay(
        imgs_b, lrm[0:1], pc, pl, pfi, N=n, W=w,
        loss_mask=loss_m, loss_dice=loss_d, scene_name=scene_name, frame_ids=frame_ids,
        chosen_object_id=chosen_object_id, pooled_iou=pooled_iou,
    )
    return {"gt_prompt": gt_img, "pred_mask": pred_img}


def _save_train_mismatch_visualization(
    save_dir,
    epoch_idx,
    batch_idx,
    sample_idx,
    scene_name,
    frame_ids,
    images_b,
    mask_cat_b,
    low_res_logits_b,
    point_coords_b,
    point_labels_b,
    point_frame_indices_b,
    num_frames,
    frame_width,
    chosen_object_id,
    loss_mask_value,
    loss_dice_value,
    pooled_iou_value,
):
    """Save GT/pred overlays for suspicious train samples."""
    safe_scene = scene_name.replace("/", "_")
    prefix = (
        f"epoch{epoch_idx + 1:03d}_batch{batch_idx:04d}_sample{sample_idx:02d}"
        f"_{safe_scene}_iou{pooled_iou_value:.3f}_dice{loss_dice_value:.3f}"
    )
    gt_img = visualize_gt_prompt_overlay(
        images_b,
        mask_cat_b,
        point_coords_b,
        point_labels_b,
        point_frame_indices_b,
        N=num_frames,
        W=frame_width,
        scene_name=scene_name,
        frame_ids=frame_ids,
        chosen_object_id=chosen_object_id,
        pooled_iou=pooled_iou_value,
    )
    pred_img = visualize_pred_mask_overlay(
        images_b,
        low_res_logits_b,
        point_coords_b,
        point_labels_b,
        point_frame_indices_b,
        N=num_frames,
        W=frame_width,
        loss_mask=loss_mask_value,
        loss_dice=loss_dice_value,
        scene_name=scene_name,
        frame_ids=frame_ids,
        chosen_object_id=chosen_object_id,
        pooled_iou=pooled_iou_value,
    )
    gt_path = os.path.join(save_dir, f"{prefix}_gt.png")
    pred_path = os.path.join(save_dir, f"{prefix}_pred.png")
    gt_img.save(gt_path)
    pred_img.save(pred_path)
    return {
        "prefix": prefix,
        "gt_img": gt_img,
        "pred_img": pred_img,
        "gt_path": gt_path,
        "pred_path": pred_path,
    }
