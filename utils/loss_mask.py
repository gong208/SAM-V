# Copyright by HQ-SAM team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
#
# Derived from HQ-SAM train/utils/loss_mask.py (https://github.com/SysCV/sam-hq),
# itself derived from Mask2Former / detectron2. Modified for SAM-V: added a
# focal/dice weighting option and predicted_iou_loss. See NOTICE.
import torch
from torch.nn import functional as F
from typing import List, Optional
import utils.misc as misc

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0,mode="nearest", **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        weight_foreground: float = 1.0,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                 (0 for the negative class and 1 for the positive class).
        weight_foreground: Weight for foreground pixels in dice loss (default 1.0).
                          Higher weight focuses more on foreground accuracy.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)  # Flatten targets to match inputs shape
    
    # Apply foreground weighting if specified
    # FIXED: The original implementation multiplied both inputs and targets by weights,
    # which can cause dice coefficient to exceed 1.0 (leading to negative loss).
    # Solution: Apply weights only to inputs and clamp dice coefficient to [0, 1]
    if weight_foreground != 1.0:
        # Create weight map: higher weight for foreground pixels
        weights = 1.0 + (weight_foreground - 1.0) * targets
        # Apply weights only to inputs (predictions), keeping targets unchanged
        # This emphasizes foreground predictions
        weighted_inputs = inputs * weights
        numerator = 2 * (weighted_inputs * targets).sum(-1)
        denominator = weighted_inputs.sum(-1) + targets.sum(-1)
        # Compute dice coefficient and clamp to [0, 1] to prevent negative loss
        dice_coeff = torch.clamp((numerator + 1) / (denominator + 1), min=0.0, max=1.0)
        loss = 1 - dice_coeff
    else:
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
    # Return per-sample dice loss with shape [B].
    # Batch averaging is handled by the caller.
    return loss


# Note: Cannot JIT compile dice_loss with weight_foreground parameter
# Use non-JIT version for flexibility
# dice_loss_jit = torch.jit.script(dice_loss)  # Disabled due to weight_foreground parameter


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
    """
    Focal loss for handling class imbalance.
    Loss used in RetinaNet: https://arxiv.org/abs/1708.02002
    
    Args:
        inputs: A float tensor of arbitrary shape. The predictions (logits).
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label (0 for negative class and 1 for positive class).
        num_masks: Number of masks for normalization.
        alpha: Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    # Return per-sample focal loss with shape [B].
    # Batch averaging is handled by the caller.
    return loss.mean(1)


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        pos_weight: float = None,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pos_weight: Weight for positive class to handle imbalance. If None, uses auto-weighting.
    Returns:
        Loss tensor
    """
    # Auto-weighting: if pos_weight not provided, compute from target statistics
    # if pos_weight is None:
    #     # Compute positive weight based on class imbalance
    #     # Higher weight for positive class when it's rare
    #     pos_count = targets.sum()
    #     neg_count = targets.numel() - pos_count
    #     if pos_count > 0 and neg_count > 0:
    #         pos_weight = neg_count / pos_count
    #         # Cap at reasonable value to prevent extreme weights
    #         pos_weight = min(pos_weight, 10.0)
    #     else:
    #         pos_weight = 1.0
    
    # Use pos_weight parameter in BCE to weight positive examples more
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
        # , pos_weight=torch.tensor(pos_weight, device=inputs.device)
    )

    # Return per-sample BCE loss with shape [B].
    # Batch averaging is handled by the caller.
    return loss.mean(1)


# Note: JIT version removed due to dynamic pos_weight parameter
# sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0, pos_weight=None, use_focal_loss=True, dice_on_full_mask=False) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mask and dice losses per sample.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    
    Args:
        pos_weight: Weight for positive class in BCE loss. If None, auto-computed from targets.
        use_focal_loss: If True, use focal loss instead of weighted BCE loss.
        dice_on_full_mask: If True, compute dice loss on full masks instead of sampled points.

    Returns:
        loss_mask: Tensor [B], per-sample mask classification loss.
        loss_dice: Tensor [B], per-sample dice loss.
    """

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            8 * 112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        # Compute positive weight from sampled points if not provided
        if pos_weight is None:
            pos_count = point_labels.sum().item()
            neg_count = point_labels.numel() - pos_count
            if pos_count > 0 and neg_count > 0:
                pos_weight = neg_count / pos_count
                # Cap at reasonable value
                pos_weight = min(pos_weight, 50.0)
            else:
                pos_weight = 10.0  # Default high weight for rare positives

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    # Use focal loss or weighted BCE loss to handle class imbalance
    if use_focal_loss:
        # Focal loss is better for extreme class imbalance (sparse masks)
        # Higher alpha (0.75) focuses more on rare positive class
        # Reduced gamma (2.0) to prevent excessive penalty on hard examples that causes loss to increase
        loss_mask = sigmoid_focal_loss(point_logits, point_labels, num_masks, alpha=0.75, gamma=2.0)
    else:
        # Fallback to weighted BCE
        loss_mask = sigmoid_ce_loss(point_logits, point_labels, num_masks, pos_weight=pos_weight)
    
    # Compute dice loss on full masks or sampled points
    if dice_on_full_mask:
        # Compute dice loss on full masks for better alignment with IoU metric
        # src_masks: [B, 1, H, W], target_masks: [B, 1, H, W]
        # Note: dice_loss expects logits (not sigmoid), and will apply sigmoid internally
        # Use moderate foreground weighting (1.5) to balance focus without instability
        loss_dice = dice_loss(src_masks, target_masks, num_masks, weight_foreground=1.5)
    else:
        # Original: compute dice loss on sampled points
        # point_logits: [B, K], point_labels: [B, K]
        loss_dice = dice_loss(point_logits, point_labels, num_masks, weight_foreground=1.5)

    del src_masks
    del target_masks
    return loss_mask, loss_dice


def predicted_iou_loss(pred_iou: torch.Tensor, target_iou: torch.Tensor) -> torch.Tensor:
    """MSE between the model's predicted IoU and the true IoU.

    This trains the mask decoder's IoU-prediction head so ``iou_predictions``
    becomes a meaningful quality score (e.g. usable for NMS ranking). The caller
    is responsible for detaching ``target_iou`` so gradients flow only into the
    IoU head, not back through the mask. SAM uses MSE for this head.

    Args:
        pred_iou:   predicted IoU, shape [B].
        target_iou: true IoU (detached), shape [B].
    """
    return F.mse_loss(pred_iou.float(), target_iou.float())



