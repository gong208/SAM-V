"""
Multi-View Everything Mode for SamVGGT.

Automatically segments all objects across multiple views of a scene by
placing a uniform point grid on **every** frame and merging the predicted
panoramic masks through NMS.

The wrapper accesses model sub-modules directly to separate image encoding
(expensive, done once) from prompt decoding (cheap, done per point batch).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms
from typing import Any, Dict, List, Optional, Tuple

from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_point_grid,
    calculate_stability_score,
    mask_to_rle_pytorch,
    rle_to_mask,
)


def _has_masks(data: MaskData) -> bool:
    """Check whether a MaskData object contains any masks."""
    return "rles" in data._stats and len(data["rles"]) > 0


class SamVGGTAutomaticMaskGenerator:
    """
    Automatic mask generator for multi-view scenes using SamVGGT.

    Algorithm overview:
        1. Encode all N views once (SAM image encoder + VGGT aggregator
           + per-pixel fusion MLP).  Features are cached.
        2. Place a uniform ``points_per_side × points_per_side`` grid on
           **every** frame, giving ``points_per_side² × N`` total prompts.
        3. For each point, run prompt encoding -> VGGT prompt fusion ->
           SAM mask decoder on the *panoramic* feature map to produce a
           mask that spans all N views.
        4. Filter by predicted IoU and stability score; panoramic box NMS
           across all masks from all frames.
        5. Assemble per-object annotation dicts with both panoramic and
           per-view masks.
    """

    def __init__(
        self,
        model: nn.Module,
        points_per_side: int = 16,
        points_per_batch: int = 8,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.90,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        output_mode: str = "binary_mask",
        nms_score: str = "iou_preds",
        nms_iou_type: str = "box",
    ) -> None:
        """
        Args:
            model: A ``SamVGGT`` model instance (see sam_vggt_model.py).
            points_per_side: Grid resolution **per frame**.
                Total prompts = ``points_per_side² × N_views``.
            points_per_batch: Points decoded in one mask-decoder call.
                Lower values save GPU memory when N is large.
            pred_iou_thresh: Keep masks with predicted IoU above this.
            stability_score_thresh: Keep masks with stability above this.
            stability_score_offset: Logit offset used to compute stability.
            box_nms_thresh: IoU threshold for panoramic NMS.
            output_mode: ``"binary_mask"`` or ``"rle"``.
            nms_score: MaskData key used to rank masks during panoramic NMS.
                Defaults to ``"iou_preds"`` (the IoU-prediction head, supervised
                by the IoU loss in the ScanNet++ v2 finetune), matching the
                benchmark CLI default. ``"stability_score"`` (computed from the
                mask logits) only helped older checkpoints whose IoU head was
                not supervised.
            nms_iou_type: Overlap metric used to merge duplicate masks during
                panoramic NMS. ``"box"`` (default) uses fast box-IoU on the
                panoramic bounding boxes; ``"mask"`` uses exact mask-IoU on the
                panoramic masks (stricter -- it won't suppress masks that share
                a bounding box but cover different pixels).
        """
        assert nms_score in ("stability_score", "iou_preds"), (
            f"Unknown nms_score '{nms_score}'. Use 'stability_score' or 'iou_preds'."
        )
        assert nms_iou_type in ("box", "mask"), (
            f"Unknown nms_iou_type '{nms_iou_type}'. Use 'box' or 'mask'."
        )
        assert output_mode in ("binary_mask", "rle"), (
            f"Unknown output_mode '{output_mode}'. Use 'binary_mask' or 'rle'."
        )

        self.model = model
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.output_mode = output_mode
        self.nms_score = nms_score
        self.nms_iou_type = nms_iou_type

        self.grid = build_point_grid(points_per_side)

        # Cached encoder outputs – populated in _encode_images
        self._N: int = 0
        self._H: int = 0
        self._W: int = 0
        self._LR_H: int = 256
        self._LR_W: int = 256
        self._device: torch.device = torch.device("cpu")
        self._concat_embed: Optional[torch.Tensor] = None
        self._dense_pe_cat: Optional[torch.Tensor] = None
        self._dense_e_cat: Optional[torch.Tensor] = None
        self._cam_tokens: Optional[torch.Tensor] = None
        self._vggt_feats_perm: Optional[torch.Tensor] = None

        # Diagnostics – populated during generate(), readable afterwards
        self.prompted_points: List[Dict[str, Any]] = []
        self.points_after_iou: List[Dict[str, Any]] = []
        self.points_after_stability: List[Dict[str, Any]] = []
        self.prompt_groups: List[Dict[str, Any]] = []

    # ==================================================================
    # Image encoding (run once per scene)
    # ==================================================================

    def _encode_images(self, images: torch.Tensor) -> None:
        """Run SAM + VGGT encoders and cache features for decoding."""
        if images.dim() == 4:
            images = images.unsqueeze(0)  # [N,3,H,W] -> [1,N,3,H,W]

        B, N, C, H, W = images.shape
        assert B == 1, "Everything mode processes one scene at a time."

        self._N = N
        self._H = H
        self._W = W
        self._device = next(self.model.parameters()).device
        images = images.to(self._device)

        # --- Preprocess -------------------------------------------------
        sam_input = self.model.sam.preprocess(images)
        vggt_input, _ = self.model.preprocess_vggt_images(images)

        # --- Encode ------------------------------------------------------
        sam_feats, _ = self.model.encode_sam_batched(sam_input)
        del sam_input
        vggt_feats, cam_tokens, _ = self.model.encode_vggt_batched(vggt_input)
        del vggt_input

        # --- Fuse per-frame features -------------------------------------
        fused = self.model.fuse_embeddings_batched(sam_feats, vggt_feats)
        del sam_feats

        # --- Panoramic image embedding [1, 256, 64, 64*N] ----------------
        _, _, Cf, Hf, Wf = fused.shape
        self._concat_embed = (
            fused.permute(0, 2, 3, 1, 4).reshape(1, Cf, Hf, N * Wf)
        )
        del fused

        # --- Dense positional encoding (panoramic) -----------------------
        dense_pe = self.model.sam.prompt_encoder.get_dense_pe()  # [1,256,64,64]
        self._dense_pe_cat = dense_pe.repeat(1, 1, 1, N)

        # --- Dense no-mask embedding (panoramic) -------------------------
        pe_mod = self.model.sam.prompt_encoder
        embed_h, embed_w = pe_mod.image_embedding_size
        no_mask = pe_mod.no_mask_embed.weight.reshape(1, -1, 1, 1)
        dense_e = no_mask.expand(1, -1, embed_h, embed_w)
        self._dense_e_cat = dense_e.repeat(1, 1, 1, N)

        # --- VGGT features for prompt fusion -----------------------------
        self._cam_tokens = cam_tokens          # [1, N, 2048]
        # Pre-permute for efficient spatial indexing: [N, 64, 64, 2048]
        self._vggt_feats_perm = (
            vggt_feats[0].permute(0, 2, 3, 1).contiguous()
        )
        del vggt_feats

    # ==================================================================
    # Point decoding
    # ==================================================================

    def _decode_prompt_groups_batch(
        self,
        points: np.ndarray,
        frame_indices: np.ndarray,
        point_labels: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode *P* grouped point prompts into panoramic masks.

        Each prompt group is treated as one batch item. A prompt group may
        contain one or more point prompts on a single source frame.

        Args:
            points: ``[P, K, 2]`` pixel coordinates ``(x, y)`` in the
                original image frame.
            frame_indices: ``[P]`` source view index for each prompt group.
            point_labels: ``[P, K]`` SAM point labels. Use ``-1`` for padded
                points that should be ignored.

        Returns:
            masks:     ``[P, 1, 256, 256*N]`` low-resolution logit masks.
            iou_preds: ``[P, 1]`` predicted IoU scores.
        """
        P = len(points)
        dev = self._device

        coords = torch.as_tensor(points, dtype=torch.float32, device=dev)
        frame_idx = torch.as_tensor(frame_indices, dtype=torch.long, device=dev)
        labels = torch.as_tensor(point_labels, dtype=torch.int, device=dev)
        if coords.ndim != 3:
            raise ValueError(f"Expected grouped point prompts [P, K, 2], got {coords.shape}")
        if labels.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f"point_labels shape {labels.shape} is incompatible with points shape {coords.shape}"
            )

        # --- SAM prompt encoder ------------------------------------------
        sparse_e, _ = self.model.sam.prompt_encoder(
            points=(coords, labels),
            boxes=None,
            masks=None,
        )

        # --- VGGT-aware prompt fusion (memory-efficient) -----------------
        D = sparse_e.shape[-1]                       # 256
        K = coords.shape[1]
        real_tokens = sparse_e[:, :K, :]
        valid_mask = labels >= 0

        if valid_mask.any():
            real_tokens_flat = real_tokens[valid_mask].unsqueeze(1)  # [M,1,256]
            coords_flat = coords[valid_mask]                         # [M,2]
            frame_idx_flat = frame_idx[:, None].expand(P, K)[valid_mask]

            scale_x = 64.0 / float(self._W)
            scale_y = 64.0 / float(self._H)
            xi = torch.clamp((coords_flat[:, 0] * scale_x).long(), 0, 63)
            yi = torch.clamp((coords_flat[:, 1] * scale_y).long(), 0, 63)

            cam_pts = self._cam_tokens[0, frame_idx_flat].unsqueeze(1)  # [M,1,2048]
            vggt_pts = self._vggt_feats_perm[
                frame_idx_flat, yi, xi
            ].unsqueeze(1)                                              # [M,1,2048]

            kv = torch.cat([cam_pts, vggt_pts], dim=1)                  # [M,2,2048]
            fused_real_flat, _ = self.model.cross_attention_fusion(
                real_tokens_flat.reshape(-1, 1, D),
                kv.reshape(-1, 2, 2048),
            )

            fused_real = real_tokens.clone()
            fused_real[valid_mask] = fused_real_flat[:, 0, :]
        else:
            fused_real = real_tokens

        fused_prompts = sparse_e.clone()
        fused_prompts[:, :K, :] = fused_real

        # --- SAM mask decoder --------------------------------------------
        masks, iou_preds = self.model.sam.mask_decoder(
            image_embeddings=self._concat_embed,       # [1, 256, 64, 64*N]
            image_pe=self._dense_pe_cat,               # [1, 256, 64, 64*N]
            sparse_prompt_embeddings=fused_prompts,    # [P, 2, 256]
            dense_prompt_embeddings=self._dense_e_cat, # [1, 256, 64, 64*N]
            multimask_output=False,
        )
        return masks, iou_preds

    # ==================================================================
    # Per-round processing: batch decode → filter
    # ==================================================================

    def _process_prompt_groups(
        self,
        prompt_groups: List[Dict[str, Any]],
    ) -> MaskData:
        """Decode all prompt groups in mini-batches, applying IoU and stability filters."""
        data = MaskData()
        mask_thresh = self.model.mask_threshold
        n_total = n_after_iou = n_after_stab = 0
        logged_first = False

        for start in range(0, len(prompt_groups), self.points_per_batch):
            batch_groups = prompt_groups[start : start + self.points_per_batch]
            batch_size = len(batch_groups)
            max_points = max(len(group["points"]) for group in batch_groups)

            batch_pts = np.zeros((batch_size, max_points, 2), dtype=np.float32)
            batch_labels = np.full((batch_size, max_points), -1, dtype=np.int64)
            batch_fi = np.zeros((batch_size,), dtype=np.int64)
            batch_prompt_ids = np.zeros((batch_size,), dtype=np.int64)
            batch_source_points = np.zeros((batch_size, 2), dtype=np.float32)
            batch_proposal_area = np.full((batch_size,), -1.0, dtype=np.float32)
            batch_proposal_pred_iou = np.full((batch_size,), np.nan, dtype=np.float32)
            batch_proposal_stability = np.full((batch_size,), np.nan, dtype=np.float32)
            batch_prompt_kind: List[str] = []

            for i, group in enumerate(batch_groups):
                group_points = np.asarray(group["points"], dtype=np.float32)
                group_labels = np.asarray(group.get("labels"), dtype=np.int64)
                num_points = len(group_points)
                batch_pts[i, :num_points] = group_points
                batch_labels[i, :num_points] = group_labels
                batch_fi[i] = int(group["frame_index"])
                batch_prompt_ids[i] = int(group["prompt_id"])
                batch_source_points[i] = group_points[0]
                if "proposal_area" in group and group["proposal_area"] is not None:
                    batch_proposal_area[i] = float(group["proposal_area"])
                if "proposal_predicted_iou" in group and group["proposal_predicted_iou"] is not None:
                    batch_proposal_pred_iou[i] = float(group["proposal_predicted_iou"])
                if "proposal_stability_score" in group and group["proposal_stability_score"] is not None:
                    batch_proposal_stability[i] = float(group["proposal_stability_score"])
                batch_prompt_kind.append(str(group.get("prompt_kind", "grid")))

            masks, ious = self._decode_prompt_groups_batch(batch_pts, batch_fi, batch_labels)
            masks = masks[:, 0]      # [P, H_lr, W_lr*N]
            ious = ious[:, 0]        # [P]
            P = len(ious)
            n_total += P

            bd = MaskData(
                masks=masks,
                iou_preds=ious,
                points=torch.as_tensor(batch_source_points, device=masks.device),
                prompt_points=torch.as_tensor(batch_pts, device=masks.device),
                prompt_labels=torch.as_tensor(batch_labels, dtype=torch.long, device=masks.device),
                frame_indices=torch.as_tensor(
                    batch_fi, dtype=torch.long, device=masks.device,
                ),
                prompt_ids=torch.as_tensor(
                    batch_prompt_ids, dtype=torch.long, device=masks.device,
                ),
                proposal_area=torch.as_tensor(batch_proposal_area, device=masks.device),
                proposal_predicted_iou=torch.as_tensor(batch_proposal_pred_iou, device=masks.device),
                proposal_stability_score=torch.as_tensor(batch_proposal_stability, device=masks.device),
                prompt_kind=batch_prompt_kind,
            )

            if not logged_first:
                logged_first = True
                iou_np = ious.cpu().float().numpy()
                logit_vals = masks.cpu().float()
                fg_frac = (masks > mask_thresh).float().mean(dim=(1, 2)).cpu().numpy()
                print(
                    f"  [first batch] IoU: min={iou_np.min():.4f} "
                    f"max={iou_np.max():.4f} mean={iou_np.mean():.4f} | "
                    f"Logits: min={logit_vals.min().item():.4f} "
                    f"max={logit_vals.max().item():.4f} | "
                    f"FG frac: mean={fg_frac.mean():.4f}"
                )

            # -- predicted-IoU filter --
            if self.pred_iou_thresh > 0.0:
                bd.filter(bd["iou_preds"] > self.pred_iou_thresh)
            n_after_iou += len(bd["iou_preds"])
            if len(bd["iou_preds"]) == 0:
                continue

            pts_after_iou = bd["prompt_points"].detach().cpu().numpy()
            labels_after_iou = bd["prompt_labels"].detach().cpu().numpy()
            fi_after_iou = bd["frame_indices"].detach().cpu().numpy()
            prompt_ids_after_iou = bd["prompt_ids"].detach().cpu().numpy()
            for i in range(len(pts_after_iou)):
                valid = labels_after_iou[i] >= 0
                for point in pts_after_iou[i][valid]:
                    self.points_after_iou.append({
                        "point": point.tolist(),
                        "view": int(fi_after_iou[i]),
                        "round": 0,
                        "prompt_id": int(prompt_ids_after_iou[i]),
                    })

            # -- stability filter --
            bd["stability_score"] = calculate_stability_score(
                bd["masks"], mask_thresh, self.stability_score_offset,
            )
            if self.stability_score_thresh > 0.0:
                bd.filter(bd["stability_score"] >= self.stability_score_thresh)
            n_after_stab += len(bd["iou_preds"])
            if len(bd["iou_preds"]) == 0:
                continue

            pts_after_stab = bd["prompt_points"].detach().cpu().numpy()
            labels_after_stab = bd["prompt_labels"].detach().cpu().numpy()
            fi_after_stab = bd["frame_indices"].detach().cpu().numpy()
            prompt_ids_after_stab = bd["prompt_ids"].detach().cpu().numpy()
            for i in range(len(pts_after_stab)):
                valid = labels_after_stab[i] >= 0
                for point in pts_after_stab[i][valid]:
                    self.points_after_stability.append({
                        "point": point.tolist(),
                        "view": int(fi_after_stab[i]),
                        "round": 0,
                        "prompt_id": int(prompt_ids_after_stab[i]),
                    })

            # -- threshold, boxes, RLE --
            bd["masks"] = bd["masks"] > mask_thresh
            bd["boxes"] = batched_mask_to_box(bd["masks"])
            bd["rles"] = mask_to_rle_pytorch(bd["masks"])
            del bd["masks"]

            data.cat(bd)

        print(
            f"  Prompt groups: {n_total} total -> "
            f"{n_after_iou} after IoU (>{self.pred_iou_thresh}) -> "
            f"{n_after_stab} after stability (>={self.stability_score_thresh})"
        )
        return data

    # ==================================================================
    # Panoramic NMS
    # ==================================================================

    def _nms(self, data: MaskData, iou_threshold: float) -> MaskData:
        """Panoramic NMS to merge duplicate masks.

        Overlap is measured by box-IoU on the panoramic bounding boxes
        (``self.nms_iou_type == "box"``) or by exact mask-IoU on the panoramic
        masks (``"mask"``). Masks are ranked by ``self.nms_score``:
        ``iou_preds`` (default; the IoU head is supervised in the v2
        finetune) or ``stability_score``. Falls back
        to ``iou_preds`` if the requested key is missing.
        """
        if not _has_masks(data):
            return data
        score_key = self.nms_score if self.nms_score in data._stats else "iou_preds"
        if self.nms_iou_type == "mask":
            keep = self._mask_nms(data["rles"], data[score_key], iou_threshold)
        else:
            keep = batched_nms(
                data["boxes"].float(),
                data[score_key],
                torch.zeros_like(data["boxes"][:, 0]),   # single category
                iou_threshold=iou_threshold,
            )
        data.filter(keep)
        return data

    @staticmethod
    def _mask_nms(
        rles: List[Dict[str, Any]],
        scores: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        """Greedy NMS using exact mask-IoU on the panoramic masks.

        Returns the kept indices (sorted by descending ``scores``), matching
        the convention of ``torchvision.ops.batched_nms`` so the result can be
        passed straight to ``MaskData.filter``.
        """
        device = scores.device
        num = len(rles)
        if num == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Decode panoramic masks into a flat [num, H*W] float matrix so mask
        # intersections reduce to a single matmul. These are low-resolution
        # (256 x 256*N) panoramas, so the dense matrix is affordable.
        masks = torch.stack([
            torch.from_numpy(rle_to_mask(rle)) for rle in rles
        ]).reshape(num, -1).to(device=device, dtype=torch.float32)

        areas = masks.sum(dim=1)                         # [num]
        inter = masks @ masks.t()                        # [num, num]
        union = areas[:, None] + areas[None, :] - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        order = torch.argsort(scores, descending=True)
        suppressed = torch.zeros(num, dtype=torch.bool, device=device)
        keep: List[int] = []
        for idx in order.tolist():
            if suppressed[idx]:
                continue
            keep.append(idx)
            overlap = iou[idx] > iou_threshold
            overlap[idx] = False
            suppressed |= overlap

        return torch.as_tensor(keep, dtype=torch.long, device=device)

    # ==================================================================
    # Output construction
    # ==================================================================

    def _build_output(self, data: MaskData) -> List[Dict[str, Any]]:
        """Assemble per-object annotation dicts with both output formats."""
        if not _has_masks(data):
            return []

        N, H, W = self._N, self._H, self._W
        lr_h, lr_w = self._LR_H, self._LR_W
        scale_y = H / lr_h
        scale_x = (W * N) / (lr_w * N)     # == W / lr_w

        data.to_numpy()
        annotations: List[Dict[str, Any]] = []

        for idx in range(len(data["rles"])):
            rle = data["rles"][idx]
            lr_mask = rle_to_mask(rle)                # [lr_h, lr_w*N] bool

            # Upsample per-frame independently then reassemble horizontally.
            lr_t = torch.from_numpy(lr_mask.astype(np.float32))      # [lr_h, lr_w*N]
            lr_batch = lr_t.reshape(lr_h, N, lr_w).permute(1, 0, 2)  # [N, lr_h, lr_w]
            hr_batch = F.interpolate(
                lr_batch.unsqueeze(1),                                # [N, 1, lr_h, lr_w]
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )                                                         # [N, 1, H, W]
            pan_mask = (hr_batch[:, 0].reshape(N, H, W)
                        .permute(1, 0, 2).reshape(H, N * W)
                        > 0.5).numpy()                                # [H, W*N]

            # Per-view split
            per_view_masks: Dict[int, Any] = {}
            bboxes_per_view: Dict[int, List[int]] = {}
            for v in range(N):
                vm = pan_mask[:, v * W : (v + 1) * W]
                if not vm.any():
                    continue
                ys, xs = np.where(vm)
                bboxes_per_view[v] = [
                    int(xs.min()), int(ys.min()),
                    int(xs.max()), int(ys.max()),
                ]
                if self.output_mode == "binary_mask":
                    per_view_masks[v] = vm.copy()
                else:
                    per_view_masks[v] = _mask_to_rle_numpy(vm)

            # Panoramic bbox (scaled from low-res)
            box_lr = data["boxes"][idx]                # [4] numpy
            bbox_pan = [
                float(box_lr[0] * scale_x),
                float(box_lr[1] * scale_y),
                float(box_lr[2] * scale_x),
                float(box_lr[3] * scale_y),
            ]

            ann: Dict[str, Any] = {
                "panoramic_mask": (
                    pan_mask if self.output_mode == "binary_mask"
                    else _mask_to_rle_numpy(pan_mask)
                ),
                "per_view_masks": per_view_masks,
                "bbox_panoramic": bbox_pan,
                "bboxes_per_view": bboxes_per_view,
                "predicted_iou": float(data["iou_preds"][idx]),
                "stability_score": float(data["stability_score"][idx]),
                "area": int(pan_mask.sum()),
                "source_view": int(data["frame_indices"][idx]),
                "point_coords": data["prompt_points"][idx][data["prompt_labels"][idx] >= 0].tolist(),
                "prompt_id": int(data["prompt_ids"][idx]),
                "prompt_kind": data["prompt_kind"][idx],
                "source_point": data["points"][idx].tolist(),
                "proposal_area": None if data["proposal_area"][idx] < 0 else float(data["proposal_area"][idx]),
                "proposal_predicted_iou": (
                    None if np.isnan(data["proposal_predicted_iou"][idx])
                    else float(data["proposal_predicted_iou"][idx])
                ),
                "proposal_stability_score": (
                    None if np.isnan(data["proposal_stability_score"][idx])
                    else float(data["proposal_stability_score"][idx])
                ),
            }
            annotations.append(ann)

        return annotations

    # ==================================================================
    # Cache management
    # ==================================================================

    def _reset_cache(self) -> None:
        self._concat_embed = None
        self._dense_pe_cat = None
        self._dense_e_cat = None
        self._cam_tokens = None
        self._vggt_feats_perm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================================================================
    # Main entry point
    # ==================================================================

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_frame_idx: Optional[int] = None,
        prompt_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment every object across all views of a scene.

        Places a ``points_per_side × points_per_side`` grid on either:
          - every frame (default), or
          - one selected frame via ``prompt_frame_idx``.
        It then decodes all prompts into panoramic masks and merges
        duplicates via NMS.

        Args:
            images: ``[N, 3, H, W]`` or ``[1, N, 3, H, W]`` tensor with
                pixel values in **[0, 255]** float range.  ``H`` and ``W``
                should be 1024 for a model trained at that resolution.
            prompt_frame_idx: Optional source-view index for prompting.
                If ``None``, prompt on every view.

        Returns:
            A list of annotation dicts (one per detected object).
        """
        self._encode_images(images)
        self.prompted_points = []
        self.points_after_iou = []
        self.points_after_stability = []

        try:
            prompt_groups_norm = self._normalize_prompt_groups(
                prompt_groups=self._build_grid_prompt_groups(prompt_frame_idx)
                if prompt_groups is None
                else prompt_groups
            )
            self.prompt_groups = prompt_groups_norm

            if prompt_groups is None:
                prompt_views = sorted({int(group["frame_index"]) for group in prompt_groups_norm})
                if len(prompt_views) == self._N:
                    print(
                        f"  Prompting {self.points_per_side}x{self.points_per_side} "
                        f"grid on each of {self._N} frames = {len(prompt_groups_norm)} prompt groups"
                    )
                else:
                    print(
                        f"  Prompting {self.points_per_side}x{self.points_per_side} "
                        f"grid on frame {int(prompt_views[0])} only = {len(prompt_groups_norm)} prompt groups"
                    )
            else:
                print(f"  Prompting from {len(prompt_groups_norm)} custom prompt groups")

            for group in prompt_groups_norm:
                for point in group["points"][group["labels"] >= 0]:
                    self.prompted_points.append({
                        "point": point.tolist(),
                        "view": int(group["frame_index"]),
                        "round": 0,
                        "prompt_id": int(group["prompt_id"]),
                    })

            all_data = self._process_prompt_groups(prompt_groups_norm)

            if _has_masks(all_data):
                n_before_nms = len(all_data["rles"])
                self._nms(all_data, self.box_nms_thresh)
                print(
                    f"  NMS: {n_before_nms} masks -> "
                    f"{len(all_data['rles'])} after NMS "
                    f"(IoU threshold={self.box_nms_thresh})"
                )

            return self._build_output(all_data)

        finally:
            self._reset_cache()

    def _build_grid_prompt_groups(self, prompt_frame_idx: Optional[int]) -> List[Dict[str, Any]]:
        pixel_grid = self.grid * np.array([self._W, self._H], dtype=np.float32)
        if prompt_frame_idx is None:
            prompt_views = np.arange(self._N, dtype=np.int64)
            frame_indices = np.repeat(prompt_views, len(pixel_grid))
            points = np.tile(pixel_grid, (self._N, 1))
        else:
            if prompt_frame_idx < 0 or prompt_frame_idx >= self._N:
                raise ValueError(
                    f"prompt_frame_idx must be in [0, {self._N - 1}], got {prompt_frame_idx}"
                )
            frame_indices = np.full(len(pixel_grid), prompt_frame_idx, dtype=np.int64)
            points = pixel_grid.copy()

        prompt_groups = []
        for prompt_id, (point, frame_index) in enumerate(zip(points, frame_indices)):
            prompt_groups.append(
                {
                    "prompt_id": int(prompt_id),
                    "frame_index": int(frame_index),
                    "points": np.asarray([point], dtype=np.float32),
                    "labels": np.asarray([1], dtype=np.int64),
                    "prompt_kind": "grid",
                }
            )
        return prompt_groups

    def _normalize_prompt_groups(self, prompt_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for prompt_id, group in enumerate(prompt_groups):
            points = np.asarray(group["points"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(f"Prompt group points must have shape [K,2], got {points.shape}")
            labels = np.asarray(
                group.get("labels", np.ones(len(points), dtype=np.int64)),
                dtype=np.int64,
            )
            if labels.shape != (len(points),):
                raise ValueError(
                    f"Prompt group labels must have shape [{len(points)}], got {labels.shape}"
                )
            if np.all(labels < 0):
                raise ValueError("Prompt group must contain at least one valid point.")
            normalized.append(
                {
                    **group,
                    "prompt_id": int(group.get("prompt_id", prompt_id)),
                    "frame_index": int(group["frame_index"]),
                    "points": points,
                    "labels": labels,
                    "prompt_kind": str(group.get("prompt_kind", "custom")),
                }
            )
        return normalized


# ======================================================================
# Standalone helpers
# ======================================================================

def _mask_to_rle_numpy(mask: np.ndarray) -> Dict[str, Any]:
    """Encode a single H×W boolean numpy array as uncompressed RLE."""
    h, w = mask.shape
    flat = mask.T.flatten()   # Fortran order (column-major)
    diff = np.diff(flat.astype(np.int8))
    change = np.concatenate([[0], np.where(diff != 0)[0] + 1, [h * w]])
    counts = np.diff(change).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": [h, w], "counts": counts}
