"""
SAM-VGGT Model: Combines SAM and VGGT encoders for multi-view segmentation.

This model integrates:
- SAM (Segment Anything Model) image encoder
- VGGT (Visual Geometry Grounded Transformer) for multi-view feature aggregation
- Fusion MLPs to combine embeddings
- SAM prompt encoder and mask decoder for segmentation
"""
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Type, Union

from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square

_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_patch_size(model) -> int:
    """Extract patch size from VGGT model."""
    cand_attrs = [
        "backbone.patch_size",
        "backbone.vit.patch_size",
        "backbone.vit.patch_embed.patch_size",
        "backbone.patch_embed.patch_size",
    ]
    for ca in cand_attrs:
        try:
            obj = model
            for part in ca.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (tuple, list)):
                return int(obj[0])
            return int(obj)
        except Exception:
            pass
    return 14


class PerPixelMLP(nn.Module):
    """Per-pixel MLP for fusing image embeddings using 1x1 convolutions."""
    
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        depth: int = 2,
        act: Type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ):
        """
        Args:
            c_in: Input channel dimension
            c_hidden: Hidden layer dimension
            c_out: Output channel dimension
            depth: Number of 1x1 conv layers (>=2). Larger depth increases capacity.
            act: Activation function (default: GELU)
            dropout: Dropout probability after hidden activations
        """
        super().__init__()
        if depth < 2:
            raise ValueError(f"PerPixelMLP depth must be >= 2, got {depth}")

        layers: List[nn.Module] = [
            nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True),
            act(),
        ]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(c_hidden, c_hidden, kernel_size=1, bias=True))
            layers.append(act())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.Conv2d(c_hidden, c_out, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, C_out, H, W]
        """
        return self.net(x)

class CrossAttentionFusion(nn.Module):
    # Adapted from VLM-3R (https://github.com/VITA-Group/VLM-3R), Apache-2.0:
    # llava/model/multimodal_fusion_block/builder.py. See NOTICE.
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(CrossAttentionFusion, self).__init__()
        
        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        
        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        
        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        
        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)
        
        # projection
        self.out_proj = nn.Linear(d_attn, d_clip)
        
        # dropout
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]
        
        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        
        # cross attention
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )
        
        # projection to D_clip dimension
        fused_features = self.out_proj(fused_features)   # [B, N_clip, D_clip]
        
        # residual connection and dropout
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features  # [B, N_clip, D_clip]
        # print(f'status_of_fused_features: max:{fused_features.max():.2f}, min:{fused_features.min():.2f}, mean:{fused_features.mean():.2f}, std:{fused_features.std():.2f}')
        # print(f'status_of_clip_features: max:{clip_features.max():.2f}, min:{clip_features.min():.2f}, mean:{clip_features.mean():.2f}, std:{clip_features.std():.2f}')
        fused_features = self.dropout(fused_features)
        
        return fused_features, attn_weights


class MLPBlock(nn.Module):
    """MLP block for fusing sparse prompt embeddings."""
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embedding_dim: Input embedding dimension
            mlp_dim: Hidden layer dimension
            out_dim: Output dimension
            act: Activation function (default: GELU)
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, out_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, embedding_dim]
        Returns:
            [B, N, out_dim]
        """
        return self.lin2(self.act(self.lin1(x)))


class SamVGGT(nn.Module):
    """
    SAM-VGGT: Multi-view segmentation model combining SAM and VGGT encoders.
    
    Architecture:
        1. SAM encoder extracts single-view features [B, 256, 64, 64]
        2. VGGT aggregator extracts multi-view features [1, N, 2048, 64, 64]
        3. Per-pixel MLP fuses concatenated features -> [N, 256, 64, 64]
        4. Features are concatenated spatially -> [1, 256, 64, 64*N]
        5. Pretrained SAM prompt encoder generates sparse and dense embeddings
        6. Dense embeddings and positional embeddings are concatenated along width for multi-frame input
        7. Prompt MLP fuses SAM prompts with VGGT camera tokens
        8. SAM mask decoder produces final segmentation masks
    """
    
    mask_threshold: float = 0.0
    
    def __init__(
        self,
        sam_model_type: str = "vit_h",
        sam_checkpoint: str = str(_REPO_ROOT / "submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth"),
        vggt_checkpoint: str = str(_REPO_ROOT / "submodules/vggt/checkpoints/model.pt"),
        vggt_img_size: int = 896,
        embed_fusion_hidden: int = 768,
        embed_fusion_depth: int = 2,
        embed_fusion_dropout: float = 0.0,
        prompt_fusion_hidden: int = 768,
        sam_encode_chunk: int = 8,
        device: str = "cuda",
        freeze_sam_encoder: bool = True,
        freeze_vggt: bool = True,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        """
        Args:
            sam_model_type: Type of SAM model ('vit_b', 'vit_l', 'vit_h')
            sam_checkpoint: Path to SAM checkpoint
            vggt_checkpoint: Path to VGGT checkpoint
            vggt_img_size: Image size for VGGT preprocessing (default: 896)
            embed_fusion_hidden: Hidden dimension for embedding fusion MLP
            embed_fusion_depth: Number of layers in embedding fusion MLP
            embed_fusion_dropout: Dropout in embedding fusion MLP hidden layers
            prompt_fusion_hidden: Hidden dimension for prompt fusion MLP
            sam_encode_chunk: Number of frames per SAM encoder micro-batch.
                0 means encode all frames at once.
            device: Device to load models on
            freeze_sam_encoder: Whether to freeze SAM encoder weights
            freeze_vggt: Whether to freeze VGGT weights
            pixel_mean: Mean values for SAM image normalization
            pixel_std: Std values for SAM image normalization
        """
        super().__init__()
        
        self.device = device
        self.vggt_img_size = vggt_img_size
        self.sam_encode_chunk = int(sam_encode_chunk)
        
        # Load SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
        self.sam.eval()
        
        if freeze_sam_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        # Load VGGT model
        self.vggt = self._load_vggt(vggt_checkpoint, device)
        self.vggt.eval()
        
        if freeze_vggt:
            for param in self.vggt.parameters():
                param.requires_grad = False
        
        # Get dimensions
        self.sam_encoder_dim = 256  # SAM encoder output channels
        self.vggt_encoder_dim = 2048  # VGGT output channels (2 * embed_dim)
        self.vggt_camera_dim = 2048  # VGGT camera token dimension
        self.fused_dim = self.sam_encoder_dim + self.vggt_encoder_dim  # 2304
        
        # Per-pixel MLP to fuse SAM and VGGT embeddings
        self.embedding_fusion_mlp = PerPixelMLP(
            c_in=self.fused_dim,
            c_hidden=embed_fusion_hidden,
            c_out=self.sam_encoder_dim,
            depth=embed_fusion_depth,
            dropout=embed_fusion_dropout,
        ).to(device)
        
        self.cross_attention_fusion = CrossAttentionFusion(
            d_clip=self.sam_encoder_dim,
            d_spatial_encoder=self.vggt_camera_dim,
            d_attn=self.sam_encoder_dim,
            num_heads=8,
        ).to(device)

        # SAM preprocessing
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        # Get VGGT patch size
        self.vggt_patch_size = get_patch_size(self.vggt) or 14

    def _load_vggt(self, ckpt: str, device: str) -> VGGT:
        """Load VGGT model from checkpoint."""
        model = VGGT()
        raw = torch.load(ckpt, map_location="cpu")
        
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw:
            sd = raw["model"]
        else:
            sd = raw
            
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        
        print(f"[VGGT] loaded. missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  (first few missing):", missing[:8])
        if unexpected:
            print("  (first few unexpected):", unexpected[:8])
            
        return model.to(device).eval()
    


    def preprocess_vggt_images(self, images: torch.Tensor, target_size: int = 896):
        """
        Preprocess images given as a tensor by center padding to square
        and resizing to target size. Also returns the position information
        of original pixels after transformation.

        Args:
            images (torch.Tensor): Tensor of images with shape (B, N, 3, H, W)
                                or (N, 3, H, W). Values can be 0–1 or 0–255.
            target_size (int): Target size for both width and height.

        Returns:
            tuple:
                images_out (torch.Tensor): (B, N, 3, target_size, target_size)
                coords     (torch.Tensor): (B, N, 6) with [x1, y1, x2, y2, width, height]
        """
        if images.dim() == 4:
            # Treat as (N, 3, H, W) -> add batch dim B=1
            images = images.unsqueeze(0)  # (1, N, 3, H, W)
            added_batch_dim = True
        elif images.dim() == 5:
            added_batch_dim = False
        else:
            raise ValueError(f"Expected images to have 4 or 5 dims, got {images.shape}")

        B, N, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {C}")

        # Convert to float in [0,1] if necessary
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0

        # --- Center pad to square (same H,W for all images in the tensor) ---
        max_dim = max(H, W)
        pad_left  = (max_dim - W) // 2
        pad_right = max_dim - W - pad_left
        pad_top   = (max_dim - H) // 2
        pad_bottom = max_dim - H - pad_top

        # F.pad pad spec is (left, right, top, bottom)
        images_flat = images.view(B * N, C, H, W)
        images_square = F.pad(images_flat, (pad_left, pad_right, pad_top, pad_bottom))

        # --- Resize to target size ---
        images_resized = F.interpolate(
            images_square,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        images_resized = images_resized.clamp(0.0, 1.0)
        images_resized = images_resized.view(B, N, C, target_size, target_size)

        # --- Compute coords for all images (vectorized) ---
        scale = target_size / float(max_dim)

        x1 = torch.full((B, N), pad_left * scale,  dtype=torch.float32, device=images.device)
        y1 = torch.full((B, N), pad_top  * scale,  dtype=torch.float32, device=images.device)
        x2 = torch.full((B, N), (pad_left + W) * scale, dtype=torch.float32, device=images.device)
        y2 = torch.full((B, N), (pad_top  + H) * scale, dtype=torch.float32, device=images.device)
        width  = torch.full((B, N), float(W), dtype=torch.float32, device=images.device)
        height = torch.full((B, N), float(H), dtype=torch.float32, device=images.device)

        coords = torch.stack([x1, y1, x2, y2, width, height], dim=-1)  # (B,N,6)

        # If original input was (N,3,H,W), drop the extra batch dim
        if added_batch_dim:
            images_resized = images_resized.squeeze(0)  # (N,3,target,target)
            coords = coords.squeeze(0)                  # (N,6)

        return images_resized, coords



    
    def encode_sam_batched(
        self, sam_pre: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            sam_pre: [B, N, 3, 1024, 1024]

        Returns:
            feats_bn: [B, N, 256, 64, 64]
            interms:  whatever the encoder returns (list of tensors) for completeness
        """
        B, N = sam_pre.shape[:2]
        flat = sam_pre.view(B * N, 3, 1024, 1024)
        chunk = self.sam_encode_chunk
        with torch.no_grad():
            if chunk and chunk > 0 and chunk < (B * N):
                feats_chunks: List[torch.Tensor] = []
                interms = None
                for start in range(0, B * N, chunk):
                    end = min(start + chunk, B * N)
                    feats_i, interms = self.sam.image_encoder(flat[start:end])
                    feats_chunks.append(feats_i)
                feats = torch.cat(feats_chunks, dim=0)
            else:
                feats, interms = self.sam.image_encoder(flat)  # [B*N,256,64,64]
        feats_bn = feats.view(B, N, 256, 64, 64)
        return feats_bn, interms

    
    def encode_vggt_batched(
        self, imgs_bn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            imgs_bn: [B, N, 3, H, W]

        Returns:
            patch_features: [B, N, 2048, 64, 64]
            camera_tokens:  [B, N, 2048]
            patch_start_idx: int
        """
        B, N, _, H, W = imgs_bn.shape
        gh, gw = H // self.vggt_patch_size, W // self.vggt_patch_size
        assert gh > 0 and gw > 0

        with torch.no_grad():
            agg_tokens_list, ps_idx = self.vggt.aggregator(imgs_bn)  # tokens per layer: [B,N,Tpf,2C]

        tokens = agg_tokens_list[-1]     # [B,N,Tpf,2C]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.vggt.camera_head.token_norm(pose_tokens)
        C2x = tokens.shape[-1]
 
        P = gh * gw
        patch_start = ps_idx
        patch_tok = tokens[:, :, patch_start:, :]  # [B,N,P,2C]
        # -> [B,N,2C,P] -> [B,N,2C,gh,gw]
        patch_tok = patch_tok.permute(0, 1, 3, 2).contiguous().view(B, N, C2x, gh, gw)
        return patch_tok, pose_tokens, patch_start


    def fuse_embeddings_batched(self, sam_feats, vggt_feats):
        # sam_feats:  [B, N, 256, 64, 64]
        # vggt_feats: [B, N, 2048, 64, 64]
        ## 2. FIXME: Consider change to a cross attention fusion blocks
        x = torch.cat([sam_feats, vggt_feats], dim=2)  # [B, N, 2304, 64, 64]
        B, N, C, H, W = x.shape
        fused = self.embedding_fusion_mlp(x.view(B * N, C, H, W))  # [B*N, 256, 64, 64]
        return fused.view(B, N, 256, H, W)

    
    def fuse_prompts(
        self,
        sam_sparse: torch.Tensor,          # [B, Np, 256]
        vggt_cam: torch.Tensor,            # [B, Nframes, 2048]
        vggt_feats: torch.Tensor,          # [B, Nframes, 2048, 64, 64]
        prompt_frame_idx: torch.Tensor,    # [B, N_real]
        point_coords: torch.Tensor,        # [B, N_real, 2]
    ) -> torch.Tensor:
        """
        Fuses SAM embeddings with BOTH the global camera token and the specific local VGGT point feature.
        """
        B, Np, D_sam = sam_sparse.shape
        device = sam_sparse.device

        # Ensure long dtype for indexing
        frame_idx = prompt_frame_idx.to(device=device, dtype=torch.long)  # [B, N_real]
        N_real = frame_idx.shape[1]

        # ----------------------------------------------------------------------
        # 1) Extract SAM point embeddings
        # ----------------------------------------------------------------------
        sam_real = sam_sparse[:, :N_real, :]  # [B, N_real, 256]

        # ----------------------------------------------------------------------
        # 2) Gather Camera Tokens
        # ----------------------------------------------------------------------
        cam_expanded = frame_idx.unsqueeze(-1).expand(B, N_real, 2048)
        cam_for_points = torch.gather(vggt_cam, dim=1, index=cam_expanded)  # [B, N_real, 2048]

        # ----------------------------------------------------------------------
        # 3) Extract Specific VGGT 3D Features at the Point Coordinates
        # ----------------------------------------------------------------------
        scale_factor = 64.0 / 1024.0
        
        x_coords = torch.clamp((point_coords[..., 0] * scale_factor).long(), 0, 63)
        y_coords = torch.clamp((point_coords[..., 1] * scale_factor).long(), 0, 63)

        # Create batch index array to match shapes
        b_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, N_real)

        # Permute VGGT feats to [B, Nframes, H, W, Channels] to easily extract specific pixels
        vggt_feats_permuted = vggt_feats.permute(0, 1, 3, 4, 2)  # [B, N, 64, 64, 2048]
        
        # Advanced indexing to grab the exact 2048-dim point feature
        vggt_point_feats = vggt_feats_permuted[b_indices, frame_idx, y_coords, x_coords] # [B, N_real, 2048]

        # ----------------------------------------------------------------------
        # 4) Cross Attention Fusion
        # Stack the Camera token and the Point Feature token to create a sequence of 2
        # ----------------------------------------------------------------------
        # Shape: [B, N_real, 2, 2048]
        kv_sequence = torch.stack([cam_for_points, vggt_point_feats], dim=2) 
        
        # Reshape for cross attention (Fold B and N_real)
        sam_query = sam_real.reshape(B * N_real, 1, D_sam)
        vggt_kv = kv_sequence.reshape(B * N_real, 2, 2048)

        # Cross attention over the 2 tokens
        fused_real_flat, _ = self.cross_attention_fusion(sam_query, vggt_kv) # [B * N_real, 1, 256]

        # ----------------------------------------------------------------------
        # 5) Write fused real tokens back into SAM token tensor
        # ----------------------------------------------------------------------
        fused_real = fused_real_flat.view(B, N_real, D_sam)                # [B, N_real, 256]
        fused_prompts = sam_sparse.clone()                                 # [B, Np, 256]
        fused_prompts[:, :N_real, :] = fused_real

        return fused_prompts

    def forward(
        self,
        sam_pre: torch.Tensor,      # [B,N,3,1024,1024]
        sam_feats_precomputed: Optional[torch.Tensor] = None,  # [B,N,256,64,64], optional offline SAM embeddings
        point_coords_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i, 2] in original coords
        point_labels_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i]
        point_frame_indices_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i] - frame index for each point
        multimask_output: bool = True,
        visualize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-view, multi-sample forward:
            - Encodes all images across the batch at once with VGGT and SAM.
            - Then loops per sample to run prompt encoding, fusion, decoding, and postprocess.

        Returns:
            A list (len B) of dicts with keys: masks, iou_predictions, low_res_logits, (optional embeddings...)

        Memory (forward peak):
            - Largest cost is usually SAM ``image_encoder`` on ``B*N`` frames and VGGT ``aggregator``
              on ``[B,N,...]``. Set ``sam_encode_chunk`` > 0 to micro-batch SAM only (see ``encode_sam_batched``).
            - VGGT is not chunked here; reduce ``N``, resolution, or batch size if OOM persists.
        """
        B, N, C, H, W = sam_pre.shape

        vggt_input, _ = self.preprocess_vggt_images(sam_pre)
        if sam_feats_precomputed is None:
            sam_input = self.sam.preprocess(sam_pre)
        if not visualize:
            del sam_pre

        if sam_feats_precomputed is None:
            sam_feats_bn, sam_interms = self.encode_sam_batched(sam_input)                      # [B,N,256,64,64]
            del sam_input
            del sam_interms
        else:
            sam_feats_bn = sam_feats_precomputed

        vggt_feats_bn, cam_tokens_bn, _ = self.encode_vggt_batched(vggt_input)             # [B,N,2048,64,64], [B,N,2048]
        del vggt_input
        # 5) Fuse per-frame features (batched) -> [B,N,256,64,64]
        fused_bn = self.fuse_embeddings_batched(sam_feats_bn, vggt_feats_bn)

        # Prepare constant dense PE once
        dense_pe_1 = self.sam.prompt_encoder.get_dense_pe()  # [1,256,64,64]


        batch_points_tuple = None
        if point_coords_list is not None:
            # Convert lists → tensors
            pc_batch = torch.stack(point_coords_list, dim=0).to(self.device)   # [B,Np,2]
            pl_batch = torch.stack(point_labels_list, dim=0).to(self.device)   # [B,Np]
            batch_points_tuple = (pc_batch, pl_batch)
            
        # Prompt encoder (batched)
        sparse_e, dense_e = self.sam.prompt_encoder(
            points=batch_points_tuple,
            boxes=None,
            masks=None,
        )   # sparse_e=[B,Np,256], dense_e=[B,256,64,64]
        B, Np, _ = sparse_e.shape    # after prompt encoder
        # print("B: ", B)
        # print("Np: ", Np)

        # Expand dense_e along width dimension (dim=3) for multi-frame concatenation
        # dense_e is [B,256,64,64], we want [B,256,64,64*N]
        dense_e_cat = dense_e.repeat(1, 1, 1, N)        # [B,256,64,64N]
        dense_pe_cat = dense_pe_1.repeat(B, 1, 1, N)    # [B,256,64,64N]
        
        # [B,N,C_f,H_f,W_f] -> [B,C_f,H_f,N,W_f] -> [B,C_f,H_f,N*W_f] (same as cat along width)
        _, _, C_f, H_f, W_f = fused_bn.shape
        concat_embed_bn = fused_bn.permute(0, 2, 3, 1, 4).reshape(B, C_f, H_f, N * W_f)

        prompt_frame_idx = torch.stack(point_frame_indices_list, dim=0).to(self.device)  # [B,N_real]

        fused_prompts = self.fuse_prompts(
            sam_sparse=sparse_e,          # [B,Np,256]
            vggt_cam=cam_tokens_bn,      # [B,N,2048]
            vggt_feats=vggt_feats_bn,       # [B,N,2048, 64, 64]
            prompt_frame_idx=prompt_frame_idx,
            point_coords=pc_batch,
        )                                 

        low_res_masks_list = []
        iou_pred_list = []
        
        for b_idx in range(B):
            # Extract single batch item (batch_size=1)
            single_image_emb = concat_embed_bn[b_idx:b_idx+1]  # [1,256,64,64N]
            single_image_pe = dense_pe_cat[b_idx:b_idx+1]      # [1,256,64,64N]
            single_sparse = fused_prompts[b_idx:b_idx+1]       # [1,Np,256]
            single_dense = dense_e_cat[b_idx:b_idx+1]         # [1,256,64,64N]
            
            # Call mask decoder for single batch item
            # With batch_size=1, repeat_interleave repeats 1 time, so shapes match correctly
            masks_b, iou_b = self.sam.mask_decoder(
                image_embeddings=single_image_emb,
                image_pe=single_image_pe,
                sparse_prompt_embeddings=single_sparse,
                dense_prompt_embeddings=single_dense,
                multimask_output=multimask_output,
            )
            low_res_masks_list.append(masks_b)
            iou_pred_list.append(iou_b)
        
        # Concatenate results back to batch
        low_res_masks_bn = torch.cat(low_res_masks_list, dim=0)  # [B, ...]
        iou_pred_bn = torch.cat(iou_pred_list, dim=0)           # [B, ...]
        outputs = {
            "iou_predictions": iou_pred_bn,
            "low_res_logits": low_res_masks_bn,
        }
        if visualize:
            masks_b = self.sam.postprocess_masks(
                low_res_masks_bn,
                input_size=(sam_pre.shape[-2], sam_pre.shape[-1] * N),
                original_size=(H, W * N),
            )
            masks_b = masks_b > self.mask_threshold
            outputs["masks"] = masks_b

        return outputs


def build_sam_vggt(
    sam_model_type: str = "vit_h",
    sam_checkpoint: str = str(_REPO_ROOT / "submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth"),
    vggt_checkpoint: str = str(_REPO_ROOT / "submodules/vggt/checkpoints/model.pt"),
    device: str = "cuda",
    **kwargs
) -> SamVGGT:
    """
    Build SAM-VGGT model with default or custom parameters.
    
    Args:
        sam_model_type: Type of SAM model ('vit_b', 'vit_l', 'vit_h')
        sam_checkpoint: Path to SAM checkpoint
        vggt_checkpoint: Path to VGGT checkpoint
        device: Device to load models on
        **kwargs: Additional arguments for SamVGGT
    
    Returns:
        SamVGGT model
    """
    model = SamVGGT(
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
        vggt_checkpoint=vggt_checkpoint,
        device=device,
        **kwargs
    )
    return model
