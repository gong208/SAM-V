"""
Script to verify target masks are loaded and processed correctly.
This helps diagnose if data issues are causing the high dice loss.
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
from utils.dataloader import create_dataloader, Resize

def verify_masks():
    """Verify that target masks are correct and properly formatted."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hypersim_dir = os.path.join(script_dir, "hypersim")
    
    # Create dataloader (same as training)
    train_loader = create_dataloader(
        root_dir=hypersim_dir,
        batch_size=1,
        num_frames=4,
        my_transforms=[
            Resize(size=[1024,1024]),
        ],
        shuffle=True
    )
    
    print("=" * 80)
    print("VERIFYING TARGET MASKS")
    print("=" * 80)
    
    # Check a few batches
    for batch_idx, data in enumerate(train_loader):
        if batch_idx >= 5:  # Check first 5 batches
            break
            
        images = data["images"]  # [B, N, 3, H, W]
        labels = data["labels"]  # [B, N, 1, H, W]
        valid_ids_list = data["valid_ids"]
        image_paths = data.get("image_paths", [])  # List[B], each entry is List[N] of paths
        label_paths = data.get("label_paths", [])  # List[B], each entry is List[N] of paths
        
        B, N, _, H, W = images.shape
        print(f"\n--- Batch {batch_idx} ---")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of frames: {N}")
        
        for b in range(B):
            print(f"\n  Sample {b}:")
            print(f"    Valid IDs: {valid_ids_list[b]}")
            
            # Check label statistics
            labels_2d = labels[b].squeeze(1)  # [N, H, W]
            
            for n in range(N):
                label_frame = labels_2d[n]  # [H, W]
                unique_ids = torch.unique(label_frame)
                print(f"    Frame {n}:")
                # Show image and label paths if available
                if image_paths and len(image_paths) > b and len(image_paths[b]) > n:
                    print(f"      Image path: {image_paths[b][n]}")
                if label_paths and len(label_paths) > b and len(label_paths[b]) > n:
                    print(f"      Label path: {label_paths[b][n]}")
                print(f"      Unique instance IDs: {unique_ids.tolist()[:10]}... (showing first 10)")
                print(f"      Label min/max: {label_frame.min().item()}/{label_frame.max().item()}")
                print(f"      Label dtype: {label_frame.dtype}")
                
                # Check if we can create a binary mask
                if len(valid_ids_list[b]) > 0:
                    chosen_id = valid_ids_list[b][0]  # Use first valid ID
                    binary_mask = (label_frame == chosen_id).float()
                    pos_ratio = binary_mask.mean().item()
                    print(f"      Binary mask (ID={chosen_id}) pos_ratio: {pos_ratio:.4f}")
                    print(f"      Binary mask min/max: {binary_mask.min().item()}/{binary_mask.max().item()}")
                    
                    # Check if mask is reasonable (not all zeros, not all ones)
                    if pos_ratio < 0.001:
                        print(f"      ⚠️  WARNING: Mask is almost all zeros!")
                    elif pos_ratio > 0.99:
                        print(f"      ⚠️  WARNING: Mask is almost all ones!")
                    else:
                        print(f"      ✓ Mask looks reasonable")
        
        # Check concatenated masks
        labels_2d = labels.squeeze(2)  # [B, N, H, W]
        labels_cat = torch.cat([labels_2d[0, i] for i in range(N)], dim=1)  # [H, W*N]
        
        if len(valid_ids_list[0]) > 0:
            chosen_id = valid_ids_list[0][0]
            binary_mask_cat = (labels_cat == chosen_id).float()
            pos_ratio_cat = binary_mask_cat.mean().item()
            print(f"\n  Concatenated mask (ID={chosen_id}):")
            print(f"    Pos ratio: {pos_ratio_cat:.4f}")
            print(f"    Shape: {binary_mask_cat.shape}")
            
            # Downsample to match prediction resolution
            target_low = F.interpolate(
                binary_mask_cat.unsqueeze(0).unsqueeze(0).float(),
                size=(256, 256 * N),
                mode="nearest"
            ).squeeze()
            pos_ratio_low = target_low.mean().item()
            print(f"    After downsampling to 256x{256*N}:")
            print(f"      Pos ratio: {pos_ratio_low:.4f}")
            print(f"      Shape: {target_low.shape}")
            
            if pos_ratio_low < 0.001:
                print(f"    ⚠️  WARNING: Downsampled mask is almost all zeros!")
            elif pos_ratio_low > 0.99:
                print(f"    ⚠️  WARNING: Downsampled mask is almost all ones!")
            else:
                print(f"    ✓ Downsampled mask looks reasonable")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    # Check point sampling
    print("\nChecking point sampling...")
    from utils.misc import sample_points_for_instances
    
    # Create a test case
    test_labels = torch.zeros(1, 1024, 1024 * 4, dtype=torch.int64)
    # Add some foreground
    test_labels[0, 500:600, 500:600] = 1
    test_labels[0, 500:600, 1500:1600] = 1
    test_labels[0, 500:600, 2500:2600] = 1
    test_labels[0, 500:600, 3500:3600] = 1
    
    chosen_ids = torch.tensor([1])
    sampled_points, sampled_labels = sample_points_for_instances(
        test_labels,
        chosen_ids,
        k=20
    )
    
    print(f"Sampled points shape: {sampled_points[0].shape}")
    print(f"Sampled labels shape: {sampled_labels[0].shape}")
    print(f"Sampled labels unique: {torch.unique(sampled_labels[0])}")
    print(f"Positive ratio in sampled points: {sampled_labels[0].float().mean().item():.4f}")
    
    if sampled_labels[0].sum() == 0:
        print("⚠️  WARNING: No positive points sampled!")
    else:
        print("✓ Point sampling looks reasonable")

if __name__ == "__main__":
    verify_masks()

