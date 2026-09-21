"""Export a slim, release-ready SAM-V checkpoint.

A training checkpoint is ~7.7 GB, but almost none of that is ours. The model
freezes the SAM image encoder, the SAM prompt encoder and the whole VGGT
backbone, and those three account for over 99% of the file. They are also
already public: ``build_sam_vggt`` loads them from the SAM and VGGT checkpoints
at construction time, so shipping them again would be redundant *and* would mean
redistributing third-party weights.

What is actually trained is three modules -- the feature-fusion MLP, the
prompt-fusion cross-attention, and the SAM mask decoder -- which together come to
roughly 30 MB.

Usage::

    python tools/export_release_checkpoint.py \\
        --input  checkpoints/sam_vggt_epoch0020_scannetpp_v2_finetune.pth \\
        --output release/sam_v_stage2.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch

# The three trainable modules. Kept in sync with the freeze logic in
# training/trainer.py (build_and_freeze_model).
TRAINABLE_PREFIXES = (
    "embedding_fusion_mlp.",
    "cross_attention_fusion.",
    "sam.mask_decoder.",
)

SLIM_FORMAT = "sam-v-slim-v1"

# Metadata worth carrying forward from the training checkpoint, if present.
_METADATA_KEYS = (
    "epoch",
    "val_iou",
    "val_loss",
    "val_mask_loss",
    "val_dice_loss",
    "best_iou",
    "best_val_loss",
    "num_frames",
    "prompt_k_min",
    "prompt_k_max",
)


def extract_trainable(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the tensors belonging to the trainable modules."""
    slim = {k: v for k, v in state_dict.items() if k.startswith(TRAINABLE_PREFIXES)}
    if not slim:
        raise SystemExit(
            "No trainable tensors found. Expected keys prefixed with one of: "
            + ", ".join(TRAINABLE_PREFIXES)
        )
    missing = [p for p in TRAINABLE_PREFIXES
               if not any(k.startswith(p) for k in slim)]
    if missing:
        raise SystemExit(f"Checkpoint is missing entire modules: {missing}")
    return slim


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Training checkpoint (.pth) to slim down.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination for the slim checkpoint.")
    parser.add_argument("--sam_model_type", default="vit_h",
                        help="SAM variant the frozen encoder came from.")
    parser.add_argument("--vggt_model", default="VGGT-1B",
                        help="VGGT variant the frozen backbone came from.")
    parser.add_argument("--note", default="",
                        help="Free-text note stored in the checkpoint metadata.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"No such checkpoint: {args.input}")

    print(f"Reading {args.input} ...")
    ckpt = torch.load(args.input, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit("Expected a dict-style checkpoint.")
    if ckpt.get("format") == SLIM_FORMAT:
        raise SystemExit(f"{args.input} is already a slim checkpoint.")

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    slim_sd = extract_trainable(state_dict)

    full_bytes = sum(v.numel() * v.element_size()
                     for v in state_dict.values() if hasattr(v, "numel"))
    slim_bytes = sum(v.numel() * v.element_size() for v in slim_sd.values())

    payload: Dict[str, Any] = {
        "format": SLIM_FORMAT,
        "model_state_dict": {k: v.clone() for k, v in slim_sd.items()},
        "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "num_tensors": len(slim_sd),
        "base_models": {"sam": args.sam_model_type, "vggt": args.vggt_model},
        "source_checkpoint": args.input.name,
    }
    if args.note:
        payload["note"] = args.note
    for key in _METADATA_KEYS:
        if key in ckpt:
            payload[key] = ckpt[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    digest = sha256_of(args.output)
    on_disk = args.output.stat().st_size

    print()
    print(f"  tensors kept   : {len(slim_sd)}")
    for prefix in TRAINABLE_PREFIXES:
        n = sum(1 for k in slim_sd if k.startswith(prefix))
        b = sum(v.numel() * v.element_size() for k, v in slim_sd.items()
                if k.startswith(prefix))
        print(f"    {prefix:28s} {n:4d} tensors  {b / 1e6:9.3f} MB")
    print(f"  weights in     : {full_bytes / 1e9:.3f} GB")
    print(f"  weights out    : {slim_bytes / 1e6:.3f} MB")
    print(f"  file on disk   : {on_disk / 1e6:.3f} MB "
          f"({args.input.stat().st_size / on_disk:.0f}x smaller)")
    print(f"  sha256         : {digest}")
    print()
    print(json.dumps({"file": args.output.name, "sha256": digest,
                      "bytes": on_disk}, indent=2))


if __name__ == "__main__":
    main()
