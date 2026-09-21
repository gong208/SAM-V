"""Loading SAM-V checkpoints, slim or full.

A released checkpoint carries only the trained modules (feature-fusion MLP,
prompt-fusion cross-attention, SAM mask decoder); the frozen SAM and VGGT
weights come from the base checkpoints ``build_sam_vggt`` already loaded. That
means a slim checkpoint *must* be loaded non-strictly -- so we verify it
explicitly instead of trusting it. A silently half-applied checkpoint would
still run and still produce plausible masks, which is the failure this module
exists to prevent.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

SLIM_CHECKPOINT_FORMAT = "sam-v-slim-v1"

# Training checkpoints also store only the trainable tensors. They differ from a
# release checkpoint in that they still carry optimizer/scheduler state, so a run
# can be resumed from one.
TRAINABLE_ONLY_FORMAT = "sam-v-trainable-only-v1"

DEFAULT_TRAINABLE_PREFIXES = (
    "embedding_fusion_mlp.",
    "cross_attention_fusion.",
    "sam.mask_decoder.",
)


def is_slim_checkpoint(ckpt: Any) -> bool:
    """A released inference checkpoint: trained weights only, no training state."""
    return isinstance(ckpt, dict) and ckpt.get("format") == SLIM_CHECKPOINT_FORMAT


def is_trainable_only_checkpoint(ckpt: Any) -> bool:
    """A training checkpoint that stores only the trainable tensors."""
    return (isinstance(ckpt, dict)
            and ckpt.get("model_state_format") == TRAINABLE_ONLY_FORMAT)


def is_partial_checkpoint(ckpt: Any) -> bool:
    """Either of the two formats whose model_state_dict omits the frozen weights."""
    return is_slim_checkpoint(ckpt) or is_trainable_only_checkpoint(ckpt)


def load_partial_checkpoint(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    """Load a checkpoint whose model_state_dict holds only the trained tensors.

    Works for both formats. A trainable-only checkpoint records the exact key list
    it was saved from, so verification does not depend on prefix conventions and
    survives a change to which modules are frozen.
    """
    if is_trainable_only_checkpoint(ckpt):
        sd = ckpt["model_state_dict"]
        expected = set(ckpt.get("trainable_keys") or sd.keys())
        result = model.load_state_dict(sd, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint has {len(result.unexpected_keys)} key(s) the model does "
                f"not define, e.g. {result.unexpected_keys[:5]}. Refusing to load."
            )
        absent = sorted(expected - set(sd))
        if absent:
            raise RuntimeError(
                f"Checkpoint declares {len(expected)} trainable tensors but is "
                f"missing {len(absent)}, e.g. {absent[:5]}. Refusing to load."
            )
        unfilled = [k for k in result.missing_keys if k in expected]
        if unfilled:
            raise RuntimeError(
                f"Checkpoint did not supply {len(unfilled)} trainable parameter(s), "
                f"e.g. {unfilled[:5]}. Refusing to load."
            )
        return
    load_slim_checkpoint(model, ckpt)


def load_slim_checkpoint(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    """Load a slim checkpoint onto an already-built model, verifying strictly."""
    sd = ckpt["model_state_dict"]
    prefixes = tuple(ckpt.get("trainable_prefixes", DEFAULT_TRAINABLE_PREFIXES))
    result = model.load_state_dict(sd, strict=False)

    if result.unexpected_keys:
        raise RuntimeError(
            f"Slim checkpoint has {len(result.unexpected_keys)} key(s) the model "
            f"does not define, e.g. {result.unexpected_keys[:5]}. Refusing to load."
        )
    # Every key the model still wants must be a frozen one. If any trainable
    # parameter went unfilled, the base weights would silently stand in for it.
    unfilled = [k for k in result.missing_keys if k.startswith(prefixes)]
    if unfilled:
        raise RuntimeError(
            f"Slim checkpoint did not supply {len(unfilled)} trainable "
            f"parameter(s), e.g. {unfilled[:5]}. Refusing to load."
        )
    expected = ckpt.get("num_tensors")
    if expected is not None and len(sd) != expected:
        raise RuntimeError(
            f"Slim checkpoint declares {expected} tensors but carries {len(sd)}."
        )
