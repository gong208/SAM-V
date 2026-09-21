"""Training configuration for SAM-VGGT.

All hyperparameters that used to be hardcoded throughout ``trainer.py`` live
here in a single :class:`TrainConfig` dataclass. Experiments are driven by YAML
files (see ``configs/``) loaded via :func:`load_config`, with a small set of CLI
flags allowed to override individual fields.

The dataclass defaults are copied verbatim from the pre-refactor trainer so the
default config reproduces previous runs (and existing checkpoints) exactly.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


def _default_datasets() -> List[Dict[str, Any]]:
    # Matches the previous default (ScanNet++ only) when --scannetpp_only was used.
    return [
        {
            "type": "scannetpp",
            "root": "${SCANNETPP_ROOT}",
        }
    ]


@dataclass
class TrainConfig:
    # --- mode -------------------------------------------------------------
    mode: str = "object_union"  # "object_union" | "random"

    # --- datasets (declarative; replaces --scannetpp_only) ----------------
    # Each entry: {"type": "scannetpp"|"hypersim", "root": <path>,
    #              "train_subdir": "train", "val_subdir": "val"}.
    # One entry -> used directly; multiple -> MergedMultiSceneDataset.
    datasets: List[Dict[str, Any]] = field(default_factory=_default_datasets)

    # --- data / loading ---------------------------------------------------
    batch_size: int = 2
    base_num_frames: int = 8           # used by "random" mode
    frame_sampling: str = "random"     # used by "random" mode
    use_offline_sam_embeddings: bool = False
    offline_sam_embedding_subdir: str = "sam_embeddings"

    # --- object-union sampler --------------------------------------------
    object_union_num_frames_per_object: int = 8
    object_union_num_objects: Optional[int] = None
    object_union_target_total_frames: Optional[int] = None
    object_union_pose_diverse_transition_epoch: int = 0
    use_object_union_sampling_for_val: bool = True

    # --- prompt / point sampling -----------------------------------------
    # Training prompts with all-positive points from a single frame on the
    # chosen mask (the same routine used at eval). The number of points is
    # drawn uniformly from [prompt_k_min, prompt_k_max] once per training batch.
    prompt_k_min: int = 3
    prompt_k_max: int = 5

    # Stage-1 prompt curriculum. None (the default) keeps the behaviour above:
    # single-frame prompts from epoch 0. Set to an epoch E to linearly ramp the
    # per-sample probability of a single-frame prompt from 0 at epoch 0 to 1 at
    # epoch >= E; before that, prompts are the multi-frame half-positive /
    # half-negative form (sample_points_for_instances, positive_only=False).
    # The released stage-1 checkpoint records prompt_transition_epoch 70.
    prompt_transition_epoch: Optional[int] = None

    # --- optimizer / scheduler -------------------------------------------
    fusion_lr: float = 8e-5
    decoder_lr: float = 8e-5
    weight_decay: float = 0.01
    scheduler_t_max: int = 200
    scheduler_eta_min: float = 1e-6

    # --- training loop ----------------------------------------------------
    epochs: int = 100
    patience: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = -1  # periodic checkpoint cadence; -1 => never (only best/latest)
    num_val_batches: Optional[int] = None
    amp: bool = True

    # --- loss -------------------------------------------------------------
    mask_loss_weight: float = 20.0
    pos_weight_cap: float = 50.0
    iou_loss_weight: float = 1.0

    # --- checkpoint / logging --------------------------------------------
    output_dir: str = "checkpoints"
    ckpt_tag: str = "scannetpp"
    wandb_entity: Optional[str] = None  # set to your own W&B entity, or leave unset
    wandb_project: str = "sam-vggt-multiview"
    wandb_run_name: Optional[str] = None  # None -> derived in trainer
    resume: Optional[str] = None
    finetune_from: Optional[str] = "checkpoints/sam_vggt_best_iou_patch.pth"

    def __post_init__(self):
        if self.mode not in ("object_union", "random"):
            raise ValueError(
                f"mode must be 'object_union' or 'random', got {self.mode!r}"
            )
        if not self.datasets:
            raise ValueError("config.datasets must contain at least one entry")
        for spec in self.datasets:
            dtype = spec.get("type")
            if dtype not in ("scannetpp", "hypersim"):
                raise ValueError(
                    f"dataset type must be 'scannetpp' or 'hypersim', got {dtype!r}"
                )
            if "root" not in spec:
                raise ValueError(f"dataset spec missing 'root': {spec!r}")
        if self.prompt_k_min < 1 or self.prompt_k_max < self.prompt_k_min:
            raise ValueError(
                f"require 1 <= prompt_k_min <= prompt_k_max, got "
                f"prompt_k_min={self.prompt_k_min}, prompt_k_max={self.prompt_k_max}"
            )


def _expand_paths(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``~`` in every string in a config.

    Dataset roots and checkpoint paths are machine-specific, so the shipped
    configs reference them as environment variables (``${SCANNETPP_ROOT}``,
    ``${HYPERSIM_ROOT}``, ``${STAGE1_CHECKPOINT}``). An unset variable is left
    verbatim so the resulting error names the variable that was missing.
    """
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_paths(v) for v in value]
    return value


def load_config(path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    """Load a YAML config into a :class:`TrainConfig`, then apply CLI overrides.

    Unknown YAML keys are rejected. CLI overrides win over the file, but only
    when their value is not ``None`` (so unset flags leave the YAML untouched).
    String values may reference environment variables as ``${VAR}``.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(data)}")
    data = _expand_paths(data)

    valid_keys = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_keys)}"
        )

    cfg = TrainConfig(**data)

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if key not in valid_keys:
                raise ValueError(f"Unknown CLI override key: {key}")
            setattr(cfg, key, value)
        cfg.__post_init__()  # re-validate after overrides

    return cfg
