from __future__ import annotations

from copy import deepcopy

# Kept for compatibility with merge_classes.py, but unused for now.
DEFAULT_CLASS_MERGE_MAP = {}

BASE_CONFIG = {
    "seed": 42,
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 2,
    "val_size": 0.15,
    "test_size": 0.15,
    "model_name": "resnet18",
    "backup_model_name": "efficientnet_b0",
    "pretrained": True,
    "freeze_backbone": True,
    "unfreeze_epoch": 8,
    "lr_head": 1e-3,
    "lr_finetune": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "patience": 5,
    "use_scheduler": True,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "min_lr": 1e-6,
    "use_weighted_loss": True,
    "augmentation_mode": "moderate_aug",  # one of: none, brightness_aug, moderate_aug
    "enable_class_merge": False,
    "class_merge_map": DEFAULT_CLASS_MERGE_MAP,
}

EXPERIMENTS = {
    "baseline_no_aug": {
        "augmentation_mode": "none",
        "enable_class_merge": False,
    },
    "brightness_aug": {
        "augmentation_mode": "brightness_aug",
        "enable_class_merge": False,
    },
    "moderate_aug": {
        "augmentation_mode": "moderate_aug",
        "enable_class_merge": False,
    },
}


def get_experiment_config(experiment_name: str) -> dict:
    if experiment_name not in EXPERIMENTS:
        valid = ", ".join(sorted(EXPERIMENTS.keys()))
        raise ValueError(f"Unknown experiment '{experiment_name}'. Valid: {valid}")

    config = deepcopy(BASE_CONFIG)
    config.update(deepcopy(EXPERIMENTS[experiment_name]))
    return config


def list_experiments() -> list[str]:
    return sorted(EXPERIMENTS.keys())