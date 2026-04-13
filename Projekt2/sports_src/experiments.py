"""Experiment helpers for PyTorch-based sports image classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .model import build_cnn_classifier
from .trainer import TorchTrainer, get_device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def create_optimizer(name: str, model: torch.nn.Module, lr: float, weight_decay: float = 1e-4):
    """Create a PyTorch optimizer by name."""

    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'.")


@dataclass
class ExperimentConfig:
    name: str
    dropout: float = 0.3
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    epochs: int = 20
    patience: int = 5
    weight_decay: float = 1e-4


def run_experiment(data: Dict[str, Any], config: ExperimentConfig, seed: int = 42) -> Dict[str, Any]:
    """Run a single PyTorch experiment and return metrics and history."""

    set_seed(seed)

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    num_classes = int(len(data["idx_to_label"]))
    device = get_device(prefer_cuda=True)

    model = build_cnn_classifier(num_classes=num_classes, dropout=config.dropout)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(config.optimizer, model=model, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    trainer = TorchTrainer(model, optimizer, criterion, device=device, scheduler=scheduler)

    print(f"Experiment: {config.name}")
    print(f"Device: {device}")
    print(model)

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config.epochs,
        patience=config.patience,
    )

    val_metrics = trainer.evaluate(val_loader)
    test_metrics = trainer.evaluate(test_loader)

    return {
        "config": config,
        "history": history,
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "trainer": trainer,
        "model": model,
        "device": str(device),
    }
