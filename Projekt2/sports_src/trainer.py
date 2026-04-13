"""PyTorch training utilities for sports image classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_macro_f1: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchTrainer:
    """Standard PyTorch training loop with validation and early stopping."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = TrainingHistory()
        self.best_state_dict: Optional[dict] = None
        self.best_val_loss = float("inf")

    def _run_epoch(self, loader, training: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        all_targets: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                if training:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

        targets_np = np.concatenate(all_targets)
        preds_np = np.concatenate(all_preds)
        avg_loss = running_loss / max(len(targets_np), 1)
        return avg_loss, targets_np, preds_np

    def fit(self, train_loader, val_loader, epochs: int = 20, patience: int = 5) -> TrainingHistory:
        patience_counter = 0
        for epoch in range(epochs):
            train_loss, train_targets, train_preds = self._run_epoch(train_loader, training=True)
            val_loss, val_targets, val_preds = self._run_epoch(val_loader, training=False)

            train_acc = float(accuracy_score(train_targets, train_preds))
            val_acc = float(accuracy_score(val_targets, val_preds))
            train_macro_f1 = float(f1_score(train_targets, train_preds, average="macro"))
            val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro"))

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_acc.append(val_acc)
            self.history.train_macro_f1.append(train_macro_f1)
            self.history.val_macro_f1.append(val_macro_f1)

            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | "
                f"train_f1={train_macro_f1:.4f} val_f1={val_macro_f1:.4f}"
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        loss, targets, preds = self._run_epoch(loader, training=False)
        return {
            "loss": float(loss),
            "accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro")),
        }

    @torch.no_grad()
    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_targets: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            preds = torch.argmax(logits, dim=1)
            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_targets), np.concatenate(all_preds)
