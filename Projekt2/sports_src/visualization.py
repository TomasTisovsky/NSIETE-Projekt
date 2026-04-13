"""Visualization utilities for training history and evaluation."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .trainer import TrainingHistory


def plot_training_curves(history: TrainingHistory, title: str | None = None) -> None:
    epochs = np.arange(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history.train_loss, label="train_loss")
    axes[0].plot(epochs, history.val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, history.train_acc, label="train_acc")
    axes[1].plot(epochs, history.val_acc, label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    axes[2].plot(epochs, history.train_macro_f1, label="train_macro_f1")
    axes[2].plot(epochs, history.val_macro_f1, label="val_macro_f1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].set_title("Macro F1")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true: Iterable[int], y_pred: Iterable[int], class_names: list[str]) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
