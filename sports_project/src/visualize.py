from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    if history_df.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    confusion,
    class_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    show_labels = n_classes <= 30

    fig_size = 10 if show_labels else 14
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        confusion,
        cmap="Blues",
        xticklabels=class_names if show_labels else False,
        yticklabels=class_names if show_labels else False,
        cbar=True,
    )

    ax.set_title(title if show_labels else f"{title} ({n_classes} classes, labels hidden)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
