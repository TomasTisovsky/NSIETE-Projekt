from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import SportsImageDataset
from src.metrics import (
    build_classification_report,
    build_confusion_matrix,
    build_metrics_summary,
)
from src.model import create_model
from src.trainer import evaluate_one_epoch
from src.transforms import build_transforms
from src.utils import get_device, load_class_mapping, load_json, save_json
from src.visualize import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained sports classifier.")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to one run directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override. Default: experiment_dir/checkpoints/best_model.pt",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()

    config_path = experiment_dir / "config_used.json"
    mapping_path = experiment_dir / "class_mapping.json"
    test_csv_path = experiment_dir / "splits" / "test.csv"

    if not config_path.exists() or not mapping_path.exists() or not test_csv_path.exists():
        raise FileNotFoundError(
            "Missing one of required files: config_used.json, class_mapping.json, splits/test.csv"
        )

    config = load_json(config_path)
    class_to_idx, idx_to_class = load_class_mapping(mapping_path)

    test_df = pd.read_csv(test_csv_path)
    if test_df.empty:
        raise RuntimeError("Test split CSV is empty.")

    _, eval_transform = build_transforms(
        image_size=int(config["image_size"]),
        augmentation_mode="none",
    )

    dataset = SportsImageDataset(test_df, class_to_idx, transform=eval_transform)
    batch_size = int(args.batch_size if args.batch_size is not None else config["batch_size"])

    device = get_device()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint is not None
        else (experiment_dir / "checkpoints" / "best_model.pt")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(
        model_name=str(config["model_name"]),
        num_classes=len(class_to_idx),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    summary = build_metrics_summary(y_true, y_pred, y_prob)
    summary["loss"] = float(test_loss)
    summary["accuracy"] = float(test_acc)

    report_text, report_dict = build_classification_report(y_true, y_pred, idx_to_class)
    confusion = build_confusion_matrix(y_true, y_pred, num_classes=len(idx_to_class))

    metrics_dir = experiment_dir / "metrics"
    plots_dir = experiment_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_json(summary, metrics_dir / "test_metrics_recomputed.json")
    save_json(report_dict, metrics_dir / "test_classification_report_recomputed.json")
    (metrics_dir / "test_classification_report_recomputed.txt").write_text(report_text, encoding="utf-8")

    confusion_df = pd.DataFrame(confusion, index=idx_to_class, columns=idx_to_class)
    confusion_df.to_csv(metrics_dir / "test_confusion_matrix_recomputed.csv")
    plot_confusion_matrix(
        confusion=confusion,
        class_names=idx_to_class,
        output_path=plots_dir / "test_confusion_matrix_recomputed.png",
        title="Test Confusion Matrix (Recomputed)",
    )

    print("Evaluation complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test accuracy: {summary['accuracy']:.4f}")
    print(f"Test macro F1: {summary['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
