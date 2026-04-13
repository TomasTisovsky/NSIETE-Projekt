from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from config import get_experiment_config, list_experiments
from src.data import (
    compute_class_weights,
    create_dataloaders,
    prepare_dataframes,
    save_split_csvs,
)
from src.metrics import (
    build_classification_report,
    build_confusion_matrix,
    build_metrics_summary,
)
from src.model import (
    SUPPORTED_MODELS,
    create_model,
    get_trainable_parameters,
    unfreeze_all_parameters,
)
from src.trainer import evaluate_one_epoch, train_one_epoch
from src.transforms import build_transforms
from src.utils import (
    count_trainable_parameters,
    ensure_dir,
    get_device,
    save_class_mapping,
    save_json,
    set_seed,
    timestamp_now,
)
from src.visualize import plot_confusion_matrix, plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple sports image classifier.")
    parser.add_argument("--data-dir", type=str, default="sports", help="Dataset root path.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline_no_aug",
        choices=list_experiments(),
        help="Which predefined experiment config to run.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help="Optional model override.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epochs override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader workers override.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Where run folders are saved.",
    )
    parser.add_argument(
        "--disable-unfreeze",
        action="store_true",
        help="Keep backbone frozen for whole training.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.disable_unfreeze:
        config["unfreeze_epoch"] = None
    return config


def create_run_dirs(output_root: str | Path, experiment_name: str):
    run_dir = ensure_dir(Path(output_root) / f"{experiment_name}_{timestamp_now()}")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_dir = ensure_dir(run_dir / "metrics")
    plots_dir = ensure_dir(run_dir / "plots")
    splits_dir = ensure_dir(run_dir / "splits")
    return run_dir, checkpoints_dir, metrics_dir, plots_dir, splits_dir


def make_scheduler(optimizer, config: dict):
    if not config.get("use_scheduler", True):
        return None

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        min_lr=float(config["min_lr"]),
    )


def main() -> None:
    args = parse_args()
    config = get_experiment_config(args.experiment)
    config = apply_cli_overrides(config, args)

    set_seed(int(config["seed"]))
    device = get_device()

    run_dir, checkpoints_dir, metrics_dir, plots_dir, splits_dir = create_run_dirs(
        output_root=args.output_root,
        experiment_name=args.experiment,
    )

    save_json(config, run_dir / "config_used.json")

    print(f"Running experiment: {args.experiment}")
    print(f"Model: {config['model_name']}")
    print(f"Device: {device}")
    print(f"Outputs: {run_dir}")

    train_transform, eval_transform = build_transforms(
        image_size=int(config["image_size"]),
        augmentation_mode=str(config["augmentation_mode"]),
    )

    splits, class_to_idx, layout = prepare_dataframes(args.data_dir, config)
    save_split_csvs(splits, splits_dir)
    save_class_mapping(class_to_idx, run_dir / "class_mapping.json")

    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    datasets, dataloaders = create_dataloaders(
        splits=splits,
        class_to_idx=class_to_idx,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(
        model_name=str(config["model_name"]),
        num_classes=len(class_to_idx),
        pretrained=bool(config["pretrained"]),
        freeze_backbone=bool(config["freeze_backbone"]),
    ).to(device)

    if bool(config["use_weighted_loss"]):
        class_weights = compute_class_weights(
            targets=datasets["train"].targets,
            num_classes=len(class_to_idx),
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr=float(config["lr_head"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, config)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_path = checkpoints_dir / "best_model.pt"

    history_rows = []
    backbone_unfrozen = False

    for epoch in range(1, int(config["epochs"]) + 1):
        unfreeze_epoch = config.get("unfreeze_epoch")
        if (
            unfreeze_epoch is not None
            and not backbone_unfrozen
            and epoch == int(unfreeze_epoch)
        ):
            unfreeze_all_parameters(model)
            backbone_unfrozen = True
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config["lr_finetune"]),
                weight_decay=float(config["weight_decay"]),
            )
            scheduler = make_scheduler(optimizer, config)
            print(f"Backbone unfrozen at epoch {epoch}.")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, _, _, _ = evaluate_one_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "model_name": config["model_name"],
                "num_classes": len(class_to_idx),
                "class_to_idx": class_to_idx,
                "best_epoch": best_epoch,
                "config": config,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config["patience"]):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(metrics_dir / "epoch_history.csv", index=False)
    plot_history(history_df, plots_dir / "training_curves.png")

    if not best_checkpoint_path.exists():
        raise RuntimeError("Best checkpoint was not saved.")

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, val_true, val_pred, val_prob = evaluate_one_epoch(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=device,
    )
    test_loss, test_acc, test_true, test_pred, test_prob = evaluate_one_epoch(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )

    val_metrics = build_metrics_summary(val_true, val_pred, val_prob)
    val_metrics["loss"] = float(val_loss)
    val_metrics["accuracy"] = float(val_acc)

    test_metrics = build_metrics_summary(test_true, test_pred, test_prob)
    test_metrics["loss"] = float(test_loss)
    test_metrics["accuracy"] = float(test_acc)

    report_text, report_dict = build_classification_report(test_true, test_pred, class_names)
    report_path = metrics_dir / "test_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    save_json(report_dict, metrics_dir / "test_classification_report.json")

    confusion = build_confusion_matrix(test_true, test_pred, num_classes=len(class_names))
    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    confusion_df.to_csv(metrics_dir / "test_confusion_matrix.csv")
    plot_confusion_matrix(
        confusion=confusion,
        class_names=class_names,
        output_path=plots_dir / "test_confusion_matrix.png",
        title="Test Confusion Matrix",
    )

    layout_serialized = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in layout.items()
    }

    summary = {
        "experiment": args.experiment,
        "model_name": config["model_name"],
        "best_epoch": best_epoch,
        "num_classes": len(class_to_idx),
        "dataset_sizes": {split_name: int(len(df)) for split_name, df in splits.items()},
        "split_layout": layout_serialized,
        "class_merge_enabled": bool(config["enable_class_merge"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, metrics_dir / "summary_metrics.json")

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
