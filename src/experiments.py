"""
Experiment utilities for running and tracking multiple MLP configurations.
"""

import csv
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from .model import Model
from .layers import Linear, Dropout
from .activations import ReLU, Tanh, Sigmoid, LeakyReLU
from .losses import BCELoss
from .optimizers import SGD, SGDMomentum, RMSprop, Adam
from .trainer import Trainer
from .utils import set_seed


def build_mlp(
    input_dim: int,
    hidden_layers: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
) -> Model:
    """Build configurable MLP model."""
    model = Model()

    activation_map = {
        "relu": ReLU,
        "tanh": Tanh,
        "sigmoid": Sigmoid,
        "leaky_relu": LeakyReLU,
    }
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation '{activation}'.")

    in_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_layers):
        model.add_module(Linear(in_dim, hidden_dim), f"linear_{idx + 1}")
        model.add_module(activation_map[activation](), f"{activation}_{idx + 1}")
        if dropout > 0.0:
            model.add_module(Dropout(p=dropout), f"dropout_{idx + 1}")
        in_dim = hidden_dim

    model.add_module(Linear(in_dim, 1), "linear_out")
    model.add_module(Sigmoid(), "sigmoid_out")
    return model


def create_optimizer(name: str, lr: float):
    """Create optimizer by name."""
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "sgd_momentum":
        return SGDMomentum(lr=lr)
    if name == "rmsprop":
        return RMSprop(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def default_experiment_configs() -> List[Dict[str, Any]]:
    """Ready-to-run experiment configs requested in assignment."""
    return [
        {
            "name": "baseline_relu_adam",
            "hidden_layers": [32, 16],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 80,
            "dropout": 0.0,
        },
        {
            "name": "deeper_relu_adam",
            "hidden_layers": [64, 32, 16],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 100,
            "dropout": 0.0,
        },
        {
            "name": "tanh_sgd",
            "hidden_layers": [32, 16],
            "activation": "tanh",
            "optimizer": "sgd",
            "learning_rate": 1e-2,
            "batch_size": 32,
            "epochs": 100,
            "dropout": 0.0,
        },
        {
            "name": "relu_rmsprop",
            "hidden_layers": [32, 16],
            "activation": "relu",
            "optimizer": "rmsprop",
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 100,
            "dropout": 0.0,
        },
        {
            "name": "relu_adam_small_lr",
            "hidden_layers": [32, 16],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "batch_size": 32,
            "epochs": 120,
            "dropout": 0.0,
        },
        {
            "name": "relu_adam_dropout",
            "hidden_layers": [64, 32, 16],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 120,
            "dropout": 0.2,
        },
    ]


def _append_result_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one experiment row to CSV, creating file with header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_name",
        "architecture",
        "activation",
        "optimizer",
        "learning_rate",
        "batch_size",
        "epochs",
        "dropout",
        "best_validation_loss",
        "validation_accuracy",
        "validation_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_single_experiment(data: Dict[str, Any], config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Run one experiment and return summary metrics."""
    set_seed(seed)

    model = build_mlp(
        input_dim=data["train"]["X"].shape[1],
        hidden_layers=config["hidden_layers"],
        activation=config["activation"],
        dropout=float(config.get("dropout", 0.0)),
    )
    optimizer = create_optimizer(config["optimizer"], lr=float(config["learning_rate"]))
    trainer = Trainer(model, BCELoss(), optimizer, batch_size=int(config["batch_size"]))

    history = trainer.fit(
        data["train"]["X"], data["train"]["y"],
        data["val"]["X"], data["val"]["y"],
        epochs=int(config["epochs"]),
        early_stopping=True,
        patience=15,
    )

    best_val_loss = float(np.min(history["val_loss"]))
    _, val_metrics = trainer.evaluate(data["val"]["X"], data["val"]["y"])
    _, test_metrics = trainer.evaluate(data["test"]["X"], data["test"]["y"])

    return {
        "experiment_name": config["name"],
        "architecture": str(config["hidden_layers"]),
        "activation": config["activation"],
        "optimizer": config["optimizer"],
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "epochs": int(config["epochs"]),
        "dropout": float(config.get("dropout", 0.0)),
        "best_validation_loss": best_val_loss,
        "validation_accuracy": float(val_metrics["accuracy"]),
        "validation_f1": float(val_metrics["f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
    }


def run_experiments(
    data: Dict[str, Any],
    configs: List[Dict[str, Any]],
    results_csv: str = "results/experiments.csv",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run multiple experiments and persist result rows to CSV."""
    csv_path = Path(results_csv)
    if csv_path.exists():
        csv_path.unlink()

    rows: List[Dict[str, Any]] = []
    for config in configs:
        print(f"\nRunning experiment: {config['name']}")
        row = run_single_experiment(data, config, seed=seed)
        _append_result_to_csv(csv_path, row)
        rows.append(row)
        print(
            f"Done {row['experiment_name']} | "
            f"val_f1={row['validation_f1']:.4f} | test_f1={row['test_f1']:.4f}"
        )

    rows.sort(key=lambda r: r["validation_f1"], reverse=True)
    return rows
