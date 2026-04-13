from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


class SportsImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.paths = self.df["path"].tolist()
        self.targets = [self.class_to_idx[label] for label in self.df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]
        return image, target


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def apply_class_merge(label: str, merge_map: dict[str, str], enable_class_merge: bool) -> str:
    if not enable_class_merge:
        return label
    return merge_map.get(label, label)


def scan_classification_root(
    root_dir: Path,
    merge_map: dict[str, str] | None = None,
    enable_class_merge: bool = False,
) -> pd.DataFrame:
    merge_map = merge_map or {}
    records = []

    if not root_dir.exists():
        return pd.DataFrame(columns=["path", "raw_label", "label"])

    for class_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        raw_label = class_dir.name
        merged_label = apply_class_merge(raw_label, merge_map, enable_class_merge)

        for file_path in class_dir.rglob("*"):
            if _is_image_file(file_path):
                records.append(
                    {
                        "path": str(file_path.resolve()),
                        "raw_label": raw_label,
                        "label": merged_label,
                    }
                )

    return pd.DataFrame(records)


def detect_split_layout(data_dir: Path) -> dict[str, Path | bool | None]:
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    final_train = train_dir if train_dir.exists() else None
    final_val = valid_dir if valid_dir.exists() else (val_dir if val_dir.exists() else None)
    final_test = test_dir if test_dir.exists() else None

    has_split_layout = final_train is not None and (final_val is not None or final_test is not None)

    return {
        "has_split_layout": has_split_layout,
        "train": final_train,
        "val": final_val,
        "test": final_test,
    }


def _safe_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    try:
        left_df, right_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["label"],
        )
    except ValueError:
        left_df, right_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    return left_df.reset_index(drop=True), right_df.reset_index(drop=True)


def split_train_val_test(
    all_df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be smaller than 1.0")

    train_df, temp_df = _safe_split(all_df, test_size=val_size + test_size, seed=seed)

    relative_test = test_size / (val_size + test_size)
    val_df, test_df = _safe_split(temp_df, test_size=relative_test, seed=seed)

    return train_df, val_df, test_df


def prepare_dataframes(data_dir: str | Path, config: dict):
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    merge_map = config.get("class_merge_map", {})
    enable_class_merge = bool(config.get("enable_class_merge", False))
    seed = int(config["seed"])
    val_size = float(config["val_size"])
    test_size = float(config["test_size"])

    layout = detect_split_layout(data_dir)

    if layout["has_split_layout"]:
        train_df = scan_classification_root(layout["train"], merge_map, enable_class_merge)

        if layout["val"] is not None:
            val_df = scan_classification_root(layout["val"], merge_map, enable_class_merge)
        else:
            train_df, val_df = _safe_split(train_df, test_size=val_size, seed=seed)

        if layout["test"] is not None:
            test_df = scan_classification_root(layout["test"], merge_map, enable_class_merge)
        else:
            val_df, test_df = _safe_split(val_df, test_size=0.5, seed=seed)

    else:
        all_df = scan_classification_root(data_dir, merge_map, enable_class_merge)
        train_df, val_df, test_df = split_train_val_test(
            all_df=all_df,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        if split_df.empty:
            raise RuntimeError(f"{split_name} split is empty. Check dataset layout and split settings.")

    all_classes = sorted(
        set(train_df["label"]).union(val_df["label"]).union(test_df["label"])
    )
    class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}

    return splits, class_to_idx, layout


def scan_dataset_records(
    data_dir: str | Path,
    merge_map: dict[str, str] | None = None,
    enable_class_merge: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    layout = detect_split_layout(data_dir)

    if layout["has_split_layout"]:
        frames = []
        for split_name in ("train", "val", "test"):
            split_dir = layout[split_name]
            if split_dir is not None:
                split_df = scan_classification_root(split_dir, merge_map, enable_class_merge)
                frames.append(split_df)

        if not frames:
            return pd.DataFrame(columns=["path", "raw_label", "label"])

        return pd.concat(frames, ignore_index=True)

    return scan_classification_root(data_dir, merge_map, enable_class_merge)


def create_dataloaders(
    splits: dict[str, pd.DataFrame],
    class_to_idx: dict[str, int],
    train_transform,
    eval_transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    datasets = {
        "train": SportsImageDataset(splits["train"], class_to_idx, transform=train_transform),
        "val": SportsImageDataset(splits["val"], class_to_idx, transform=eval_transform),
        "test": SportsImageDataset(splits["test"], class_to_idx, transform=eval_transform),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return datasets, dataloaders


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(targets), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(targets) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def save_split_csvs(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        split_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
