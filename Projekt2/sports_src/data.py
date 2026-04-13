"""Dataset loading and preprocessing for sports image classification using PyTorch.

The goal is to use a fast, GPU-friendly pipeline with torchvision datasets,
transforms and DataLoader workers. The module keeps the EDA logic separate
from training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
SPLIT_FOLDER_NAMES = {"train", "valid", "val", "test"}


@dataclass
class ImagePreprocessingConfig:
    """Configuration for image preprocessing and dataset splitting."""

    image_size: Tuple[int, int] = (128, 128)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    max_classes: Optional[int] = None
    max_images_per_class: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True


def infer_label_from_path(img_path: Path, root: Path) -> str:
    rel_parts = img_path.relative_to(root).parts
    if len(rel_parts) >= 2 and rel_parts[0].lower() in SPLIT_FOLDER_NAMES:
        return rel_parts[1]
    if len(rel_parts) >= 1:
        return rel_parts[0]
    return img_path.parent.name


def index_image_files(dataset_root: Path) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records = []
    for img_path in dataset_root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in ALLOWED_EXTENSIONS:
            records.append({"label": infer_label_from_path(img_path, dataset_root), "path": img_path})

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No image files found. Check dataset structure and extensions.")
    return df


def _inspect_image(path: Path) -> Dict[str, object]:
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
        return {"is_corrupted": False, "error": None, "width": width, "height": height, "mode": mode}
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return {"is_corrupted": True, "error": str(e), "width": np.nan, "height": np.nan, "mode": None}


def build_metadata(df_paths: pd.DataFrame) -> pd.DataFrame:
    meta_rows = []
    for _, row in df_paths.iterrows():
        meta_rows.append(_inspect_image(row["path"]))
    df_meta = pd.concat([df_paths.reset_index(drop=True), pd.DataFrame(meta_rows)], axis=1)
    df_meta["aspect_ratio"] = df_meta["width"] / df_meta["height"]
    df_meta["n_pixels"] = df_meta["width"] * df_meta["height"]
    return df_meta


def filter_valid_images(df_meta: pd.DataFrame) -> pd.DataFrame:
    return df_meta[~df_meta["is_corrupted"]].copy().reset_index(drop=True)


def restrict_classes(df: pd.DataFrame, max_classes: Optional[int]) -> pd.DataFrame:
    if max_classes is None:
        return df
    counts = df["label"].value_counts().sort_values(ascending=False)
    keep_labels = set(counts.head(max_classes).index)
    return df[df["label"].isin(keep_labels)].copy().reset_index(drop=True)


def cap_images_per_class(df: pd.DataFrame, max_images_per_class: Optional[int], *, random_state: int) -> pd.DataFrame:
    if max_images_per_class is None:
        return df
    rng = np.random.default_rng(seed=random_state)
    parts = []
    for _, group in df.groupby("label"):
        if len(group) <= max_images_per_class:
            parts.append(group)
        else:
            indices = rng.choice(len(group), size=max_images_per_class, replace=False)
            parts.append(group.iloc[indices])
    return pd.concat(parts, axis=0).reset_index(drop=True)


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    df = df.copy()
    df["label_idx"] = df["label"].map(label_to_idx).astype(int)
    return df, label_to_idx, idx_to_label


def compute_mean_std_from_paths(paths: pd.Series, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over a list of image paths."""

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for path in paths:
        with Image.open(Path(path)) as img:
            arr = np.asarray(img.convert("RGB").resize(image_size), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sum_sq += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]

    mean = channel_sum / max(n_pixels, 1)
    std = np.sqrt(channel_sum_sq / max(n_pixels, 1) - mean ** 2)
    return mean.astype(np.float32), (std + 1e-6).astype(np.float32)


def build_transforms(image_size: Tuple[int, int], mean: np.ndarray, std: np.ndarray, augment: bool = True):
    train_transforms = [
        transforms.Resize(image_size),
    ]
    if augment:
        train_transforms.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    return transforms.Compose(train_transforms), eval_transforms


class SportsImageDataset(Dataset):
    """Dataset backed by a DataFrame with file paths and integer labels."""

    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[int(idx)]
        with Image.open(Path(row["path"])) as img:
            sample = img.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        target = int(row["label_idx"])
        return sample, target


def load_sports_dataset(
    dataset_root: Path,
    config: Optional[ImagePreprocessingConfig] = None,
) -> Dict[str, object]:
    """Prepare PyTorch datasets and DataLoaders for sports classification."""

    if config is None:
        config = ImagePreprocessingConfig()

    if not np.isclose(config.train_ratio + config.val_ratio + config.test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    df_paths = index_image_files(dataset_root)
    df_meta = filter_valid_images(build_metadata(df_paths))
    df_meta = restrict_classes(df_meta, config.max_classes)
    df_meta = cap_images_per_class(df_meta, config.max_images_per_class, random_state=config.random_state)
    df_meta, label_to_idx, idx_to_label = encode_labels(df_meta)

    filtered_samples = df_meta.copy().reset_index(drop=True)
    filtered_samples["path"] = filtered_samples["path"].apply(Path)

    indices = np.arange(len(filtered_samples))
    labels = filtered_samples["label_idx"].to_numpy(dtype=np.int64)

    train_idx, temp_idx, y_train_idx, y_temp_idx = train_test_split(
        indices,
        labels,
        train_size=config.train_ratio,
        random_state=config.random_state,
        stratify=labels,
    )
    val_ratio_relative = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_idx, test_idx, y_val_idx, y_test_idx = train_test_split(
        temp_idx,
        y_temp_idx,
        train_size=val_ratio_relative,
        random_state=config.random_state,
        stratify=y_temp_idx,
    )

    train_mean, train_std = compute_mean_std_from_paths(filtered_samples.iloc[train_idx]["path"], config.image_size)
    train_transform, eval_transform = build_transforms(config.image_size, train_mean, train_std, augment=config.augment)

    train_dataset = SportsImageDataset(filtered_samples.iloc[train_idx].reset_index(drop=True), transform=train_transform)
    val_dataset = SportsImageDataset(filtered_samples.iloc[val_idx].reset_index(drop=True), transform=eval_transform)
    test_dataset = SportsImageDataset(filtered_samples.iloc[test_idx].reset_index(drop=True), transform=eval_transform)

    class_counts = np.bincount(labels[train_idx], minlength=len(label_to_idx))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels[train_idx]]
    sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "mean": train_mean,
        "std": train_std,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_dataframe": filtered_samples.iloc[train_idx].reset_index(drop=True),
        "val_dataframe": filtered_samples.iloc[val_idx].reset_index(drop=True),
        "test_dataframe": filtered_samples.iloc[test_idx].reset_index(drop=True),
    }
