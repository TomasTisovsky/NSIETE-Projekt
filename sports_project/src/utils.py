from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_class_mapping(class_to_idx: dict[str, int], output_path: str | Path) -> None:
    idx_to_class = [None] * len(class_to_idx)
    for class_name, index in class_to_idx.items():
        idx_to_class[index] = class_name

    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    save_json(payload, output_path)


def load_class_mapping(path: str | Path) -> tuple[dict[str, int], list[str]]:
    payload = load_json(path)
    class_to_idx = {k: int(v) for k, v in payload["class_to_idx"].items()}
    idx_to_class = list(payload["idx_to_class"])
    return class_to_idx, idx_to_class


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
