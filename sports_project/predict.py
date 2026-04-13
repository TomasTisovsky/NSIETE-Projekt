from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model import create_model
from src.transforms import build_transforms
from src.utils import get_device, load_class_mapping, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-k classes for one image.")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to one run directory.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override. Default: experiment_dir/checkpoints/best_model.pt",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many classes to show.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    image_path = Path(args.image_path).resolve()

    config = load_json(experiment_dir / "config_used.json")
    class_to_idx, idx_to_class = load_class_mapping(experiment_dir / "class_mapping.json")

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint is not None
        else (experiment_dir / "checkpoints" / "best_model.pt")
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()

    model = create_model(
        model_name=str(config["model_name"]),
        num_classes=len(class_to_idx),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    _, eval_transform = build_transforms(
        image_size=int(config["image_size"]),
        augmentation_mode="none",
    )

    with Image.open(image_path) as img:
        image_tensor = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(int(args.top_k), len(idx_to_class))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Top predictions:")

    for rank, (prob, index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        class_name = idx_to_class[index]
        print(f"{rank:>2}. {class_name:<30} {prob:.4f}")


if __name__ == "__main__":
    main()
