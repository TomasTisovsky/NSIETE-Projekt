from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("resnet18", "efficientnet_b0")


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}. Use one of {SUPPORTED_MODELS}.")

    if freeze_backbone:
        freeze_backbone_parameters(model, model_name)

    return model


def freeze_backbone_parameters(model: nn.Module, model_name: str) -> None:
    if model_name == "resnet18":
        head_prefixes = ("fc.",)
    elif model_name == "efficientnet_b0":
        head_prefixes = ("classifier.1",)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for name, param in model.named_parameters():
        is_head = any(name.startswith(prefix) for prefix in head_prefixes)
        param.requires_grad = is_head


def unfreeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
