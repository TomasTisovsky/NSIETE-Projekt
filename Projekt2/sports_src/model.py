"""PyTorch CNN models for sports image classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SportsCNN(nn.Module):
    """Compact convolutional neural network for image classification.

    The network keeps the input as a 3D image tensor for as long as
    possible, which is substantially more appropriate than flattening
    images into vectors before learning.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@dataclass(frozen=True)
class ModelConfig:
    """Lightweight config for CNN construction."""

    num_classes: int
    dropout: float = 0.3


def build_cnn_classifier(num_classes: int, dropout: float = 0.3) -> SportsCNN:
    """Factory helper for the CNN used in Projekt2."""

    return SportsCNN(num_classes=num_classes, dropout=dropout)
