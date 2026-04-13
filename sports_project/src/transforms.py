from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, augmentation_mode: str = "moderate_aug"):
    train_steps = [transforms.Resize((image_size, image_size))]

    if augmentation_mode == "none":
        pass

    elif augmentation_mode == "brightness_aug":
        train_steps.extend(
            [
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]
        )

    elif augmentation_mode == "moderate_aug":
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        )

    else:
        raise ValueError(
            f"Unknown augmentation_mode: {augmentation_mode}. "
            "Use one of: none, brightness_aug, moderate_aug"
        )

    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_transform = transforms.Compose(train_steps)
    return train_transform, eval_transform