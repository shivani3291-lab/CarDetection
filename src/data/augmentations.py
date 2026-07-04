"""Albumentations transforms for classification and detection."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _color_augment(p: float) -> A.BasicTransform:
    return A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=p,
    )


def get_train_transform(image_size: int = 224, aug_params: dict | None = None) -> A.Compose:
    params = aug_params or {}
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=params.get("horizontal_flip_p", 0.5)),
            A.RandomBrightnessContrast(p=params.get("brightness_contrast_p", 0.3)),
            _color_augment(params.get("color_jitter_p", 0.2)),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


def get_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


def get_classifier_transform(image_size: int = 224, train: bool = True, aug_params: dict | None = None) -> A.Compose:
    if train:
        params = aug_params or {}
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=params.get("horizontal_flip_p", 0.5)),
                A.RandomBrightnessContrast(p=params.get("brightness_contrast_p", 0.3)),
                _color_augment(params.get("color_jitter_p", 0.2)),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
