"""PyTorch Dataset wrappers for classification and detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentations import get_classifier_transform, get_train_transform, get_val_transform
from src.data.dataset import CarSample, load_manifest, resolve_image_path


def _read_rgb(image_path: str) -> np.ndarray:
    resolved = resolve_image_path(image_path)
    bgr = cv2.imread(str(resolved))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {resolved}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class CarClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[CarSample],
        image_size: int = 224,
        train: bool = True,
        aug_params: dict | None = None,
    ):
        self.samples = samples
        self.transform = get_classifier_transform(image_size, train=train, aug_params=aug_params)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = _read_rgb(sample.image_path)
        tensor = self.transform(image=image)["image"]
        return tensor, sample.class_id


class CarDetectionDataset(Dataset):
    def __init__(
        self,
        samples: list[CarSample],
        image_size: int = 640,
        train: bool = True,
        aug_params: dict | None = None,
    ):
        self.samples = samples
        self.image_size = image_size
        self.transform = (
            get_train_transform(image_size, aug_params) if train else get_val_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        image = _read_rgb(sample.image_path)
        x1, y1, x2, y2 = sample.bbox

        transformed = self.transform(
            image=image,
            bboxes=[[x1, y1, x2, y2]],
            class_labels=[sample.class_id],
        )
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels + 1,
            "image_id": torch.tensor([idx]),
            "area": (
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                if len(boxes)
                else torch.zeros(0)
            ),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return transformed["image"], target


def load_split_datasets(
    manifest_path: str | Path,
    image_size: int = 224,
    aug_params: dict | None = None,
) -> tuple[CarClassificationDataset, CarClassificationDataset]:
    samples = load_manifest(manifest_path)
    train_samples = [s for s in samples if s.split == "train"]
    test_samples = [s for s in samples if s.split == "test"]
    train_ds = CarClassificationDataset(
        train_samples, image_size=image_size, train=True, aug_params=aug_params
    )
    test_ds = CarClassificationDataset(test_samples, image_size=image_size, train=False)
    return train_ds, test_ds


def detection_collate_fn(batch: list) -> tuple:
    images, targets = zip(*batch)
    return list(images), list(targets)
