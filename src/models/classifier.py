"""Transfer-learning classifiers — ResNet-50 and EfficientNet-B3."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_classifier(
    model_name: str,
    num_classes: int = 196,
    pretrained: bool = True,
) -> nn.Module:
    weights = "DEFAULT" if pretrained else None

    if model_name == "custom_cnn":
        from src.models.cnn import CarCNN

        return CarCNN(num_classes=num_classes)

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "efficientnet_b3":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "custom_cnn":
        for param in model.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
