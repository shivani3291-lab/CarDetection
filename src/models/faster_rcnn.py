"""Faster R-CNN detection model wrapper."""

from __future__ import annotations

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int = 197, pretrained: bool = True) -> fasterrcnn_resnet50_fpn:
    """num_classes includes background (196 car classes + 1 background)."""
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    return model
