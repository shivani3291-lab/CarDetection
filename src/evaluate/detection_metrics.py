"""Detection evaluation — mAP and inference latency."""

from __future__ import annotations

import time
from collections import defaultdict

import torch


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(box1[0], box2[:, 0])
    y1 = torch.max(box1[1], box2[:, 1])
    x2 = torch.min(box1[2], box2[:, 2])
    y2 = torch.min(box1[3], box2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def evaluate_detection_map(
    model,
    data_loader,
    device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Compute mAP@0.5 for single-class-per-image detection."""
    model.eval()
    per_class_tp: dict[int, list[float]] = defaultdict(list)
    per_class_fp: dict[int, list[float]] = defaultdict(list)
    per_class_gt: dict[int, int] = defaultdict(int)

    n_seen = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"].to(device)
            gt_labels = (target["labels"] - 1).tolist()
            for label in gt_labels:
                per_class_gt[int(label)] += 1

            pred_boxes = output["boxes"]
            pred_labels = (output["labels"] - 1).tolist()
            pred_scores = output["scores"].tolist()

            matched_gt: set[int] = set()
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score < score_threshold:
                    continue
                label = int(label)
                if len(gt_boxes) == 0:
                    per_class_fp[label].append(float(score))
                    continue

                ious = box_iou(box, gt_boxes)
                best_iou, best_idx = float(ious.max()), int(ious.argmax())
                if best_iou >= iou_threshold and best_idx not in matched_gt:
                    matched_gt.add(best_idx)
                    per_class_tp[label].append(float(score))
                else:
                    per_class_fp[label].append(float(score))

        n_seen += len(images)
        if max_samples and n_seen >= max_samples:
            break

    aps = []
    for cls_id in set(list(per_class_gt.keys()) + list(per_class_tp.keys())):
        tp = sorted(per_class_tp.get(cls_id, []), reverse=True)
        fp = sorted(per_class_fp.get(cls_id, []), reverse=True)
        n_gt = per_class_gt.get(cls_id, 0)
        if n_gt == 0:
            continue
        scores = sorted(tp + fp, reverse=True)
        tp_cum, fp_cum = 0, 0
        precisions, recalls = [], []
        tp_set = set(tp)
        for s in scores:
            if s in tp_set:
                tp_cum += 1
                tp_set.remove(s)
            else:
                fp_cum += 1
            precisions.append(tp_cum / max(tp_cum + fp_cum, 1))
            recalls.append(tp_cum / n_gt)
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += (max(prec_at_recall) if prec_at_recall else 0.0) / 11
        aps.append(ap)

    map50 = sum(aps) / len(aps) if aps else 0.0
    return {"mAP@0.5": map50, "num_classes_evaluated": len(aps)}


@torch.no_grad()
def measure_inference_latency(model, sample_input: list, device, n_warmup: int = 5, n_runs: int = 20) -> float:
    model.eval()
    images = [img.to(device) for img in sample_input]

    for _ in range(n_warmup):
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000
