"""Classification metrics — top-1 and top-5 accuracy."""

from __future__ import annotations

import torch


@torch.no_grad()
def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    return correct[:, :k].any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate_classifier(model, dataloader, device, k: int = 5) -> dict[str, float]:
    model.eval()
    top1_total = 0.0
    top5_total = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        top1_total += top_k_accuracy(outputs, labels, k=1)
        top5_total += top_k_accuracy(outputs, labels, k=k)
        n_batches += 1

    if n_batches == 0:
        return {"top1": 0.0, "top5": 0.0}

    return {
        "top1": top1_total / n_batches,
        "top5": top5_total / n_batches,
    }
