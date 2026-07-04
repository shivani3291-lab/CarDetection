"""Faster R-CNN detection training with MLflow tracking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import load_class_names, load_manifest
from src.data.pytorch_datasets import CarDetectionDataset, detection_collate_fn
from src.evaluate.detection_metrics import evaluate_detection_map, measure_inference_latency
from src.models.faster_rcnn import build_faster_rcnn


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/detection_fasterrcnn.yaml")
    parser.add_argument("--max-eval-samples", type=int, default=500, help="Cap mAP eval for speed")
    args = parser.parse_args()

    root = _project_root()
    with open(root / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(root / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    det_params = {**params["detection"], **config.get("overrides", {})}
    num_classes = len(load_class_names(root / "data" / "class_names.json")) + 1

    manifest_path = root / "data" / "processed" / "manifest.json"
    samples = load_manifest(manifest_path)
    train_samples = [s for s in samples if s.split == "train"]
    test_samples = [s for s in samples if s.split == "test"]

    train_ds = CarDetectionDataset(train_samples, image_size=640, train=True)
    test_ds = CarDetectionDataset(test_samples, image_size=640, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=det_params["batch_size"],
        shuffle=True,
        num_workers=det_params.get("num_workers", 0),
        collate_fn=detection_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=det_params["lr"], momentum=0.9, weight_decay=0.0005)

    output_dir = root / "models" / "detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"

    mlflow.set_experiment(config.get("experiment_name", "car_detection"))
    best_map = 0.0

    with mlflow.start_run(run_name=config.get("run_name", "fasterrcnn")):
        mlflow.log_params(det_params)

        for epoch in range(det_params["epochs"]):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)
            epoch_time = time.time() - t0

            model.eval()
            map_metrics = evaluate_detection_map(
                model, test_loader, device, max_samples=args.max_eval_samples
            )

            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "epoch_time_s": epoch_time,
                    "mAP@0.5": map_metrics["mAP@0.5"],
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch + 1}/{det_params['epochs']} - loss={avg_loss:.4f} "
                f"mAP@0.5={map_metrics['mAP@0.5']:.3f}"
            )

            if map_metrics["mAP@0.5"] > best_map:
                best_map = map_metrics["mAP@0.5"]
                torch.save(model.state_dict(), best_path)

        if not best_path.exists():
            torch.save(model.state_dict(), best_path)

        model.eval()
        sample_images, _ = next(iter(test_loader))
        latency_ms = measure_inference_latency(model, sample_images, device)
        mlflow.log_metrics({"best_mAP@0.5": best_map, "inference_latency_ms": latency_ms})
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

        metrics_path = output_dir / "metrics.json"
        import json
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"mAP@0.5": best_map, "inference_latency_ms": latency_ms}, f, indent=2)

        print(f"Saved detection model -> {best_path} (mAP@0.5={best_map:.3f}, latency {latency_ms:.1f} ms)")


if __name__ == "__main__":
    main()
