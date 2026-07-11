"""Classifier training with MLflow tracking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from torch.utils.data import DataLoader

from src.data.dataset import load_class_names
from src.data.pytorch_datasets import load_split_datasets
from src.evaluate.metrics import evaluate_classifier
from src.models.classifier import build_classifier, freeze_backbone, unfreeze_all
from src.models.cnn import count_parameters


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier_resnet50.yaml")
    args = parser.parse_args()

    root = _project_root()
    config = _load_config(root / args.config)

    with open(root / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    clf_params = {**params["classifier"], **config.get("overrides", {})}
    aug_params = params.get("augmentation", {})
    image_size = params["preprocess"]["image_size"]

    manifest_path = root / "data" / "processed" / "manifest.json"
    train_ds, val_ds = load_split_datasets(
        manifest_path, image_size=image_size, aug_params=aug_params
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=clf_params["batch_size"],
        shuffle=True,
        num_workers=clf_params.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=clf_params["batch_size"],
        shuffle=False,
        num_workers=clf_params.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = clf_params["model"]
    num_classes = len(load_class_names(root / "data" / "class_names.json"))

    model = build_classifier(model_name, num_classes=num_classes).to(device)
    freeze_epochs = clf_params.get("freeze_epochs", 5)
    epochs = clf_params["epochs"]
    label_smoothing = clf_params.get("label_smoothing", 0.0)

    if model_name != "custom_cnn" and freeze_epochs > 0:
        freeze_backbone(model, model_name)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    output_dir = root / "models" / "classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=clf_params["lr"],
        weight_decay=clf_params.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    mlflow.set_experiment(config.get("experiment_name", "car_classification"))
    best_top1 = 0.0
    best_path = output_dir / "best.pt"

    with mlflow.start_run(run_name=config.get("run_name", model_name)):
        mlflow.log_params({
            "model": model_name,
            "lr": clf_params["lr"],
            "batch_size": clf_params["batch_size"],
            "epochs": epochs,
            "freeze_epochs": freeze_epochs,
            "num_parameters": count_parameters(model),
        })

        for epoch in range(epochs):
            if epoch == freeze_epochs and model_name != "custom_cnn":
                unfreeze_all(model)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=clf_params["lr"],
                    weight_decay=clf_params.get("weight_decay", 0.01),
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(epochs - epoch, 1)
                )

            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            metrics = evaluate_classifier(model, val_loader, device)
            scheduler.step()
            epoch_time = time.time() - t0

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_top1": metrics["top1"],
                    "val_top5": metrics["top5"],
                    "epoch_time_s": epoch_time,
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - loss={train_loss:.4f} "
                f"top1={metrics['top1']:.3f} top5={metrics['top5']:.3f}"
            )

            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                torch.save(model.state_dict(), best_path)

        if not best_path.exists():
            torch.save(model.state_dict(), best_path)

        mlflow.log_metric("best_val_top1", best_top1)

        model.eval()
        example_input = torch.randn(1, 3, image_size, image_size)
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 3, image_size, image_size))]),
            outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes))]),
        )
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            signature=signature,
            input_example=example_input.numpy(),
        )
        print(f"Saved best model (top1={best_top1:.3f}) -> {best_path}")


if __name__ == "__main__":
    main()
