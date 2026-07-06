"""Post-training classifier evaluation — confusion matrix and classification report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import load_class_names
from src.data.pytorch_datasets import load_split_datasets
from src.models.classifier import build_classifier


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier_resnet50.yaml")
    args = parser.parse_args()

    root = _project_root()
    with open(root / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open(root / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    clf_params = {**params["classifier"], **config.get("overrides", {})}
    image_size = params["preprocess"]["image_size"]
    class_names = load_class_names(root / "data" / "class_names.json")

    _, test_ds = load_split_datasets(root / "data" / "processed" / "manifest.json", image_size=image_size)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_classifier(clf_params["model"], num_classes=len(class_names), pretrained=False)
    weights = root / "models" / "classifier" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Train first: {weights}")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            out = model(images)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())

    present = sorted(set(labels) | set(preds))
    report = classification_report(
        labels, preds, labels=present, target_names=[class_names[i] for i in present], output_dict=False
    )
    print(report)

    out_dir = root / "reports" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm[:20, :20], cmap="Purples")
    ax.set_title("Confusion matrix (first 20 classes)")
    fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved evaluation reports -> {out_dir}")


if __name__ == "__main__":
    main()
