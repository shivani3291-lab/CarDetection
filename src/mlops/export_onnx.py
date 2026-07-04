"""Export trained classifier to ONNX for Streamlit serving."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.data.dataset import load_class_names
from src.models.classifier import build_classifier


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _project_root()
    with open(root / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    model_name = params["classifier"]["model"]
    num_classes = len(load_class_names(root / "data" / "class_names.json"))
    image_size = params["preprocess"]["image_size"]

    weights_path = root / "models" / "classifier" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            "Run classifier training first: python -m src.train.classifier_train"
        )

    model = build_classifier(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    output_path = root / "models" / "car_detector.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    _ = session.run(None, {"image": dummy.numpy()})

    metadata = {
        "model_type": "onnx_classifier",
        "architecture": model_name,
        "description": f"{model_name} classifier (ONNX)",
        "num_classes": num_classes,
        "image_size": image_size,
    }
    det_meta = root / "models" / "detection" / "metrics.json"
    if det_meta.exists():
        with open(det_meta, encoding="utf-8") as f:
            det = json.load(f)
        metadata["mAP@0.5"] = det.get("mAP@0.5")

    with open(root / "models" / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported and verified ONNX model -> {output_path}")


if __name__ == "__main__":
    main()
