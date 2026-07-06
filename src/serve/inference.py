"""Model loading and prediction for Streamlit serving."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.serve.preprocessing import preprocess

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ONNX = _PROJECT_ROOT / "models" / "car_detector.onnx"
_DEFAULT_DETECTION = _PROJECT_ROOT / "models" / "detection" / "best.pt"
_DEFAULT_CLASSES = _PROJECT_ROOT / "data" / "class_names.json"
_METADATA_PATH = _PROJECT_ROOT / "models" / "model_metadata.json"


def _resolve_model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", _DEFAULT_ONNX))


def _resolve_class_names_path() -> Path:
    return Path(os.environ.get("CLASS_NAMES_PATH", _DEFAULT_CLASSES))


def _download_model_if_needed(model_path: Path) -> None:
    if model_path.exists():
        return
    url = os.environ.get("MODEL_URL")
    try:
        import streamlit as st

        url = url or st.secrets.get("MODEL_URL")
    except Exception:
        pass
    if not url:
        return

    import requests

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def load_class_names(path: Path | None = None) -> list[str]:
    path = path or _resolve_class_names_path()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["classes"] if isinstance(data, dict) else data


def load_model_metadata() -> dict:
    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_onnx_session(model_path: Path | None = None):
    import onnxruntime as ort

    path = model_path or _resolve_model_path()
    _download_model_if_needed(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train a classifier and run export_onnx, "
            "or set MODEL_URL / st.secrets['MODEL_URL']."
        )
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_detection_model(num_classes: int = 197):
    import torch

    from src.models.faster_rcnn import build_faster_rcnn

    if not _DEFAULT_DETECTION.exists():
        return None
    model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(_DEFAULT_DETECTION, map_location="cpu", weights_only=True))
    model.eval()
    return model


class Predictor:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.class_names = load_class_names()
        self.metadata = load_model_metadata()
        self._onnx_session = None
        self._detection_model = None
        self._torch_device = None

        if mode in ("auto", "onnx"):
            try:
                self._onnx_session = load_onnx_session()
                self.mode = "onnx"
            except FileNotFoundError:
                pass

        if self._onnx_session is None and mode in ("auto", "detection"):
            det = load_detection_model(len(self.class_names) + 1)
            if det is not None:
                import torch

                self._detection_model = det
                self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._detection_model.to(self._torch_device)
                self.mode = "detection"

    @property
    def loaded(self) -> bool:
        return self._onnx_session is not None or self._detection_model is not None

    def predict(self, image: Image.Image, conf_threshold: float = 0.5, image_size: int = 224) -> list[dict[str, Any]]:
        if self._onnx_session is not None:
            return predict_classifier(image, self._onnx_session, self.class_names, conf_threshold, image_size)
        if self._detection_model is not None:
            return predict_detection(image, self._detection_model, self.class_names, self._torch_device, conf_threshold)
        return []

    def predict_topk(self, image: Image.Image, k: int = 5, image_size: int = 224) -> list[dict[str, Any]]:
        """Ranked top-k guesses, classifier mode only (empty list otherwise)."""
        if self._onnx_session is not None:
            return predict_classifier_topk(image, self._onnx_session, self.class_names, k, image_size)
        return []


def load_predictor() -> Predictor:
    return Predictor(mode="auto")


def predict_classifier(
    image: Image.Image,
    session,
    class_names: list[str],
    conf_threshold: float = 0.5,
    image_size: int = 224,
) -> list[dict[str, Any]]:
    tensor = preprocess(image, image_size=image_size)
    outputs = session.run(None, {"image": tensor})
    logits = outputs[0][0]
    probs = _softmax(logits)
    top_idx = int(np.argmax(probs))
    score = float(probs[top_idx])

    if score < conf_threshold:
        return []

    w, h = image.size
    return [{"class": class_names[top_idx], "score": score, "bbox": [0.0, 0.0, float(w), float(h)]}]


def predict_classifier_topk(
    image: Image.Image,
    session,
    class_names: list[str],
    k: int = 5,
    image_size: int = 224,
) -> list[dict[str, Any]]:
    """Top-k ranked guesses (no confidence threshold - always returns k candidates)."""
    tensor = preprocess(image, image_size=image_size)
    outputs = session.run(None, {"image": tensor})
    logits = outputs[0][0]
    probs = _softmax(logits)
    top_indices = np.argsort(probs)[::-1][:k]
    return [{"class": class_names[i], "score": float(probs[i])} for i in top_indices]


def predict_detection(
    image: Image.Image,
    model,
    class_names: list[str],
    device,
    conf_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    import torch
    import torchvision.transforms.functional as TF

    img_tensor = TF.to_tensor(image.convert("RGB"))
    with torch.no_grad():
        outputs = model([img_tensor.to(device)])[0]

    results = []
    w, h = image.size
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        score_f = float(score)
        if score_f < conf_threshold:
            continue
        cls_idx = int(label) - 1
        if cls_idx < 0 or cls_idx >= len(class_names):
            continue
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "class": class_names[cls_idx],
            "score": score_f,
            "bbox": [x1, y1, x2, y2],
        })
    return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
