"""YOLOv8 training wrapper."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def train_yolo(
    data_yaml: str | Path,
    model_name: str = "yolov8m.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "yolov8m_stanford",
) -> Path:
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
    )
    return Path(results.save_dir)
