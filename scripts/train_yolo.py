"""Train YOLOv8 on the Stanford Cars YOLO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.dataset import project_root
from src.models.yolo import train_yolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stanford_cars.yaml")
    args = parser.parse_args()

    root = project_root()
    config_path = root / args.config
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Run: python scripts/generate_yolo_dataset.py"
        )

    with open(root / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["yolo"]

    save_dir = train_yolo(
        data_yaml=config_path,
        model_name=params["model"],
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch"],
    )

    metrics_path = root / "models" / "yolo" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results_csv = save_dir / "results.csv"
    metrics = {"save_dir": str(save_dir)}
    if results_csv.exists():
        import pandas as pd

        df = pd.read_csv(results_csv)
        if "metrics/mAP50(B)" in df.columns:
            metrics["mAP@0.5"] = float(df["metrics/mAP50(B)"].iloc[-1])
        if "metrics/mAP50-95(B)" in df.columns:
            metrics["mAP@0.5:0.95"] = float(df["metrics/mAP50-95(B)"].iloc[-1])

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"YOLO training complete -> {save_dir}")
    print(f"Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
