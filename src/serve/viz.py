"""Visualisation helpers for EDA and Streamlit."""

from __future__ import annotations

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(image: Image.Image, results: list[dict[str, Any]], color: tuple = (127, 119, 221)) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{r['class']} {r['score']:.0%}"
        draw.text((x1, max(0, y1 - 14)), label, fill=color)
    return annotated


def draw_boxes_cv(image_bgr: np.ndarray, boxes: list, labels: list[str], scores: list[float]) -> np.ndarray:
    out = image_bgr.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (221, 119, 127), 2)
        cv2.putText(out, f"{label} {score:.0%}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (221, 119, 127), 1)
    return out


def plot_class_distribution(class_counts: dict[str, int], save_path: str | None = None) -> plt.Figure:
    names = list(class_counts.keys())
    counts = list(class_counts.values())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(names)), counts, color="#7F77DD")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def render_3d_car(rotation_y: int = 25) -> str:
    return f"""
    <div style="perspective: 700px; display:flex; justify-content:center;">
      <div style="width:200px; height:120px; transform-style:preserve-3d;
                  transform: rotateX(-16deg) rotateY({rotation_y}deg);
                  transition: transform 0.3s;">
        <div style="position:absolute; width:200px; height:80px; top:20px;
                    border:1px solid #7F77DD; border-radius:8px;
                    background:rgba(127,119,221,0.15); transform: translateZ(35px);"></div>
        <div style="position:absolute; width:200px; height:80px; top:20px;
                    border:1px solid #7F77DD; border-radius:8px;
                    background:rgba(127,119,221,0.15); transform: translateZ(-35px);"></div>
      </div>
    </div>
    """
