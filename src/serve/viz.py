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


def render_3d_car(label: str | None = None, score: float | None = None) -> str:
    """A continuously auto-rotating low-poly 3D car (real CSS 3D transforms, not an image).

    With no args it shows a generic default car; pass a predicted class name
    (and optionally its confidence score) to label the same rotating car with
    a detection result instead.
    """
    label_html = ""
    if label:
        pct = f" &middot; {score:.1%}" if score is not None else ""
        label_html = f'<div class="car3d-label">{label}{pct}</div>'

    return f"""
    <style>
      .car3d-stage {{
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
        overflow: hidden;
        font-family: -apple-system, "Segoe UI", sans-serif;
        box-sizing: border-box;
      }}
      .car3d-rig {{
        perspective: 900px;
        width: 100%;
        max-width: 190px;
        height: 130px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }}
      .car3d {{
        position: relative;
        width: 170px;
        height: 66px;
        transform-style: preserve-3d;
        animation: car3d-spin 10s linear infinite;
      }}
      @keyframes car3d-spin {{
        from {{ transform: rotateX(-10deg) rotateY(0deg); }}
        to   {{ transform: rotateX(-10deg) rotateY(360deg); }}
      }}
      .car3d .face {{ position: absolute; border: 1px solid rgba(255,255,255,0.08); }}
      .car3d .front, .car3d .back {{
        width: 170px; height: 66px;
        border-radius: 18px 18px 6px 6px;
        background: linear-gradient(180deg, #2b3542 0%, #141a21 65%, #0e1319 100%);
      }}
      .car3d .front {{ transform: translateZ(42px); }}
      .car3d .back {{
        transform: rotateY(180deg) translateZ(42px);
        background: linear-gradient(180deg, #232c36 0%, #10151b 65%, #0c1015 100%);
      }}
      .car3d .left, .car3d .right {{
        width: 84px; height: 66px;
        border-radius: 18px 34px 8px 8px;
        background: linear-gradient(180deg, #333f4c 0%, #1b232c 55%, #0f1318 100%);
      }}
      .car3d .right {{ transform: rotateY(90deg) translateZ(84px); }}
      .car3d .left {{
        transform: rotateY(-90deg) translateZ(84px);
        border-radius: 34px 18px 8px 8px;
      }}
      .car3d .top {{
        width: 170px; height: 84px;
        border-radius: 24px;
        background: linear-gradient(90deg, #38424e 0%, #48555f 45%, #38424e 100%);
        transform: rotateX(90deg) translateZ(33px);
      }}
      .car3d .bottom {{
        width: 170px; height: 84px;
        background: #05070a;
        transform: rotateX(-90deg) translateZ(33px);
      }}
      .car3d .band {{
        position: absolute; left: 9%; right: 9%; top: 8px; height: 19px;
        border-radius: 9px 9px 3px 3px;
        background: linear-gradient(180deg, rgba(120,200,255,0.22), rgba(10,14,18,0.55));
        border: 1px solid rgba(255,255,255,0.08);
      }}
      .car3d .left .band, .car3d .right .band {{ left: 14%; right: 14%; }}
      .car3d .lamp {{ position: absolute; bottom: 12px; width: 24px; height: 7px; border-radius: 4px; }}
      .car3d .lamp.l {{ left: 8px; }}
      .car3d .lamp.r {{ right: 8px; }}
      .car3d .front .lamp {{ background: #52e3d4; box-shadow: 0 0 10px 2px rgba(82,227,212,0.5); }}
      .car3d .back .lamp {{ background: #ffb84d; box-shadow: 0 0 8px 2px rgba(255,184,77,0.45); }}
      .car3d .wheel {{
        position: absolute; bottom: 3px; width: 22px; height: 22px; border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #555, #0a0c0e 70%);
        border: 2px solid #05070a;
      }}
      .car3d .wheel.a {{ left: 6px; }}
      .car3d .wheel.b {{ right: 6px; }}
      .car3d-ground {{
        width: 150px; height: 14px; margin-top: 8px;
        background: radial-gradient(ellipse at center, rgba(82,227,212,0.18), rgba(82,227,212,0) 72%);
        filter: blur(1px);
      }}
      .car3d-label {{
        margin-top: 10px; font-size: 12.5px; font-weight: 600; color: #52e3d4;
        font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
        text-align: center; letter-spacing: 0.01em;
      }}
      @media (prefers-reduced-motion: reduce) {{ .car3d {{ animation: none; }} }}
    </style>
    <div class="car3d-stage">
      <div class="car3d-rig">
        <div class="car3d">
          <div class="face front"><div class="band"></div><div class="lamp l"></div><div class="lamp r"></div></div>
          <div class="face back"><div class="band"></div><div class="lamp l"></div><div class="lamp r"></div></div>
          <div class="face left"><div class="band"></div><div class="wheel a"></div><div class="wheel b"></div></div>
          <div class="face right"><div class="band"></div><div class="wheel a"></div><div class="wheel b"></div></div>
          <div class="face top"></div>
          <div class="face bottom"></div>
        </div>
      </div>
      <div class="car3d-ground"></div>
      {label_html}
    </div>
    """
