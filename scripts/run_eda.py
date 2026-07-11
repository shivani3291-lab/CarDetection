"""Generate EDA plots and save to reports/."""

from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt

from src.data.dataset import load_manifest, project_root
from src.serve.viz import draw_boxes, plot_class_distribution


def main() -> None:
    root = project_root()
    manifest_path = root / "data" / "processed" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run preprocess first: python -m src.data.preprocess")

    samples = load_manifest(manifest_path, root=root)
    train = [s for s in samples if s.split == "train"]
    reports = root / "reports" / "eda"
    reports.mkdir(parents=True, exist_ok=True)

    counts = Counter(s.class_name for s in train)
    plot_class_distribution(dict(counts), save_path=str(reports / "class_distribution.png"))
    plt.close()

    widths, heights = [], []
    for s in train[:1000]:
        x1, y1, x2, y2 = s.bbox
        widths.append(x2 - x1)
        heights.append(y2 - y1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(widths, bins=30, color="#7F77DD")
    axes[0].set_title("BBox width")
    axes[1].hist(heights, bins=30, color="#7F77DD")
    axes[1].set_title("BBox height")
    fig.savefig(reports / "bbox_distribution.png", bbox_inches="tight", dpi=120)
    plt.close()

    from PIL import Image

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for ax, s in zip(axes.flat, train[:25]):
        img = Image.open(s.image_path).convert("RGB")
        ann = draw_boxes(img, [{"class": s.class_name, "score": 1.0, "bbox": list(s.bbox)}])
        ax.imshow(ann)
        ax.axis("off")
    fig.suptitle("Sample grid with ground-truth bboxes")
    fig.savefig(reports / "sample_grid.png", bbox_inches="tight", dpi=120)
    plt.close()

    summary = {
        "total": len(samples),
        "train": len(train),
        "test": len(samples) - len(train),
        "num_classes": len(counts),
    }
    with open(reports / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"EDA reports saved -> {reports}")


if __name__ == "__main__":
    main()
