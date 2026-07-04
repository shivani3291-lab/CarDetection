"""Convert manifest to YOLO format for ultralytics training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from PIL import Image

from src.data.dataset import load_class_names, load_manifest, project_root, resolve_image_path


def _unique_stem(class_id: int, class_name: str, original_stem: str) -> str:
    safe = class_name.replace(" ", "_").replace("/", "-")[:40]
    digest = hashlib.md5(f"{class_id}_{original_stem}".encode()).hexdigest()[:8]
    return f"{class_id:03d}_{safe}_{original_stem}_{digest}"


def main() -> None:
    root = project_root()
    manifest_path = root / "data" / "processed" / "manifest.json"
    class_names = load_class_names(root / "data" / "class_names.json")
    samples = load_manifest(manifest_path, root=root)

    yolo_root = root / "data" / "yolo"
    for split in ("train", "val"):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        yolo_split = "train" if sample.split == "train" else "val"
        src = resolve_image_path(sample.image_path, root=root)
        stem = _unique_stem(sample.class_id, sample.class_name, src.stem)
        dst_img = yolo_root / "images" / yolo_split / f"{stem}.jpg"
        label_path = yolo_root / "labels" / yolo_split / f"{stem}.txt"

        if not dst_img.exists():
            import shutil

            shutil.copy2(src, dst_img)

        x1, y1, x2, y2 = sample.bbox
        with Image.open(src) as img:
            w, h = img.size
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{sample.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    config = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    out_yaml = root / "configs" / "stanford_cars.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"YOLO dataset ready -> {yolo_root} ({len(samples)} samples)")
    print(f"Config updated -> {out_yaml}")


if __name__ == "__main__":
    main()
