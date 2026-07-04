"""Stanford Cars dataset loading and class mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

Split = Literal["train", "test"]


@dataclass
class CarSample:
    image_path: str
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (Pascal VOC)
    split: Split


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_class_names(class_names_path: str | Path) -> list[str]:
    path = Path(class_names_path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["classes"] if isinstance(data, dict) else data

    classes: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    return classes


def _raw_root(data_dir: Path) -> Path:
    candidates = [
        data_dir / "Car Images",
        data_dir,
    ]
    for candidate in candidates:
        if (candidate / "Train Images").exists():
            return candidate
    raise FileNotFoundError(f"Could not find Train Images under {data_dir}")


def _full_image_bbox(image_path: Path) -> tuple[float, float, float, float]:
    with Image.open(image_path) as img:
        w, h = img.size
    return (0.0, 0.0, float(w), float(h))


def resolve_image_path(image_path: str, root: Path | None = None) -> Path:
    """Resolve manifest path (relative or legacy absolute) to an existing file."""
    root = root or project_root()
    path = Path(image_path)

    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            root / path,
            root / "data" / "raw" / path,
        ])
    else:
        try:
            rel = path.relative_to(root)
            candidates.append(root / rel)
        except ValueError:
            pass

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {image_path}")


def build_manifest(
    data_dir: str | Path,
    class_names_path: str | Path,
    annotations_dir: str | Path | None = None,
    project_root_path: Path | None = None,
) -> list[CarSample]:
    """Build image -> class -> bbox manifest from folder structure."""
    data_dir = Path(data_dir)
    root = project_root_path or project_root()
    class_names = load_class_names(class_names_path)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    raw_root = _raw_root(data_dir)

    split_dirs = {
        "train": raw_root / "Train Images",
        "test": raw_root / "Test Images",
    }

    samples: list[CarSample] = []
    for split, split_dir in split_dirs.items():
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in class_to_id:
                continue
            class_id = class_to_id[class_name]
            for image_path in sorted(class_dir.glob("*.jpg")):
                bbox = _resolve_bbox(image_path, annotations_dir)
                rel_path = image_path.relative_to(root).as_posix()
                samples.append(
                    CarSample(
                        image_path=rel_path,
                        class_id=class_id,
                        class_name=class_name,
                        bbox=bbox,
                        split=split,  # type: ignore[arg-type]
                    )
                )
    return samples


def _resolve_bbox(
    image_path: Path,
    annotations_dir: str | Path | None,
) -> tuple[float, float, float, float]:
    if annotations_dir is None:
        return _full_image_bbox(image_path)

    ann_path = Path(annotations_dir) / f"{image_path.stem}.json"
    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)
        return tuple(ann["bbox"])  # type: ignore[return-value]
    return _full_image_bbox(image_path)


def manifest_to_records(samples: list[CarSample]) -> list[dict]:
    return [
        {
            "image_path": s.image_path,
            "class_id": s.class_id,
            "class_name": s.class_name,
            "bbox": list(s.bbox),
            "split": s.split,
        }
        for s in samples
    ]


def save_manifest(samples: list[CarSample], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_to_records(samples), f, indent=2)


def load_manifest(manifest_path: str | Path, root: Path | None = None) -> list[CarSample]:
    root = root or project_root()
    with open(manifest_path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        CarSample(
            image_path=str(resolve_image_path(r["image_path"], root=root)),
            class_id=r["class_id"],
            class_name=r["class_name"],
            bbox=tuple(r["bbox"]),
            split=r["split"],
        )
        for r in records
    ]


def split_counts(samples: list[CarSample]) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "test": 0}
    for s in samples:
        counts[s.split] += 1
    return counts
