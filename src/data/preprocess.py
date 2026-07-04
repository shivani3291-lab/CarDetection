"""DVC preprocess stage — build manifest and class index."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from src.data.dataset import build_manifest, load_class_names, save_manifest, split_counts


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    params_path = _project_root() / "params.yaml"
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    root = _project_root()
    params = _load_params()

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_names_src = root / "data" / "class_names.json"
    if not class_names_src.exists():
        csv_path = root / "Car names and make.csv"
        if csv_path.exists():
            names = load_class_names(csv_path)
            with open(class_names_src, "w", encoding="utf-8") as f:
                json.dump({"classes": names}, f, indent=2)

    samples = build_manifest(raw_dir, class_names_src, project_root_path=root)
    manifest_path = processed_dir / "manifest.json"
    save_manifest(samples, manifest_path)

    counts = split_counts(samples)
    summary = {
        "total_samples": len(samples),
        "train": counts["train"],
        "test": counts["test"],
        "num_classes": len(load_class_names(class_names_src)),
        "image_size": params["preprocess"]["image_size"],
    }
    with open(processed_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    dest_names = processed_dir / "class_names.json"
    if not dest_names.exists():
        shutil.copy(class_names_src, dest_names)

    print(f"Preprocessed {len(samples)} samples -> {manifest_path}")
    print(f"Train: {counts['train']}, Test: {counts['test']}")


if __name__ == "__main__":
    main()
