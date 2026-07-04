"""Bootstrap dataset on a fresh machine."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from src.data.dataset import load_class_names, project_root


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {dest} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print("Extraction complete.")


def ensure_class_names(root: Path) -> Path:
    out = root / "data" / "class_names.json"
    if out.exists():
        return out

    csv_path = root / "Car names and make.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing class names file: {csv_path}")

    import json

    names = load_class_names(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"classes": names}, f, indent=2)
    print(f"Created {out}")
    return out


def find_zip(root: Path) -> Path | None:
    for candidate in [root / "Car Images.zip", root / "data" / "Car Images.zip"]:
        if candidate.exists():
            return candidate
    return None


def data_ready(root: Path) -> bool:
    raw = root / "data" / "raw"
    return (raw / "Car Images" / "Train Images").exists() or (raw / "Train Images").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Stanford Cars dataset on a new machine")
    parser.add_argument("--zip", type=str, default=None, help="Path to Car Images.zip")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction if data exists")
    args = parser.parse_args()

    root = project_root()
    raw_dir = root / "data" / "raw"

    ensure_class_names(root)

    if data_ready(root):
        print("Dataset already present in data/raw/")
    elif args.skip_extract:
        raise FileNotFoundError("data/raw/ is empty. Provide --zip or extract Car Images.zip manually.")
    else:
        zip_path = Path(args.zip) if args.zip else find_zip(root)
        if zip_path is None:
            raise FileNotFoundError(
                "Car Images.zip not found. Place it in the project root or pass --zip PATH"
            )
        extract_zip(zip_path, raw_dir)

    print("Next steps:")
    print("  python -m src.data.preprocess")
    print("  python -m src.data.validate")


if __name__ == "__main__":
    main()
