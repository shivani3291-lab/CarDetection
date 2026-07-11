"""Great Expectations validation gate for the cars dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from PIL import Image

from src.data.dataset import load_class_names, load_manifest, project_root, resolve_image_path


def _load_validation_params() -> dict:
    params_path = project_root() / "params.yaml"
    if not params_path.exists():
        return {"min_train_class_coverage": 0.99, "fail_on_error": True}
    with open(params_path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get("validation", {"min_train_class_coverage": 0.99, "fail_on_error": True})


def validate_manifest(
    manifest_path: Path, class_names_path: Path, val_params: dict | None = None
) -> dict:
    val_params = val_params or _load_validation_params()
    root = project_root()
    samples = load_manifest(manifest_path, root=root)
    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)

    errors: list[str] = []
    warnings: list[str] = []
    train_class_ids: set[int] = set()
    missing_images = 0
    corrupt_images = 0
    bbox_violations = 0

    for sample in samples:
        try:
            path = resolve_image_path(sample.image_path, root=root)
        except FileNotFoundError:
            missing_images += 1
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w == 0 or h == 0:
                    corrupt_images += 1
                    continue
        except Exception:
            corrupt_images += 1
            continue

        x1, y1, x2, y2 = sample.bbox
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
            bbox_violations += 1

        if sample.split == "train":
            train_class_ids.add(sample.class_id)

    if missing_images:
        errors.append(f"{missing_images} missing image files")
    if corrupt_images:
        errors.append(f"{corrupt_images} corrupt images")
    if bbox_violations:
        errors.append(f"{bbox_violations} bbox coordinate violations")

    coverage = len(train_class_ids) / num_classes if num_classes else 0.0
    min_coverage = val_params.get("min_train_class_coverage", 0.99)
    if coverage < min_coverage:
        errors.append(
            f"Train class coverage {coverage:.1%} below threshold {min_coverage:.1%} "
            f"({len(train_class_ids)}/{num_classes} classes)"
        )
    elif len(train_class_ids) < num_classes:
        warnings.append(
            f"Only {len(train_class_ids)}/{num_classes} classes in train split "
            "(within coverage threshold)"
        )

    ge_results = _run_great_expectations(samples, class_names, root)
    if ge_results.get("failed_expectations"):
        for item in ge_results["failed_expectations"]:
            errors.append(f"GE: {item}")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "total_samples": len(samples),
        "num_classes_expected": num_classes,
        "train_classes_found": len(train_class_ids),
        "train_class_coverage": coverage,
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "bbox_violations": bbox_violations,
        "warnings": warnings,
        "errors": errors,
        "great_expectations": ge_results,
    }


def _run_great_expectations(samples: list, class_names: list[str], root: Path) -> dict:
    """Run Great Expectations checks when available; fall back to equivalent pandas checks."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "class_id": [s.class_id for s in samples],
            "split": [s.split for s in samples],
            "bbox_area": [(s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]) for s in samples],
        }
    )

    failed: list[str] = []

    try:
        from great_expectations.dataset import PandasDataset

        ge_df = PandasDataset(df)
        expectations = [
            ("class_id range", ge_df.expect_column_values_to_be_between(
                "class_id", min_value=0, max_value=len(class_names) - 1
            )),
            ("split values", ge_df.expect_column_values_to_be_in_set(
                "split", value_set=["train", "test"]
            )),
            ("bbox_area positive", ge_df.expect_column_values_to_be_between(
                "bbox_area", min_value=0
            )),
        ]
        for name, result in expectations:
            if not result.success:
                failed.append(name)
        return {
            "engine": "great_expectations",
            "failed_expectations": failed,
            "checks_run": len(expectations),
        }
    except Exception:
        if df["class_id"].min() < 0 or df["class_id"].max() >= len(class_names):
            failed.append("class_id out of range")
        if not set(df["split"].unique()).issubset({"train", "test"}):
            failed.append("invalid split values")
        if (df["bbox_area"] < 0).any():
            failed.append("negative bbox area")
        return {"engine": "pandas_fallback", "failed_expectations": failed, "checks_run": 3}


def main() -> None:
    root = project_root()
    manifest_path = root / "data" / "processed" / "manifest.json"
    class_names_path = root / "data" / "class_names.json"
    val_params = _load_validation_params()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Run preprocess first: {manifest_path} not found")

    report = validate_manifest(manifest_path, class_names_path, val_params)
    out_path = root / "data" / "processed" / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    status = "PASSED" if report["passed"] else "FAILED"
    print(f"Validation {status}: {out_path}")
    for warn in report.get("warnings", []):
        print(f"  warning: {warn}")
    for err in report["errors"]:
        print(f"  error: {err}")

    if not report["passed"] and val_params.get("fail_on_error", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
