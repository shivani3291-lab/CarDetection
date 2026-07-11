"""Evidently AI data drift monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset import load_manifest, project_root


def build_reference_dataframe(manifest_path: Path) -> pd.DataFrame:
    samples = load_manifest(manifest_path, root=project_root())
    return pd.DataFrame(
        {
            "class_name": [s.class_name for s in samples if s.split == "train"],
        }
    )


def build_current_dataframe(predictions: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_name": [p["class"] for p in predictions],
            "confidence": [p["score"] for p in predictions],
        }
    )


def build_demo_current_from_test(manifest_path: Path, limit: int = 500) -> pd.DataFrame:
    samples = load_manifest(manifest_path, root=project_root())
    test_samples = [s for s in samples if s.split == "test"][:limit]
    return pd.DataFrame({"class_name": [s.class_name for s in test_samples]})


def run_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: Path
) -> dict:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    return report.as_dict()


def main() -> None:
    root = project_root()
    manifest_path = root / "data" / "processed" / "manifest.json"
    reference_df = build_reference_dataframe(manifest_path)
    current_df = build_demo_current_from_test(manifest_path)

    report_dict = run_drift_report(reference_df, current_df, root / "reports" / "drift_report.html")
    drifted = report_dict["metrics"][0]["result"].get("dataset_drift", False)
    print(f"Drift report saved. Dataset drift detected: {drifted}")


if __name__ == "__main__":
    main()
