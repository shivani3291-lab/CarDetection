"""Promote model to Production if it beats the current champion."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest_metric(client: MlflowClient, experiment_name: str, metric_key: str) -> tuple[float | None, str | None]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None, None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_key} DESC", "start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, None
    value = runs[0].data.metrics.get(metric_key)
    return value, runs[0].info.run_id


def get_production_metric(client: MlflowClient, model_name: str, metric_key: str) -> float | None:
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception:
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get(metric_key)


def main() -> None:
    root = _project_root()
    with open(root / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    mlops = params["mlops"]
    margin = mlops["promotion_margin"]
    task = mlops.get("promotion_task", "classification")

    if task == "detection":
        experiment = "car_detection"
        metric_key = "mAP@0.5"
        if metric_key not in mlops.get("metric_aliases", {}):
            metric_key = "mAP@0.5"
    else:
        experiment = "car_classification"
        metric_key = "best_val_top1"
        if metric_key not in mlops:
            metric_key = "val_top1"

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(root / "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    new_metric, run_id = _latest_metric(client, experiment, metric_key)
    if new_metric is None:
        new_metric, run_id = _latest_metric(client, experiment, "val_top1")
        metric_key = "val_top1"
    if new_metric is None or run_id is None:
        print("No runs found - skipping promotion")
        return

    model_registry_name = "CarDetector"
    prod_metric = get_production_metric(client, model_registry_name, metric_key)

    if prod_metric is None or new_metric > prod_metric + margin:
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_registry_name)
        client.transition_model_version_stage(
            name=model_registry_name,
            version=result.version,
            stage="Production",
        )
        prev = f"{prod_metric:.3f}" if prod_metric is not None else "none"
        print(f"Promoted: {prev} -> {new_metric:.3f} ({metric_key})")
    else:
        print(f"No promotion: {new_metric:.3f} did not beat {prod_metric:.3f} (+{margin})")


if __name__ == "__main__":
    main()
