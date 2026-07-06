"""Fine-tune the classifier on accumulated user feedback.

Meant to be run periodically (see scripts/run_pipeline.ps1's sibling task
scheduler entry, or invoke directly). Each run:

1. Reads every feedback record logged since the last run
   (data/feedback/feedback_log.jsonl, cursor tracked in
   data/feedback/retrain_state.json).
2. Keeps only records that resolve to one of the 196 known classes -
   confirmed-correct predictions, plus "incorrect" reports where the user
   picked the right answer from the dropdown. Free-text "Other" corrections
   that aren't a known class are skipped (the classifier has no output slot
   for a class it's never seen) and left for manual review.
3. Skips entirely if there aren't enough new usable samples yet
   (params.yaml: mlops.feedback_retrain.min_new_samples) - a couple of
   labels shouldn't move a 196-way classifier, and this avoids grinding
   through a full retrain for nothing.
4. Fine-tunes the current champion checkpoint (not from scratch) for a few
   epochs at a low learning rate, then evaluates on the untouched manifest
   test split. Only overwrites the serving model (models/classifier/best.pt
   -> re-exported ONNX) if the fine-tuned checkpoint's top-1 accuracy is at
   least as good as the checkpoint it started from - never silently ships a
   regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Run directly (`python scripts/retrain_from_feedback.py`), e.g. from a scheduled
# task with no interactive shell/cwd setup - the project root isn't on sys.path
# by default the way it is for `python -m src...` invocations, so add it explicitly.
_root = _project_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.serve.feedback import read_feedback  # noqa: E402


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _state_path(root: Path) -> Path:
    return root / "data" / "feedback" / "retrain_state.json"


def load_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.exists():
        return {"consumed_count": 0, "last_retrain_at": None, "last_result": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(root: Path, state: dict[str, Any]) -> None:
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def select_usable_feedback(
    records: list[dict[str, Any]], class_names: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split feedback records into (usable, skipped_no_label, skipped_out_of_taxonomy).

    usable entries are {"image_path": ..., "class_name": ...}, ready to become
    training samples. Pure function, no I/O - easy to test without a GPU/ML env.
    """
    known = {c.lower(): c for c in class_names}
    usable: list[dict[str, str]] = []
    skipped_no_label: list[dict[str, Any]] = []
    skipped_out_of_taxonomy: list[dict[str, Any]] = []

    for r in records:
        if r.get("is_correct"):
            usable.append({"image_path": r["image_path"], "class_name": r["predicted_class"]})
            continue

        corrected = r.get("corrected_class")
        if not corrected:
            skipped_no_label.append(r)
        elif corrected.lower() in known:
            usable.append({"image_path": r["image_path"], "class_name": known[corrected.lower()]})
        else:
            skipped_out_of_taxonomy.append(r)

    return usable, skipped_no_label, skipped_out_of_taxonomy


def _run_finetune(
    root: Path,
    usable: list[dict[str, str]],
    class_names: list[str],
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    """Heavy-import path: only touched once the gate above says a real retrain
    is warranted, so `--dry-run` never needs torch/mlflow installed."""
    import mlflow
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader

    from src.data.dataset import CarSample, load_manifest, resolve_image_path
    from src.data.pytorch_datasets import CarClassificationDataset
    from src.evaluate.metrics import evaluate_classifier
    from src.mlops.export_onnx import main as export_onnx_main
    from src.models.classifier import build_classifier
    from src.train.classifier_train import train_one_epoch

    params = _load_yaml(root / "params.yaml")
    clf_params = params["classifier"]
    aug_params = params.get("augmentation", {})
    image_size = params["preprocess"]["image_size"]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    manifest_path = root / "data" / "processed" / "manifest.json"
    samples = load_manifest(manifest_path)
    train_samples = [s for s in samples if s.split == "train"]
    test_samples = [s for s in samples if s.split == "test"]
    if not test_samples:
        raise RuntimeError("No test split in the manifest - can't evaluate a retrain safely.")

    feedback_samples = []
    for item in usable:
        resolved = resolve_image_path(item["image_path"], root=root)
        with Image.open(resolved) as img:
            w, h = img.size
        feedback_samples.append(
            CarSample(
                image_path=str(resolved),
                class_id=class_to_id[item["class_name"]],
                class_name=item["class_name"],
                bbox=(0.0, 0.0, float(w), float(h)),
                split="train",
            )
        )

    weights_path = root / "models" / "classifier" / "best.pt"
    if not weights_path.exists():
        raise RuntimeError(f"No champion checkpoint at {weights_path} to fine-tune from.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = clf_params["model"]
    model = build_classifier(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.to(device)

    test_ds = CarClassificationDataset(test_samples, image_size=image_size, train=False)
    test_loader = DataLoader(test_ds, batch_size=clf_params["batch_size"], shuffle=False)

    baseline_top1 = evaluate_classifier(model, test_loader, device)["top1"]

    train_ds = CarClassificationDataset(
        train_samples + feedback_samples, image_size=image_size, train=True, aug_params=aug_params
    )
    train_loader = DataLoader(
        train_ds, batch_size=clf_params["batch_size"], shuffle=True, num_workers=clf_params.get("num_workers", 0)
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=clf_params.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=clf_params.get("weight_decay", 0.01))

    mlflow.set_experiment("car_classification")
    best_top1 = baseline_top1
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    with mlflow.start_run(run_name=f"feedback-retrain-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        mlflow.log_params({
            "source": "feedback_retrain",
            "num_new_samples": len(feedback_samples),
            "epochs": epochs,
            "lr": lr,
            "baseline_top1": baseline_top1,
        })
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            metrics = evaluate_classifier(model, test_loader, device)
            mlflow.log_metrics({"train_loss": train_loss, "val_top1": metrics["top1"]}, step=epoch)
            print(f"[feedback-retrain] epoch {epoch + 1}/{epochs} loss={train_loss:.4f} top1={metrics['top1']:.3f}")
            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        promoted = best_top1 >= baseline_top1
        mlflow.log_metric("new_top1", best_top1)
        mlflow.log_metric("promoted", int(promoted))

        if promoted:
            torch.save(best_state, weights_path)
            export_onnx_main()
        else:
            candidate_path = root / "models" / "classifier" / "feedback_candidate.pt"
            torch.save(best_state, candidate_path)
            print(f"[feedback-retrain] did not beat baseline - candidate saved to {candidate_path}, champion untouched.")

    return {"baseline_top1": baseline_top1, "new_top1": best_top1, "promoted": promoted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, change nothing.")
    parser.add_argument("--force", action="store_true", help="Retrain even if below min_new_samples.")
    args = parser.parse_args()

    root = _project_root()
    params = _load_yaml(root / "params.yaml")
    feedback_cfg = params["mlops"]["feedback_retrain"]

    with open(root / "data" / "class_names.json", encoding="utf-8") as f:
        class_data = json.load(f)
    class_names = class_data["classes"] if isinstance(class_data, dict) else class_data

    state = load_state(root)
    records = read_feedback()
    new_records = records[state["consumed_count"]:]

    usable, skipped_no_label, skipped_out_of_taxonomy = select_usable_feedback(new_records, class_names)

    print(
        f"[feedback-retrain] {len(new_records)} new feedback record(s) since last run: "
        f"{len(usable)} usable, {len(skipped_no_label)} skipped (no label), "
        f"{len(skipped_out_of_taxonomy)} skipped (not in the 196 classes)."
    )

    min_new = feedback_cfg["min_new_samples"]
    if not args.force and len(usable) < min_new:
        print(f"[feedback-retrain] below min_new_samples ({min_new}) - skipping, cursor left unchanged.")
        return

    if args.dry_run:
        print(f"[feedback-retrain] dry run - would fine-tune on {len(usable)} sample(s). Nothing changed.")
        return

    result = _run_finetune(
        root, usable, class_names, epochs=feedback_cfg["epochs"], lr=feedback_cfg["lr"]
    )

    state["consumed_count"] = len(records)
    state["last_retrain_at"] = datetime.now(timezone.utc).isoformat()
    state["last_result"] = {**result, "usable_samples": len(usable)}
    save_state(root, state)

    verdict = "promoted to production" if result["promoted"] else "kept as candidate (no improvement)"
    print(
        f"[feedback-retrain] done - baseline_top1={result['baseline_top1']:.3f} "
        f"new_top1={result['new_top1']:.3f} -> {verdict}"
    )


if __name__ == "__main__":
    main()
