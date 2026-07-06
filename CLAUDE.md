# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CarVision AI: an end-to-end deep learning system for the Stanford Cars dataset (196 classes, ~16K images), covering data preprocessing, classification (CNN/ResNet50/EfficientNet), detection (Faster R-CNN, YOLOv8), MLOps (DVC + MLflow + ONNX export + drift monitoring), and a multi-page Streamlit dashboard for serving.

Note: the repo root as seen by tools is `CarDetection/`, but the actual project (git repo, source, configs) lives one level down in `CarDetection/CarDetection/`. Run all commands from that inner directory.

## Common commands

```bash
# Environment setup
conda env create -f environment.yml && conda activate car-detect
# or: pip install -r requirements.txt

# Data pipeline (must run in this order)
python scripts/setup_data.py --zip "Car Images.zip"   # extracts zip into data/raw/
python -m src.data.preprocess                          # builds data/processed/manifest.json
python -m src.data.validate                             # writes validation_report.json
python scripts/run_eda.py                                # writes reports/eda/

# Or drive the same stages via DVC (reads params.yaml, respects deps/outs in dvc.yaml)
dvc repro

# Training
python -m src.train.classifier_train --config configs/classifier_custom_cnn.yaml
python -m src.train.classifier_train --config configs/classifier_resnet50.yaml
python -m src.train.classifier_train --config configs/classifier_efficientnet.yaml
python -m src.train.detection_train --config configs/detection_fasterrcnn.yaml
python scripts/generate_yolo_dataset.py && python scripts/train_yolo.py
mlflow ui   # view runs logged under mlruns/ (or MLFLOW_TRACKING_URI)

# Evaluation, export, MLOps
python -m src.evaluate.evaluate_classifier
python -m src.mlops.export_onnx          # requires trained classifier weights
python -m src.mlops.promote_model        # promotes to MLflow "Production" if it beats current champion by params.yaml:mlops.promotion_margin
python -m src.mlops.drift_monitor

# Serving
streamlit run app.py

# Full pipeline in one shot
powershell -File scripts/run_pipeline.ps1   # Windows
bash scripts/run_pipeline.sh                # Linux/macOS

# Lint and test (same as CI)
ruff check src/ tests/ app.py pages/ scripts/
pytest tests/ -v -k "not integration"       # unit-only, matches CI
pytest tests/ -v                             # includes integration tests that need a real dataset
pytest tests/test_dataset.py::test_car_cnn_forward   # single test
```

Integration tests (`test_build_manifest_integration`, etc.) self-skip via `pytest.skip(...)` when `data/raw/Car Images/Train Images` isn't present — no dataset means those tests are silently skipped rather than failing.

## Architecture

**Config layering:** two YAML sources are always merged at training time. `params.yaml` holds shared/default hyperparameters per stage (`preprocess`, `augmentation`, `classifier`, `detection`, `yolo`, `mlops`). Each file under `configs/*.yaml` holds an experiment name, run name, and an `overrides` dict that is shallow-merged on top of the matching `params.yaml` section (see `classifier_train.py`: `{**params["classifier"], **config.get("overrides", {})}`). When changing a hyperparameter, decide whether it's a global default (`params.yaml`) or an experiment-specific override (`configs/*.yaml`).

**Path resolution is root-relative and reconstructible.** `src/data/dataset.py:project_root()` derives the repo root from `__file__`, so scripts work regardless of cwd. Manifests store POSIX-relative image paths (`resolve_image_path` tries several candidate roots including `data/raw/`) rather than absolute paths, so `data/processed/manifest.json` stays portable across machines/CI. Bounding boxes default to full-image (`_full_image_bbox`) when no `.mat`/JSON annotation is found — the provided Stanford Cars zip ships without annotation files, so this is the common path, not a fallback edge case.

**Pipeline stages are declared twice, deliberately:** once as plain Python entrypoints (`python -m src.data.preprocess`, invoked directly by README/CI) and once as DVC stages in `dvc.yaml` (adds dep/param/output tracking and caching via `dvc repro`). Keep both in sync when adding a stage — a new script needs a corresponding `dvc.yaml` stage if it should participate in `dvc repro` and DVC caching.

**Serving supports two independent model families with automatic fallback** (`src/serve/inference.py:Predictor`, mode `"auto"`): first tries an ONNX classifier (`models/car_detector.onnx`, single top-1 label + full-image bbox), then falls back to a PyTorch Faster R-CNN checkpoint (`models/detection/best.pt`, real multi-box detection). `MODEL_PATH`/`CLASS_NAMES_PATH` env vars (or `MODEL_URL`/`st.secrets["MODEL_URL"]` for on-demand download) override the defaults — this is how Streamlit Cloud deploys without committing large weight files to git.

**MLflow experiment/metric naming is a load-bearing convention.** `promote_model.py` looks up runs by experiment name (`car_classification` or `car_detection`) and a specific metric key (`best_val_top1`/`val_top1` for classification, `mAP@0.5` for detection) to decide whether to promote a run to the `CarDetector` registry's `Production` stage. Training scripts must log under these exact experiment/metric names for promotion to find them.

**Streamlit app is multi-page via the `pages/` convention**: `app.py` is the landing page; `pages/1_live_detection.py`, `2_upload_detect.py`, `3_analytics.py`, `4_about.py` are auto-registered by Streamlit's file-based routing. Shared state/model loading lives in `src/serve/session.py` (sidebar model status) and `src/serve/inference.py` (the `Predictor`), not duplicated per page.

**CI (`\.github/workflows/ml_pipeline.yml`)** runs lint + unit tests (`-k "not integration"`) on every push/PR, then on pushes only, conditionally reproduces the DVC preprocess/validate stages (only if a dataset zip or `data/raw/` is present) and attempts model promotion (non-fatal if it fails — no dataset/MLflow server in CI by default).
