#!/usr/bin/env bash
set -euo pipefail

echo "=== Phase 0/1: Data setup ==="
python scripts/setup_data.py --skip-extract || python scripts/setup_data.py
python -m src.data.preprocess
python -m src.data.validate
python scripts/run_eda.py

echo "=== Phase 2: Classifier training ==="
python -m src.train.classifier_train --config configs/classifier_custom_cnn.yaml
python -m src.train.classifier_train --config configs/classifier_resnet50.yaml

echo "=== Phase 3: Detection ==="
python -m src.train.detection_train --config configs/detection_fasterrcnn.yaml
python scripts/generate_yolo_dataset.py
python scripts/train_yolo.py

echo "=== Phase 4: MLOps ==="
python -m src.mlops.export_onnx
python -m src.mlops.promote_model
python -m src.mlops.drift_monitor

echo "=== Phase 5: Evaluation ==="
python -m src.evaluate.evaluate_classifier --config configs/classifier_resnet50.yaml

echo "=== Done. Launch app: streamlit run app.py ==="
