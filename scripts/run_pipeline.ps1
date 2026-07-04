# Full pipeline runner (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== Phase 0/1: Data setup ==="
python scripts/setup_data.py --skip-extract
if ($LASTEXITCODE -ne 0) { python scripts/setup_data.py }
python -m src.data.preprocess
python -m src.data.validate
python scripts/run_eda.py

Write-Host "=== Phase 2: Classifier training ==="
python -m src.train.classifier_train --config configs/classifier_custom_cnn.yaml
python -m src.train.classifier_train --config configs/classifier_resnet50.yaml

Write-Host "=== Phase 3: Detection ==="
python -m src.train.detection_train --config configs/detection_fasterrcnn.yaml
python scripts/generate_yolo_dataset.py
python scripts/train_yolo.py

Write-Host "=== Phase 4: MLOps ==="
python -m src.mlops.export_onnx
python -m src.mlops.promote_model
python -m src.mlops.drift_monitor

Write-Host "=== Phase 5: Evaluation ==="
python -m src.evaluate.evaluate_classifier --config configs/classifier_resnet50.yaml

Write-Host "=== Done. Launch app: streamlit run app.py ==="
