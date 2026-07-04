# Model Card — CarDetector v1.0

## Model details

- **Architecture:** ResNet-50 classifier (ONNX serving) / Faster R-CNN ResNet-50 FPN / YOLOv8m
- **Task:** Multi-class car classification and bounding-box detection
- **Classes:** 196 (Make, Model, Year)

## Training data

- Stanford Cars Dataset (Krause et al., ICCV 2013)
- 8,103 training images, 8,000 test images (from provided zip)
- Class names in `data/class_names.json`

## Evaluation metrics (targets)

| Model | Val Top-1 | Val Top-5 | mAP@0.5 | Inference |
|---|---|---|---|---|
| Custom CNN | ~45% | ~72% | — | — |
| ResNet-50 FT | ~82% | ~95% | — | — |
| EfficientNet-B3 FT | ~85% | ~96% | — | — |
| Faster R-CNN | — | — | ~0.72 | ~120ms |
| YOLOv8m | — | — | ~0.69 | ~18ms |

Run `python -m src.evaluate.evaluate_classifier` after training for actual metrics.

## Intended use

Automotive surveillance, parking management, traffic analysis, portfolio demonstration.

## Limitations

- May degrade on night or heavily occluded images
- 196 classes cover the 2012 Stanford Cars release
- Provided zip has no `.mat` bbox files; full-image boxes used unless annotations are added under `data/raw/annotations/`

## Reproduce on a new machine

```bash
python scripts/setup_data.py --zip "Car Images.zip"
python -m src.data.preprocess
python -m src.data.validate
python -m src.train.classifier_train --config configs/classifier_resnet50.yaml
python -m src.mlops.export_onnx
streamlit run app.py
```
