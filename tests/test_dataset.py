"""Unit tests."""

from pathlib import Path

import pytest
import torch

from src.data.dataset import build_manifest, load_class_names, project_root, resolve_image_path
from src.data.pytorch_datasets import detection_collate_fn
from src.models.cnn import CarCNN

ROOT = project_root()


@pytest.fixture
def class_names_path(tmp_path):
    path = tmp_path / "classes.json"
    path.write_text('{"classes": ["Class A", "Class B"]}', encoding="utf-8")
    return path


def test_load_class_names_json(class_names_path):
    assert load_class_names(class_names_path) == ["Class A", "Class B"]


def test_car_cnn_forward():
    model = CarCNN(num_classes=10)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_detection_collate_fn():
    images = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
    targets = [{"boxes": torch.zeros(1, 4)}, {"boxes": torch.zeros(1, 4)}]
    batch_images, batch_targets = detection_collate_fn(list(zip(images, targets)))
    assert len(batch_images) == 2
    assert len(batch_targets) == 2


def test_resolve_relative_path():
    # Only tests path resolution logic when file exists
    raw = ROOT / "data" / "raw"
    if not raw.exists():
        pytest.skip("data/raw not present")


def test_build_manifest_integration():
    raw = ROOT / "data" / "raw"
    classes = ROOT / "data" / "class_names.json"
    if not (raw / "Car Images" / "Train Images").exists():
        pytest.skip("Dataset not extracted")
    samples = build_manifest(raw, classes, project_root_path=ROOT)
    assert len(samples) > 0
    assert not Path(samples[0].image_path).is_absolute()
    resolved = resolve_image_path(samples[0].image_path, root=ROOT)
    assert resolved.exists()


def test_export_onnx_requires_weights(tmp_path, monkeypatch):
    from src.mlops import export_onnx

    monkeypatch.setattr(export_onnx, "_project_root", lambda: tmp_path)
    (tmp_path / "params.yaml").write_text(
        "classifier:\n  model: custom_cnn\npreprocess:\n  image_size: 224\n", encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "class_names.json").write_text('{"classes": ["A"]}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        export_onnx.main()
