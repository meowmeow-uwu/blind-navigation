import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_registry import (  # noqa: E402
    DisabledModelError,
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
)

CONFIG = ROOT / "configs" / "models.yaml"


def test_load_registry():
    registry = ModelRegistry(CONFIG)
    assert len(registry.names()) >= 1


def test_registry_contains_yolov8n():
    spec = ModelRegistry(CONFIG).get("yolov8n")
    assert isinstance(spec, ModelSpec)
    assert spec.checkpoint == "yolov8n.pt"
    assert spec.role == "baseline"
    assert spec.enabled is True


def test_get_unknown_model_raises_error():
    registry = ModelRegistry(CONFIG)
    with pytest.raises(UnknownModelError) as exc:
        registry.get("yolo_does_not_exist")
    assert "yolo_does_not_exist" in str(exc.value)
    assert "yolov8n" in str(exc.value)  # lists known models


def test_disabled_model_is_not_returned_by_default(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "defaults:\n  input_size: 320\n"
        "models:\n"
        "  a:\n    checkpoint: a.pt\n    enabled: true\n"
        "  b:\n    checkpoint: b.pt\n    enabled: false\n",
        encoding="utf-8",
    )
    registry = ModelRegistry(cfg)
    assert registry.names() == ["a"]
    assert registry.names(include_disabled=True) == ["a", "b"]
    with pytest.raises(DisabledModelError):
        registry.get("b")
    assert registry.get("b", include_disabled=True).enabled is False


def test_default_input_size_is_applied(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "defaults:\n  task: detect\n  input_size: 416\n"
        "models:\n"
        "  a:\n    checkpoint: a.pt\n"
        "  b:\n    checkpoint: b.pt\n    input_size: 640\n",
        encoding="utf-8",
    )
    registry = ModelRegistry(cfg)
    assert registry.get("a").input_size == 416
    assert registry.get("a").task == "detect"
    assert registry.get("b").input_size == 640
