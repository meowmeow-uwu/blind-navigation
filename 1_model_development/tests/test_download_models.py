import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from download_models import process_model  # noqa: E402
from models.model_registry import ModelSpec  # noqa: E402


def test_unknown_official_checkpoint_is_marked_unsupported_without_renaming():
    spec = ModelSpec(
        name="yolo99n",
        checkpoint="yolo99n.pt",
        role="test",
        input_size=320,
        task="detect",
        enabled=True,
    )
    record = process_model(spec)

    assert record["status"] == "unsupported"
    assert record["checkpoint"] == "yolo99n.pt"  # never renamed
    assert "yolo99n.pt" in record["error"]
    assert record["framework"]["ultralytics"]
    assert record["framework"]["python"]
    assert record["framework"]["torch"]
    assert not (ROOT / "weights" / "pytorch" / "yolo99n.pt").exists()
