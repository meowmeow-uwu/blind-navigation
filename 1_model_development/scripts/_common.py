"""Helpers shared by the Phase A scripts (paths, hashing, manifest, versions)."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 1_model_development/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "models.yaml"
WEIGHTS_DIR = ROOT / "weights"
PYTORCH_DIR = WEIGHTS_DIR / "pytorch"
METADATA_DIR = WEIGHTS_DIR / "metadata"
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"
SUMMARIES_DIR = ROOT / "benchmarks" / "summaries"

# Terminal states recorded in metadata / manifest.
STATUS_VALIDATED = "validated"
STATUS_UNSUPPORTED = "unsupported"
STATUS_DOWNLOAD_FAILED = "download_failed"
STATUS_LOAD_FAILED = "load_failed"

# COCO class ids every compared checkpoint must agree on.
COCO_KEY_CLASSES = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3}
EXPECTED_CLASS_COUNT = 80


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_names_hash(names: dict[int, str]) -> str:
    """Stable hash of the {class_id: class_name} taxonomy."""
    canonical = json.dumps(
        {str(k): v for k, v in sorted(names.items())},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def framework_versions() -> dict[str, str | None]:
    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except Exception:  # noqa: BLE001
        ultralytics_version = None
    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        torch_version = None
    return {
        "ultralytics": ultralytics_version,
        "torch": torch_version,
        "python": platform.python_version(),
    }


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_manifest() -> dict | None:
    if not MANIFEST_PATH.is_file():
        return None
    return load_json(MANIFEST_PATH)


def update_manifest(entries: dict[str, dict]) -> dict:
    """Merge `entries` into weights/manifest.json (creating it if needed)."""
    manifest = load_manifest() or {"created_at": now_iso(), "models": {}}
    manifest["updated_at"] = now_iso()
    manifest.setdefault("models", {}).update(entries)
    write_json(MANIFEST_PATH, manifest)
    return manifest


def resolve_manifest_checkpoint(manifest: dict, name: str) -> Path | None:
    """Return the checkpoint path recorded in the manifest, or None."""
    entry = manifest.get("models", {}).get(name)
    if not entry or not entry.get("checkpoint"):
        return None
    return WEIGHTS_DIR / entry["checkpoint"]


def resolve_manifest_metadata(manifest: dict, name: str) -> dict | None:
    entry = manifest.get("models", {}).get(name)
    if not entry or not entry.get("metadata"):
        return None
    path = WEIGHTS_DIR / entry["metadata"]
    return load_json(path) if path.is_file() else None
