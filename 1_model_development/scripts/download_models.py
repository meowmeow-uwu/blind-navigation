"""Download official Ultralytics checkpoints and record metadata + manifest.

Usage (from 1_model_development/):
    python scripts/download_models.py --models yolov8n
    python scripts/download_models.py --models yolov8n yolo11n yolo12n yolo26n

Each model is processed independently: a failure never stops the others.
Resulting status per model: validated | unsupported | download_failed | load_failed.
Checkpoints are never renamed or replaced with unofficial files.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CONFIG_PATH,
    METADATA_DIR,
    PYTORCH_DIR,
    STATUS_DOWNLOAD_FAILED,
    STATUS_LOAD_FAILED,
    STATUS_UNSUPPORTED,
    STATUS_VALIDATED,
    WEIGHTS_DIR,
    class_names_hash,
    framework_versions,
    sha256_file,
    update_manifest,
    write_json,
)
from models.model_registry import (  # noqa: E402
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
)

# Error fragments that mean "this Ultralytics version does not know the architecture"
# (as opposed to a corrupt file or other load problem).
_UNSUPPORTED_MARKERS = (
    "Can't get attribute",
    "No module named 'ultralytics",
    "newer version of ultralytics",
    "is not supported",
)


def build_record(spec: ModelSpec, status: str, error: str | None = None, **extra) -> dict:
    record = {
        "name": spec.name,
        "role": spec.role,
        "checkpoint": spec.checkpoint,
        "source": "official",
        "sha256": None,
        "file_size_bytes": None,
        "file_size_mb": None,
        "parameters": None,
        "task": spec.task,
        "class_count": None,
        "class_names_hash": None,
        "input_size": spec.input_size,
        "framework": framework_versions(),
        "status": status,
        "error": error,
    }
    record.update(extra)
    return record


def format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def official_asset_names() -> set[str]:
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    return set(GITHUB_ASSETS_NAMES)


def official_release() -> str:
    import inspect

    from ultralytics.utils.downloads import attempt_download_asset

    return inspect.signature(attempt_download_asset).parameters["release"].default


def process_model(spec: ModelSpec, force: bool = False) -> dict:
    """Download + validate one model. Always returns a metadata record."""
    ckpt_path = PYTORCH_DIR / spec.checkpoint
    fw = framework_versions()
    print(f"\n=== {spec.name} ({spec.checkpoint}) ===")

    # 1. Does this Ultralytics version recognise the official checkpoint name?
    try:
        names = official_asset_names()
    except Exception as exc:  # noqa: BLE001
        msg = f"Cannot import Ultralytics download API: {format_exception(exc)}"
        print(f"[download_failed] {msg}")
        return build_record(spec, STATUS_DOWNLOAD_FAILED, msg, error_traceback=traceback.format_exc())

    if spec.checkpoint not in names:
        msg = (
            f"Checkpoint '{spec.checkpoint}' is not an official asset known to "
            f"ultralytics=={fw['ultralytics']} (python {fw['python']}, torch {fw['torch']}). "
            "Upgrade ultralytics or keep this model unsupported; "
            "checkpoint name is intentionally NOT remapped."
        )
        print(f"[unsupported] {msg}")
        return build_record(spec, STATUS_UNSUPPORTED, msg)

    # 2. Download through the official API.
    release = official_release()
    source_url = f"https://github.com/ultralytics/assets/releases/download/{release}/{spec.checkpoint}"
    try:
        PYTORCH_DIR.mkdir(parents=True, exist_ok=True)
        if force and ckpt_path.exists():
            ckpt_path.unlink()
        if ckpt_path.exists():
            print(f"Checkpoint already present: {ckpt_path}")
        else:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(ckpt_path.resolve(), release=release)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Download finished without producing {ckpt_path} (source: {source_url})"
            )
    except Exception as exc:  # noqa: BLE001
        msg = format_exception(exc)
        print(f"[download_failed] {msg}")
        return build_record(
            spec,
            STATUS_DOWNLOAD_FAILED,
            msg,
            source_url=source_url,
            error_traceback=traceback.format_exc(),
        )

    # 3. Hash the file that was saved.
    sha256 = sha256_file(ckpt_path)
    size_bytes = ckpt_path.stat().st_size
    print(f"sha256={sha256}  size={size_bytes / 1e6:.2f} MB")

    # 4. Load it back.
    try:
        from ultralytics import YOLO

        model = YOLO(str(ckpt_path))
        if model.task != spec.task:
            raise ValueError(
                f"Checkpoint task is '{model.task}', but models.yaml expects '{spec.task}'"
            )
        class_map = {int(k): str(v) for k, v in model.names.items()}
        parameters = int(sum(p.numel() for p in model.model.parameters()))
    except Exception as exc:  # noqa: BLE001
        msg = format_exception(exc)
        tb = traceback.format_exc()
        unsupported = any(marker in tb for marker in _UNSUPPORTED_MARKERS)
        status = STATUS_UNSUPPORTED if unsupported else STATUS_LOAD_FAILED
        print(f"[{status}] {msg}")
        return build_record(
            spec,
            status,
            msg,
            sha256=sha256,
            file_size_bytes=size_bytes,
            file_size_mb=round(size_bytes / (1024 * 1024), 2),
            source_url=source_url,
            error_traceback=tb,
        )

    print(f"[validated] params={parameters:,} classes={len(class_map)}")
    return build_record(
        spec,
        STATUS_VALIDATED,
        None,
        sha256=sha256,
        file_size_bytes=size_bytes,
        file_size_mb=round(size_bytes / (1024 * 1024), 2),
        parameters=parameters,
        class_count=len(class_map),
        class_names_hash=class_names_hash(class_map),
        source_url=source_url,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True, help="Model names from models.yaml")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()

    try:
        registry = ModelRegistry(args.config)
    except ModelRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest_entries: dict[str, dict] = {}
    exit_code = 0
    for name in args.models:
        try:
            spec = registry.get(name)
        except ModelRegistryError as exc:
            print(f"\n=== {name} ===\n[error] {exc}", file=sys.stderr)
            exit_code = 2
            continue

        record = process_model(spec, force=args.force)
        meta_path = METADATA_DIR / f"{spec.name}.json"
        write_json(meta_path, record)
        print(f"Metadata written: {meta_path.relative_to(WEIGHTS_DIR.parent)}")

        validated = record["status"] == STATUS_VALIDATED
        manifest_entries[spec.name] = {
            "status": record["status"],
            "metadata": f"metadata/{spec.name}.json",
            "checkpoint": f"pytorch/{spec.checkpoint}" if validated else None,
            "sha256": record["sha256"] if validated else None,
        }
        if not validated:
            exit_code = max(exit_code, 1)

    if manifest_entries:
        update_manifest(manifest_entries)
        print(f"\nManifest updated: {WEIGHTS_DIR.name}/manifest.json")

    print("\nSummary:")
    for name, entry in manifest_entries.items():
        print(f"  {name:10s} {entry['status']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
