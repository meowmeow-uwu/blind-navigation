"""Smoke-test downloaded checkpoints on a single image.

Usage (from 1_model_development/):
    python scripts/smoke_test_model.py --model yolov8n --source assets/smoke_test.jpg --imgsz 320 --device cpu
    python scripts/smoke_test_model.py --models yolov8n yolo11n yolo12n yolo26n \
        --source assets/smoke_test.jpg --imgsz 320 --device cpu \
        --output benchmarks/summaries/smoke_test.json

Checkpoints are resolved through weights/manifest.json (never by scanning folders).
Timings are only for spotting anomalies; they are NOT benchmark results.
Having zero detections is valid as long as the output structure is valid.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ROOT,
    STATUS_VALIDATED,
    load_manifest,
    resolve_manifest_checkpoint,
    resolve_manifest_metadata,
    write_json,
)
from models.model_registry import ModelRegistry, ModelRegistryError  # noqa: E402

# Statuses that make the process exit non-zero.
FAILING_STATUSES = {"failed", "unknown_model", "not_downloaded"}


def validate_result(result, names: dict[int, str], image_shape: tuple[int, int]) -> int:
    """Validate an Ultralytics result. Returns detection count; raises AssertionError."""
    import numpy as np

    boxes = result.boxes
    assert boxes is not None, "result.boxes is None (not a detect result)"

    data = boxes.data.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy().reshape(-1)
    cls = boxes.cls.cpu().numpy().reshape(-1)
    n = len(xyxy)

    assert xyxy.ndim == 2 and xyxy.shape[1] == 4, f"xyxy shape {xyxy.shape}, expected (N, 4)"
    assert conf.shape == (n,) and cls.shape == (n,), (
        f"inconsistent output lengths: xyxy={n}, conf={conf.shape}, cls={cls.shape}"
    )
    assert np.isfinite(data).all(), "NaN/Inf found in output tensor"

    if n:
        h, w = image_shape
        tol = 1.0  # pixels
        assert (xyxy[:, 2] > xyxy[:, 0]).all() and (xyxy[:, 3] > xyxy[:, 1]).all(), (
            "box with non-positive width/height"
        )
        assert (xyxy[:, 0] >= -tol).all() and (xyxy[:, 1] >= -tol).all(), "box has negative coordinates"
        assert (xyxy[:, 2] <= w + tol).all() and (xyxy[:, 3] <= h + tol).all(), (
            f"box outside image bounds {w}x{h}"
        )
        assert ((conf >= 0.0) & (conf <= 1.0)).all(), "confidence outside [0, 1]"
        assert (cls == np.round(cls)).all(), "non-integer class id"
        bad = sorted({int(c) for c in cls if int(c) not in names})
        assert not bad, f"class ids {bad} not in checkpoint taxonomy (0..{len(names) - 1})"
    return int(n)


def smoke_test_one(
    name: str,
    registry: ModelRegistry,
    manifest: dict | None,
    image,
    imgsz: int,
    device: str,
    conf: float,
) -> dict:
    # Registry: is the model known / enabled?
    try:
        spec = registry.get(name)
    except ModelRegistryError as exc:
        return {"status": "unknown_model", "error": str(exc)}

    if manifest is None:
        return {
            "status": "not_downloaded",
            "error": "weights/manifest.json not found. Run scripts/download_models.py first.",
        }

    entry = manifest.get("models", {}).get(name)
    if entry is None:
        return {
            "status": "not_downloaded",
            "error": f"'{name}' is not in weights/manifest.json. "
            f"Run: python scripts/download_models.py --models {name}",
        }

    if entry["status"] != STATUS_VALIDATED:
        meta = resolve_manifest_metadata(manifest, name) or {}
        return {"status": entry["status"], "error": meta.get("error") or "see metadata"}

    ckpt = resolve_manifest_checkpoint(manifest, name)
    if ckpt is None or not ckpt.is_file():
        return {"status": "failed", "error": f"Checkpoint missing on disk: {ckpt}"}

    try:
        from ultralytics import YOLO

        t0 = time.perf_counter()
        model = YOLO(str(ckpt))
        load_ms = (time.perf_counter() - t0) * 1000
        names = {int(k): v for k, v in model.names.items()}

        predict_kwargs = dict(
            imgsz=imgsz, device=device, conf=conf, batch=1, verbose=False
        )
        # Untimed warm-up so one-off initialisation does not look like an anomaly.
        model.predict(image, **predict_kwargs)

        t0 = time.perf_counter()
        results = model.predict(image, **predict_kwargs)
        inference_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == 1, f"expected 1 result for 1 image, got {len(results)}"
        detections = validate_result(results[0], names, image.shape[:2])
    except AssertionError as exc:
        return {"status": "failed", "error": f"Output validation failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    return {
        "status": "passed",
        "load_ms": round(load_ms),
        "inference_ms": round(inference_ms),
        "detections": detections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", "--model", dest="models", nargs="+", required=True,
        help="One or more model names from models.yaml",
    )
    parser.add_argument("--source", type=Path, default=ROOT / "assets" / "smoke_test.jpg")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON summary here")
    args = parser.parse_args()

    import cv2

    source = args.source if args.source.is_absolute() else Path.cwd() / args.source
    if not source.is_file():
        print(f"ERROR: source image not found: {source}", file=sys.stderr)
        return 2
    image = cv2.imread(str(source))
    if image is None or image.size == 0:
        print(f"ERROR: cannot decode image: {source}", file=sys.stderr)
        return 2
    print(f"Image: {source} shape={image.shape} | imgsz={args.imgsz} device={args.device} conf={args.conf} batch=1")

    try:
        registry = ModelRegistry()
    except ModelRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    manifest = load_manifest()

    summary: dict[str, dict] = {}
    for name in args.models:
        result = smoke_test_one(name, registry, manifest, image, args.imgsz, args.device, args.conf)
        summary[name] = result
        line = f"[{result['status']}] {name}"
        if result["status"] == "passed":
            line += f"  load={result['load_ms']}ms  inference={result['inference_ms']}ms  detections={result['detections']}"
        else:
            line += f"  {result.get('error', '')}"
        print(line)

    if args.output:
        out = args.output if args.output.is_absolute() else Path.cwd() / args.output
        write_json(out, summary)
        print(f"Summary written: {out}")

    return 1 if any(r["status"] in FAILING_STATUSES for r in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
