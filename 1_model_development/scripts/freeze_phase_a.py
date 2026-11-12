#!/usr/bin/env python3
"""Freeze Phase A baseline artifacts, hashes, configurations, and environment provenance."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return res.stdout.strip()
    except Exception:
        return None


def freeze_phase_a(root_dir: Path, output_file: Path) -> dict:
    models = ["yolov8n", "yolo11n", "yolo12n", "yolo26n"]

    model_hashes = {}
    for m in models:
        pt_p = root_dir / "weights" / "pytorch" / f"{m}.pt"
        onnx_p = root_dir / "weights" / "onnx" / f"{m}_320_fp32.onnx"
        model_hashes[m] = {
            "pytorch_sha256": sha256_file(pt_p) if pt_p.exists() else None,
            "onnx_sha256": sha256_file(onnx_p) if onnx_p.exists() else None,
        }

    config_hashes = {}
    configs = [
        "configs/models.yaml",
        "configs/export.yaml",
        "configs/benchmark.yaml",
        "configs/danger_classes.yaml",
        "configs/datasets/rod.yaml",
        "configs/datasets/rod_to_coco.yaml",
    ]
    for cfg in configs:
        p = root_dir / cfg
        if p.exists():
            config_hashes[cfg] = sha256_file(p)

    dataset_manifest_path = root_dir / "datasets" / "manifests" / "dataset_manifest.json"
    dataset_hash = sha256_file(dataset_manifest_path) if dataset_manifest_path.exists() else None

    frozen = {
        "phase": "Phase A",
        "version": "v1.0",
        "frozen_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "dataset": {
            "name": "ROD-Dataset",
            "manifest_sha256": dataset_hash,
            "is_placeholder": True,
            "note": "synthetic 11-image placeholder; real ROD not downloaded",
            "num_classes": 25,
            "shared_coco_classes": 11,
            "domain_only_classes": 14,
        },
        "models": model_hashes,
        "configs": config_hashes,
        "validation_status": {
            "dataset_validation": "passed_on_placeholder",
            "taxonomy_validation": "passed",
            "onnx_export": "passed",
            "onnx_graph_check": "passed",
            "pytorch_onnx_equivalence": "not_run",
            "determinism": "not_run",
            "host_benchmarks": "completed",
            "pi_harness_packaged": True,
            "pi_hardware_run": "pending",
            "frozen_ready": False,
        }
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(frozen, f, indent=2)

    print(f"✓ Phase A baseline frozen to: {output_file}")
    return frozen


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    out = root / "weights" / "phase_a_frozen_manifest.json"
    freeze_phase_a(root, out)
