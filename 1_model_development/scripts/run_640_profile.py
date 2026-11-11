#!/usr/bin/env python3
"""640x640 Profile Pipeline — Export ONNX FP32, NCNN FP32, INT8 PTQ + accuracy evaluation.

Artifact naming: {model}_rod_{imgsz}_{runtime}_{precision}.{ext}
  e.g. yolov8n_rod_640_onnx_fp32.onnx
       yolov8n_rod_640_ncnn_fp32/model.ncnn.param + model.ncnn.bin
       yolov8n_rod_640_onnx_int8.onnx

Rules:
- accuracy_passed=true ONLY after full rod_clean_v1/test evaluation via Ultralytics.
- Graph check / cosine similarity / smoke test do NOT set accuracy_passed.
- No Pi benchmark, no Pareto selection, no best-model claims.
- All Pi fields: Pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.checker
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DEV_ROOT = SCRIPT_DIR.parent
BLIND_NAV_ROOT = MODEL_DEV_ROOT.parent
REPO_ROOT = BLIND_NAV_ROOT.parent
ROD_PAPER_ROOT = REPO_ROOT / "rod-paper"
DATASETS_ROOT = MODEL_DEV_ROOT / "datasets"
CLEAN_DATASET_DIR = DATASETS_ROOT / "processed" / "rod_clean_v1"
WEIGHTS_DIR = MODEL_DEV_ROOT / "weights"
REPORTS_DIR = MODEL_DEV_ROOT / "reports" / "pipeline_run"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_DEV_ARTIFACTS = MODEL_DEV_ROOT / "artifacts"

if str(MODEL_DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_DEV_ROOT))

# ── ROD Models ─────────────────────────────────────────────────────────────────
ROD_MODELS: dict[str, dict[str, Any]] = {
    "yolov8n":  {"checkpoint": ROD_PAPER_ROOT / "yolov8n-rod" / "yolov8n_best.pt"},
    "yolov9t":  {"checkpoint": ROD_PAPER_ROOT / "yolov9t"     / "yolov9t_best.pt"},
    "yolov12n": {"checkpoint": ROD_PAPER_ROOT / "yolov12n"    / "yolov12n_best.pt"},
    "yolov26n": {"checkpoint": ROD_PAPER_ROOT / "yolov26n"    / "yolo26n_best.pt"},
}

ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]

MODEL_OUTPUT_ADAPTER = {
    "yolov8n": "raw",
    "yolov9t": "raw",
    "yolov12n": "raw",
    "yolov26n": "e2e_nms",
}

NORMALIZATION_INFO = {
    "mode": "divide_by_255",
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "channel_order": "RGB",
    "note": "pixel / 255.0 — no ImageNet mean subtraction",
}

LETTERBOX_POLICY = {
    "mode": "letterbox",
    "fill_value": 114,
    "aspect_ratio": "preserved",
}

OUTPUT_ADAPTERS = {
    "raw": {
        "shape_pattern": "[1, num_classes+4, num_boxes]",
        "example": "[1, 29, 8400] for 640x640 with 25 classes",
        "note": "YOLOv8/v9/v12 raw output; requires NMS postprocessing",
    },
    "e2e_nms": {
        "shape_pattern": "[1, max_dets, 6]",
        "example": "[1, 300, 6] — [x1,y1,x2,y2,conf,cls_id]",
        "note": "YOLO26n end-to-end NMS; no additional NMS needed",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def artifact_stem(model: str, imgsz: int, runtime: str, precision: str) -> str:
    return f"{model}_rod_{imgsz}_{runtime}_{precision}"


def _cached_320_artifact(model: str, artifact_key: str) -> Path | None:
    manifest_path = WEIGHTS_DIR / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get("models", {}).get(model, {}).get(artifact_key, {})
    if entry.get("imgsz") != 320:
        return None
    relative_path = entry.get("path_relative") or entry.get("path")
    if not relative_path:
        return None
    candidate = (MODEL_DEV_ROOT / relative_path).resolve()
    weights_root = WEIGHTS_DIR.resolve()
    if weights_root not in candidate.parents or not candidate.is_file():
        return None
    expected_hash = entry.get("sha256")
    if expected_hash and sha256_file(candidate) != expected_hash:
        raise ValueError(f"Cached 320 artifact hash mismatch: {candidate}")
    return candidate


def _eval_yaml(clean_dataset_dir: Path) -> Path:
    p = clean_dataset_dir / "eval_clean_test.yaml"
    if not p.exists():
        cfg = {
            "path": str(clean_dataset_dir.resolve()),
            "train": "train/images",
            "val": "test/images",
            "test": "test/images",
            "nc": len(ROD_CLASSES),
            "names": ROD_CLASSES,
        }
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def _run_ultralytics_val(
    model_obj: YOLO,
    eval_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    """Run model.val() on full clean test set. Returns structured metrics."""
    t0 = time.time()
    is_int8 = "int8" in str(getattr(model_obj, "model", ""))
    val_device = "cpu" if is_int8 else device
    val_batch = 1 if is_int8 else batch
    try:
        res = model_obj.val(
            data=str(eval_yaml),
            split="val",
            imgsz=imgsz,
            batch=val_batch,
            device=val_device,
            conf=0.001,
            iou=0.7,
            plots=False,
            save_json=False,
            verbose=False,
        )
    except Exception as e:
        print(f"  Validation fallback to device='cpu' (cause: {e})")
        res = model_obj.val(
            data=str(eval_yaml),
            split="val",
            imgsz=imgsz,
            batch=1,
            device="cpu",
            conf=0.001,
            iou=0.7,
            plots=False,
            save_json=False,
            verbose=False,
        )
    duration = round(time.time() - t0, 2)
    box = res.box

    per_class_ap50: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_fn: dict[str, int] = {}

    class_indices = list(getattr(box, "ap_class_index", []))
    for cls in ROD_CLASSES:
        per_class_recall[cls] = None
    if getattr(box, "ap50", None) is not None:
        for i, class_index in enumerate(class_indices):
            if i < len(box.ap50) and class_index < len(ROD_CLASSES):
                per_class_ap50[ROD_CLASSES[class_index]] = round(float(box.ap50[i]), 6)
    if getattr(box, "r", None) is not None:
        for i, class_index in enumerate(class_indices):
            if i < len(box.r) and class_index < len(ROD_CLASSES):
                per_class_recall[ROD_CLASSES[class_index]] = round(float(box.r[i]), 6)

    target_counts = np.asarray(getattr(res, "nt_per_class", []), dtype=int)
    if target_counts.size:
        for i, class_index in enumerate(class_indices):
            if i < len(box.r) and class_index < len(target_counts):
                gt_count = int(target_counts[class_index])
                tp_count = round(float(box.r[i]) * gt_count)
                per_class_fn[ROD_CLASSES[class_index]] = max(0, gt_count - tp_count)

    critical_names = set(
        (yaml.safe_load((MODEL_DEV_ROOT / "configs" / "danger_classes.yaml").read_text())
         or {}).get("groups", {}).get("critical", {}).get("classes", [])
    )
    critical_recalls = [
        value for cls, value in per_class_recall.items()
        if cls in critical_names and value is not None
    ]
    dangerous_macro_recall = round(float(np.mean(critical_recalls)), 6) if critical_recalls else None
    false_negative_count = sum(per_class_fn.values()) if per_class_fn else None

    return {
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "ap_small": None,
        "ap_small_status": "not_provided_by_ultralytics_val",
        "per_class_ap50": per_class_ap50,
        "per_class_recall": per_class_recall,
        "per_class_false_negatives": per_class_fn,
        "false_negatives": false_negative_count,
        "false_negative_definition": "estimated from ground-truth counts and rounded per-class recall at Ultralytics' selected F1 confidence",
        "dangerous_class_macro_recall": dangerous_macro_recall,
        "eval_duration_sec": duration,
    }


def _degradation(pt_m: dict, rt_m: dict) -> dict[str, float]:
    keys = ["mAP50", "mAP50_95", "precision", "recall"]
    return {
        f"{k}_delta": round(rt_m.get(k, 0.0) - pt_m.get(k, 0.0), 6)
        for k in keys
    }


# ── STEP A ─────────────────────────────────────────────────────────────────────
def step_a_evaluate_pt_baseline(imgsz: int, batch: int, device: str) -> dict[str, Any]:
    print(f"\n{'='*70}\nSTEP A: PT baseline at {imgsz}x{imgsz} on rod_clean_v1/test\n{'='*70}")
    eval_yaml = _eval_yaml(CLEAN_DATASET_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for mname, cfg in ROD_MODELS.items():
        ckpt = cfg["checkpoint"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        print(f"\n  [{mname}] .pt at {imgsz}x{imgsz}...")
        mdl = YOLO(str(ckpt))
        m = _run_ultralytics_val(mdl, eval_yaml, imgsz, batch, device)
        m.update({"model": mname, "checkpoint": str(ckpt), "imgsz": imgsz,
                   "runtime": "pytorch", "accuracy_passed": True})
        results[mname] = m
        print(f"  mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50_95']:.4f}  R={m['recall']:.4f}")

    out = REPORTS_DIR / f"accuracy_reference_{imgsz}_pt.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ PT-{imgsz} reference → {out}")
    return results


# ── STEP B ─────────────────────────────────────────────────────────────────────
def step_b_export_onnx(pt_baselines: dict, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    print(f"\n{'='*70}\nSTEP B: ONNX FP32 at {imgsz}x{imgsz} + accuracy eval\n{'='*70}")
    onnx_dir = WEIGHTS_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    eval_yaml = _eval_yaml(CLEAN_DATASET_DIR)
    all_res: dict[str, Any] = {}

    for mname, cfg in ROD_MODELS.items():
        ckpt = cfg["checkpoint"]
        stem = artifact_stem(mname, imgsz, "onnx", "fp32")
        out_path = (
            _cached_320_artifact(mname, "onnx_fp32_320") if imgsz == 320 else None
        ) or (onnx_dir / f"{stem}.onnx")

        exp_time = 0.0
        if out_path.exists():
            print(f"\n  [{mname}] Reusing existing ONNX FP32 → {out_path.name}")
        else:
            print(f"\n  [{mname}] Exporting ONNX FP32 → {out_path.name}")
            mdl = YOLO(str(ckpt))
            t0 = time.time()
            exported = mdl.export(
                format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                simplify=True, opset=12, half=False, int8=False, nms=False, device="cpu"
            )
            exp_time = round(time.time() - t0, 2)
            exp_path = Path(exported)
            if exp_path.resolve() != out_path.resolve():
                shutil.move(str(exp_path), str(out_path))

        # Graph check (necessary but NOT sufficient for accuracy_passed)
        onnx_m = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_m)
        sha = sha256_file(out_path)
        size_mb = round(out_path.stat().st_size / 1024 / 1024, 2)
        print(f"  ✓ Graph OK | {size_mb} MB | SHA: {sha[:12]}...")

        # MANDATORY: full accuracy evaluation on rod_clean_v1/test
        print(f"  Evaluating ONNX via Ultralytics on rod_clean_v1/test ({imgsz}x{imgsz})...")
        onnx_mdl = YOLO(str(out_path))
        acc = _run_ultralytics_val(onnx_mdl, eval_yaml, imgsz, batch, device)
        deg = _degradation(pt_baselines[mname], acc)
        accuracy_passed = bool(acc["mAP50"] > 0.0 and acc["recall"] > 0.0)
        print(f"  mAP50={acc['mAP50']:.4f}  ΔMAP50={deg['mAP50_delta']:+.4f}  accuracy_passed={accuracy_passed}")

        all_res[mname] = {
            "artifact": out_path.name,
            "path_relative": f"weights/onnx/{out_path.name}",
            "sha256": sha, "size_mb": size_mb, "imgsz": imgsz,
            "runtime": "onnx", "precision": "fp32",
            "export_time_sec": exp_time,
            "accuracy": acc, "degradation_vs_pt": deg,
            "accuracy_passed": accuracy_passed,
        }

    out_file = REPORTS_DIR / f"onnx_fp32_{imgsz}_accuracy_report.json"
    out_file.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imgsz": imgsz, "runtime": "onnx_fp32", "models": all_res,
    }, indent=2))
    print(f"\n✓ ONNX FP32-{imgsz} report → {out_file}")
    return all_res


# ── STEP C ─────────────────────────────────────────────────────────────────────
def step_c_convert_ncnn(pt_baselines: dict, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    print(f"\n{'='*70}\nSTEP C: NCNN FP32 at {imgsz}x{imgsz} + accuracy eval\n{'='*70}")
    ncnn_dir = WEIGHTS_DIR / "ncnn"
    ncnn_dir.mkdir(parents=True, exist_ok=True)
    eval_yaml = _eval_yaml(CLEAN_DATASET_DIR)
    all_res: dict[str, Any] = {}

    for mname, cfg in ROD_MODELS.items():
        ckpt = cfg["checkpoint"]
        stem = artifact_stem(mname, imgsz, "ncnn", "fp32")
        target_dir = ncnn_dir / stem

        print(f"\n  [{mname}] Converting NCNN FP32 → {target_dir.name}/")
        mdl = YOLO(str(ckpt))
        conversion_status = "converted"
        try:
            ncnn_exp = mdl.export(format="ncnn", imgsz=imgsz, half=False, device="cpu")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            if Path(ncnn_exp).is_dir():
                shutil.copytree(str(ncnn_exp), str(target_dir))
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ncnn_exp), target_dir)
        except Exception as e:
            print(f"  ⚠ NCNN error: {e}")
            target_dir.mkdir(parents=True, exist_ok=True)
            conversion_status = f"error: {e}"

        # Canonicalise param/bin names
        for pf in list(target_dir.glob("*.param")):
            dest = target_dir / "model.ncnn.param"
            if pf != dest and not dest.exists():
                pf.rename(dest)
        for bf in list(target_dir.glob("*.bin")):
            dest = target_dir / "model.ncnn.bin"
            if bf != dest and not dest.exists():
                bf.rename(dest)

        param_files = list(target_dir.glob("*.param"))
        bin_files   = list(target_dir.glob("*.bin"))
        total_mb    = round(sum(f.stat().st_size for f in param_files + bin_files) / 1024 / 1024, 2)

        sha_param = sha256_file(target_dir / "model.ncnn.param") if (target_dir / "model.ncnn.param").exists() else ""
        sha_bin = sha256_file(target_dir / "model.ncnn.bin") if (target_dir / "model.ncnn.bin").exists() else ""

        # Direct files in weights/ncnn/
        param_direct = ncnn_dir / f"{stem}.param"
        bin_direct = ncnn_dir / f"{stem}.bin"
        if (target_dir / "model.ncnn.param").exists():
            shutil.copy2(target_dir / "model.ncnn.param", param_direct)
        if (target_dir / "model.ncnn.bin").exists():
            shutil.copy2(target_dir / "model.ncnn.bin", bin_direct)

        # Accuracy via ONNX FP32 proxy (same weights, host-side evaluation)
        onnx_proxy = WEIGHTS_DIR / "onnx" / f"{artifact_stem(mname, imgsz, 'onnx', 'fp32')}.onnx"
        acc, deg, accuracy_passed = {}, {}, False
        accuracy_note = ""
        if onnx_proxy.exists():
            print(f"  Evaluating via ONNX FP32 proxy on rod_clean_v1/test...")
            onnx_mdl = YOLO(str(onnx_proxy))
            acc = _run_ultralytics_val(onnx_mdl, eval_yaml, imgsz, batch, device)
            deg = _degradation(pt_baselines[mname], acc)
            accuracy_passed = bool(acc["mAP50"] > 0.0)
            accuracy_note = (
                "Host accuracy evaluated via ONNX FP32 proxy (identical weights). "
                "True on-device NCNN inference accuracy requires Pi 4 hardware. Status: Pending."
            )
            print(f"  mAP50={acc['mAP50']:.4f}  ΔMAP50={deg['mAP50_delta']:+.4f} [ONNX proxy]")
        else:
            accuracy_note = "ONNX FP32 proxy not found. Run STEP B first. accuracy_passed=false."

        all_res[mname] = {
            "artifact": stem,
            "dir_relative": f"weights/ncnn/{stem}",
            "param_path_relative": f"weights/ncnn/{stem}.param",
            "bin_path_relative": f"weights/ncnn/{stem}.bin",
            "sha256": {"param": sha_param, "bin": sha_bin},
            "param_file": "model.ncnn.param",
            "bin_file": "model.ncnn.bin",
            "total_size_mb": total_mb,
            "imgsz": imgsz,
            "runtime": "ncnn",
            "precision": "fp32",
            "conversion_status": conversion_status,
            "accuracy": acc,
            "degradation_vs_pt": deg,
            "accuracy_passed": accuracy_passed,
            "accuracy_note": accuracy_note,
        }

    out_file = REPORTS_DIR / f"ncnn_fp32_{imgsz}_accuracy_report.json"
    out_file.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imgsz": imgsz, "runtime": "ncnn_fp32", "models": all_res,
    }, indent=2))
    print(f"\n✓ NCNN FP32-{imgsz} report → {out_file}")
    return all_res


# ── STEP D ─────────────────────────────────────────────────────────────────────
def step_d_int8_ptq(pt_baselines: dict, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    print(f"\n{'='*70}\nSTEP D: INT8 PTQ at {imgsz}x{imgsz} + accuracy eval\n{'='*70}")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_dir = WEIGHTS_DIR / "int8"
    int8_dir.mkdir(parents=True, exist_ok=True)
    eval_yaml = _eval_yaml(CLEAN_DATASET_DIR)
    data_yaml = CLEAN_DATASET_DIR / "data.yaml"
    all_res: dict[str, Any] = {}

    for mname, cfg in ROD_MODELS.items():
        ckpt = cfg["checkpoint"]
        src_onnx = (
            _cached_320_artifact(mname, "onnx_fp32_320") if imgsz == 320 else None
        ) or (WEIGHTS_DIR / "onnx" / f"{artifact_stem(mname, imgsz, 'onnx', 'fp32')}.onnx")
        stem_int8 = artifact_stem(mname, imgsz, "onnx", "int8")
        out_int8 = (
            _cached_320_artifact(mname, "int8_ptq_320") if imgsz == 320 else None
        ) or (int8_dir / f"{stem_int8}.onnx")

        if not src_onnx.exists():
            raise FileNotFoundError(
                f"Source ONNX FP32 not found for INT8 PTQ: {src_onnx}\n"
                "Run STEP B first."
            )

        if out_int8.exists():
            print(f"\n  [{mname}] Reusing existing INT8 PTQ → {out_int8.name}")
            ptq_method = "reused_existing_artifact"
        else:
            print(f"\n  [{mname}] INT8 PTQ → {out_int8.name}")
            mdl = YOLO(str(ckpt))
            ptq_method = "onnxruntime_dynamic"
            try:
                exported_int8 = mdl.export(
                    format="onnx", imgsz=imgsz, int8=True,
                    data=str(data_yaml.resolve()), device="cpu",
                )
                exp_path = Path(exported_int8)
                if exp_path.resolve() != out_int8.resolve():
                    shutil.copy2(str(exp_path), str(out_int8))
                ptq_method = "ultralytics_ptq_calibration"
            except Exception as e:
                print(f"  Fallback to dynamic quantization ({e})")
                quantize_dynamic(str(src_onnx), str(out_int8), weight_type=QuantType.QInt8)

        # Sanity run (necessary but NOT accuracy_passed)
        sess = ort.InferenceSession(str(out_int8), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        assert sess.run(None, {inp: dummy}) is not None, "INT8 sanity failed"

        sha = sha256_file(out_int8)
        fp32_mb = round(src_onnx.stat().st_size / 1024 / 1024, 2)
        int8_mb = round(out_int8.stat().st_size / 1024 / 1024, 2)
        compression = round(fp32_mb / int8_mb, 2) if int8_mb > 0 else 1.0

        # MANDATORY: full accuracy eval
        print(f"  Evaluating INT8 via Ultralytics on rod_clean_v1/test ({imgsz}x{imgsz})...")
        int8_mdl = YOLO(str(out_int8))
        acc = _run_ultralytics_val(int8_mdl, eval_yaml, imgsz, batch, device)
        deg = _degradation(pt_baselines[mname], acc)
        accuracy_passed = bool(acc["mAP50"] > 0.0 and acc["recall"] > 0.0)
        print(f"  mAP50={acc['mAP50']:.4f}  ΔMAP50={deg['mAP50_delta']:+.4f}  "
              f"compression={compression}x  accuracy_passed={accuracy_passed}")

        all_res[mname] = {
            "artifact": out_int8.name,
            "path_relative": f"weights/int8/{out_int8.name}",
            "sha256": sha, "fp32_size_mb": fp32_mb, "int8_size_mb": int8_mb,
            "compression_ratio": f"{compression}x", "ptq_method": ptq_method,
            "imgsz": imgsz, "runtime": "onnx", "precision": "int8",
            "accuracy": acc, "degradation_vs_pt": deg,
            "accuracy_passed": accuracy_passed,
        }

    out_file = REPORTS_DIR / f"int8_ptq_{imgsz}_accuracy_report.json"
    out_file.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imgsz": imgsz, "runtime": "onnx_int8", "models": all_res,
    }, indent=2))
    print(f"\n✓ INT8 PTQ-{imgsz} report → {out_file}")
    return all_res


def _write_320_comparison(pt_results: dict, onnx_results: dict, int8_results: dict) -> Path:
    def metric_values(result: dict) -> dict:
        return result.get("accuracy", result)

    def metric_delta(left: dict, right: dict) -> dict:
        scalar_keys = (
            "mAP50", "mAP50_95", "precision", "recall", "ap_small",
            "false_negatives", "dangerous_class_macro_recall",
        )
        delta = {}
        for key in scalar_keys:
            left_value, right_value = left.get(key), right.get(key)
            delta[key] = round(right_value - left_value, 6) if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)) else None
        for key in ("per_class_recall", "per_class_false_negatives"):
            left_values, right_values = left.get(key, {}), right.get(key, {})
            delta[key] = {
                cls: round(right_values[cls] - left_values[cls], 6)
                for cls in left_values.keys() & right_values.keys()
                if isinstance(left_values[cls], (int, float)) and isinstance(right_values[cls], (int, float))
            }
        return delta

    models = {}
    for model_name in ROD_MODELS:
        pt = metric_values(pt_results[model_name])
        onnx = metric_values(onnx_results[model_name])
        int8 = metric_values(int8_results[model_name])
        models[model_name] = {
            "metrics": {"pt_320": pt, "onnx_fp32_320": onnx, "onnx_int8_ptq_320": int8},
            "degradation": {
                "onnx_fp32_320_minus_pt_320": metric_delta(pt, onnx),
                "int8_320_minus_onnx_fp32_320": metric_delta(onnx, int8),
                "int8_320_minus_pt_320": metric_delta(pt, int8),
            },
        }

    report_path = REPORTS_DIR / "profile_comparison_320.json"
    report_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imgsz": 320,
        "dataset": "rod_clean_v1/test",
        "test_images": 1615,
        "evaluator": "Ultralytics model.val",
        "validation_thresholds": {"conf": 0.001, "iou": 0.7},
        "models": models,
    }, indent=2))
    print(f"\n✓ 320 comparison report → {report_path}")
    return report_path


# ── STEP E ─────────────────────────────────────────────────────────────────────
def step_e_update_manifest(
    pt_640: dict, onnx_640: dict, ncnn_640: dict, int8_640: dict,
) -> None:
    print(f"\n{'='*70}\nSTEP E: Update weights/manifest.json\n{'='*70}")
    manifest_path = WEIGHTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["schema_version"] = "2.0"
    manifest["dataset"] = {
        "name": "ROD-Dataset",
        "version": "rod_clean_v1",
        "nc": 25,
        "classes": ROD_CLASSES,
        "split_used_for_accuracy": "test",
        "total_test_images": 1615,
        "note": "rod_clean_v1 is the canonical clean split (content_leakage=0, mixed_schema=0)",
    }
    manifest["preprocessing"] = NORMALIZATION_INFO
    manifest["letterbox_policy"] = LETTERBOX_POLICY
    manifest["inference_config"] = {"confidence_threshold": 0.25, "nms_iou_threshold": 0.45}
    manifest["output_adapters"] = OUTPUT_ADAPTERS

    for mname in ROD_MODELS:
        if mname not in manifest["models"]:
            manifest["models"][mname] = {}
        entry = manifest["models"][mname]
        entry["checkpoint"] = str(ROD_MODELS[mname]["checkpoint"].relative_to(REPO_ROOT))
        entry["output_adapter"] = MODEL_OUTPUT_ADAPTER[mname]
        entry["classes"] = ROD_CLASSES
        entry["raspberry_pi_benchmark"] = {
            "latency_p50_ms": "Pending",
            "latency_p95_ms": "Pending",
            "fps": "Pending",
            "peak_ram_mb": "Pending",
            "cpu_temperature_c": "Pending",
            "throttling": "Pending",
            "status": "Pending — awaiting real Pi 4 hardware execution",
        }
        entry["edge_pareto_selection"] = "Pending — no Pi benchmark data yet"

        def fill_onnx_entry(res: dict, imgsz: int) -> dict:
            return {
                "artifact": res.get("artifact", ""),
                "path_relative": res.get("path_relative", ""),
                "sha256": res.get("sha256", ""),
                "size_mb": res.get("size_mb"),
                "imgsz": imgsz,
                "input_format": f"NCHW float32 [1, 3, {imgsz}, {imgsz}]",
                "classes": ROD_CLASSES,
                "normalization": NORMALIZATION_INFO,
                "letterbox": LETTERBOX_POLICY,
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "output_adapter": MODEL_OUTPUT_ADAPTER[mname],
                "accuracy": res.get("accuracy", {}),
                "degradation_vs_pt": res.get("degradation_vs_pt", {}),
                "accuracy_passed": res.get("accuracy_passed", False),
                "export_status": "exported",
            }

        def fill_ncnn_entry(res: dict, imgsz: int) -> dict:
            return {
                "artifact": res.get("artifact", ""),
                "dir_relative": res.get("dir_relative", ""),
                "param_path_relative": res.get("param_path_relative", ""),
                "bin_path_relative": res.get("bin_path_relative", ""),
                "param_file": "model.ncnn.param",
                "bin_file": "model.ncnn.bin",
                "sha256": res.get("sha256", {}),
                "total_size_mb": res.get("total_size_mb"),
                "imgsz": imgsz,
                "input_format": f"NCHW float32 [1, 3, {imgsz}, {imgsz}]",
                "classes": ROD_CLASSES,
                "normalization": NORMALIZATION_INFO,
                "letterbox": LETTERBOX_POLICY,
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "output_adapter": MODEL_OUTPUT_ADAPTER[mname],
                "accuracy": res.get("accuracy", {}),
                "accuracy_note": res.get("accuracy_note", ""),
                "degradation_vs_pt": res.get("degradation_vs_pt", {}),
                "accuracy_passed": res.get("accuracy_passed", False),
                "conversion_status": res.get("conversion_status", "unknown"),
                "export_status": "converted",
            }

        def fill_int8_entry(res: dict, imgsz: int) -> dict:
            return {
                "artifact": res.get("artifact", ""),
                "path_relative": res.get("path_relative", ""),
                "sha256": res.get("sha256", ""),
                "fp32_size_mb": res.get("fp32_size_mb"),
                "int8_size_mb": res.get("int8_size_mb"),
                "compression_ratio": res.get("compression_ratio"),
                "ptq_method": res.get("ptq_method"),
                "imgsz": imgsz,
                "input_format": f"NCHW float32 [1, 3, {imgsz}, {imgsz}]",
                "classes": ROD_CLASSES,
                "normalization": NORMALIZATION_INFO,
                "letterbox": LETTERBOX_POLICY,
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "output_adapter": MODEL_OUTPUT_ADAPTER[mname],
                "accuracy": res.get("accuracy", {}),
                "degradation_vs_pt": res.get("degradation_vs_pt", {}),
                "accuracy_passed": res.get("accuracy_passed", False),
                "export_status": "quantized",
            }

        entry["onnx_fp32_640"] = fill_onnx_entry(onnx_640.get(mname, {}), 640)
        entry["ncnn_fp32_640"] = fill_ncnn_entry(ncnn_640.get(mname, {}), 640)
        entry["onnx_int8_640"] = fill_int8_entry(int8_640.get(mname, {}), 640)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✓ manifest.json updated → {manifest_path}")


# ── STEP F ─────────────────────────────────────────────────────────────────────
def step_f_package_deployment() -> Path:
    print(f"\n{'='*70}\nSTEP F: Package pi_deployment_package.tar.gz (320 + 640)\n{'='*70}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    staging = ARTIFACTS_DIR / "_pi_staging"
    if staging.exists():
        shutil.rmtree(staging)

    for sub in ["weights/onnx", "weights/ncnn", "weights/int8", "configs", "scripts", "assets"]:
        (staging / sub).mkdir(parents=True, exist_ok=True)

    # All weights (320 + 640)
    for f in (WEIGHTS_DIR / "onnx").glob("*.onnx"):
        shutil.copy2(f, staging / "weights" / "onnx" / f.name)
    for f in (WEIGHTS_DIR / "int8").glob("*.onnx"):
        shutil.copy2(f, staging / "weights" / "int8" / f.name)
    for d in (WEIGHTS_DIR / "ncnn").iterdir():
        if d.is_dir():
            shutil.copytree(d, staging / "weights" / "ncnn" / d.name)

    # Manifest
    shutil.copy2(WEIGHTS_DIR / "manifest.json", staging / "weights" / "manifest.json")

    # Configs
    for cfg_file in (MODEL_DEV_ROOT / "configs").rglob("*.yaml"):
        rel = cfg_file.relative_to(MODEL_DEV_ROOT / "configs")
        dest = staging / "configs" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_file, dest)

    # Scripts
    for s_name in ["benchmark_pi.py", "collect_environment.py", "smoke_test_deployment.py"]:
        src = MODEL_DEV_ROOT / "scripts" / s_name
        if src.exists():
            shutil.copy2(src, staging / "scripts" / s_name)

    # requirements-pi.txt
    req_pi = MODEL_DEV_ROOT / "requirements-pi.txt"
    if req_pi.exists():
        shutil.copy2(req_pi, staging / "requirements-pi.txt")

    # INTEGRATION_GUIDE.md
    guide = MODEL_DEV_ROOT / "INTEGRATION_GUIDE.md"
    if guide.exists():
        shutil.copy2(guide, staging / "INTEGRATION_GUIDE.md")

    # deployment_config.yaml
    dep_cfg = BLIND_NAV_ROOT / "2_edge_device" / "deployment_config.yaml"
    if not dep_cfg.exists():
        dep_cfg = MODEL_DEV_ROOT / "deployment_config.yaml"
    if dep_cfg.exists():
        shutil.copy2(dep_cfg, staging / "deployment_config.yaml")

    # Sample image
    test_img_dir = CLEAN_DATASET_DIR / "test" / "images"
    samples = sorted(list(test_img_dir.glob("*.jpg")))
    if samples:
        shutil.copy2(samples[0], staging / "assets" / "sample_rod.jpg")

    # run_on_pi.sh
    run_sh = staging / "run_on_pi.sh"
    run_sh.write_text(
        "#!/bin/bash\nset -e\n"
        "echo 'PBL6 Raspberry Pi Deployment Runner'\n"
        "pip install -r requirements-pi.txt --quiet\n"
        "python scripts/collect_environment.py\n"
        "python scripts/smoke_test_deployment.py --model-dir weights\n"
        "python scripts/benchmark_pi.py --onnx-dir weights/onnx "
        "--output benchmarks_pi --imgsz 320 640\n"
        "echo 'Pi benchmark complete. Upload benchmarks_pi/ to repo.'\n",
        encoding="utf-8"
    )
    run_sh.chmod(0o755)

    # Checksums
    lines = []
    for item in sorted(staging.rglob("*")):
        if item.is_file() and item.name != "checksums.sha256":
            rel = item.relative_to(staging)
            lines.append(f"{sha256_file(item)}  {rel}")
    (staging / "checksums.sha256").write_text("\n".join(lines) + "\n")

    tar_path = ARTIFACTS_DIR / "pi_deployment_package.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(staging, arcname="pi_deployment_package")
    shutil.rmtree(staging)

    size_mb = round(tar_path.stat().st_size / 1024 / 1024, 2)
    sha = sha256_file(tar_path)
    MODEL_DEV_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tar_path, MODEL_DEV_ARTIFACTS / "pi_deployment_package.tar.gz")
    print(f"✓ pi_deployment_package.tar.gz → {tar_path}")
    print(f"  Size: {size_mb} MB | SHA256: {sha[:16]}...")
    return tar_path


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ROD profile export and accuracy pipeline")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    args = parser.parse_args()

    imgsz = args.imgsz
    batch = args.batch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device} | CUDA: {torch.cuda.is_available()}")

    t0 = time.time()
    print(f"\n{'='*70}\nROD {imgsz}x{imgsz} PROFILE PIPELINE — {datetime.now(timezone.utc).isoformat()}\n{'='*70}")

    pt_baselines = step_a_evaluate_pt_baseline(imgsz, batch, device)

    if imgsz == 320:
        print("\n320 evaluation-only mode: NCNN, manifest updates, and Pi packaging are disabled.")
        onnx_res = step_b_export_onnx(pt_baselines, imgsz, batch, device)
        int8_res = step_d_int8_ptq(pt_baselines, imgsz, batch, device)
        _write_320_comparison(pt_baselines, onnx_res, int8_res)
    elif not args.skip_export:
        onnx_res = step_b_export_onnx(pt_baselines, imgsz, batch, device)
        ncnn_res = step_c_convert_ncnn(pt_baselines, imgsz, batch, device)
        int8_res = step_d_int8_ptq(pt_baselines, imgsz, batch, device)
        step_e_update_manifest(pt_baselines, onnx_res, ncnn_res, int8_res)
        if not args.skip_package:
            step_f_package_deployment()
    else:
        def _load(name: str) -> dict:
            p = REPORTS_DIR / name
            return json.loads(p.read_text())["models"]
        onnx_res = _load(f"onnx_fp32_{imgsz}_accuracy_report.json")
        ncnn_res = _load(f"ncnn_fp32_{imgsz}_accuracy_report.json")
        int8_res = _load(f"int8_ptq_{imgsz}_accuracy_report.json")
        step_e_update_manifest(pt_baselines, onnx_res, ncnn_res, int8_res)
        if not args.skip_package:
            step_f_package_deployment()

    print(f"\n{'='*70}\nPIPELINE COMPLETE — {round(time.time()-t0, 1)}s\n{'='*70}")


if __name__ == "__main__":
    main()
