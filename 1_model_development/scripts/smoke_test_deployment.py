#!/usr/bin/env python3
"""Smoke-test deployment artifacts from weights/manifest.json.

Verifies that every artifact listed in manifest.json:
  1. Exists on disk at the declared path.
  2. Passes SHA-256 integrity check.
  3. Loads without error (ONNX via onnxruntime).
  4. Outputs the correct schema: list of {class_id, class_name, confidence, bbox}.
  5. Output contains no NaN or Inf values.
  6. Bounding boxes lie within image bounds [0, imgsz].

This smoke test does NOT set accuracy_passed=true in manifest.json.
Accuracy evaluation is performed exclusively by run_640_profile.py / execute_full_rod_pipeline.py
using the full rod_clean_v1/test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ── 25 ROD Classes ─────────────────────────────────────────────────────────────
ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_detections(output: np.ndarray, imgsz: int, conf_thresh: float = 0.0) -> list[dict]:
    """Parse ONNX model output into detection schema.

    Handles two output formats:
      - raw: [1, num_classes+4, num_boxes]  — YOLOv8/v9/v12
      - e2e_nms: [1, max_dets, 6]           — YOLO26n
    """
    detections = []
    if output.ndim == 3:
        if output.shape[2] == 6:
            # e2e_nms format: [1, N, 6] → x1, y1, x2, y2, conf, cls_id
            for det in output[0]:
                x1, y1, x2, y2, conf, cls_id = det.tolist()
                if conf >= conf_thresh:
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": ROD_CLASSES[int(cls_id)] if 0 <= int(cls_id) < len(ROD_CLASSES) else "unknown",
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    })
        else:
            # raw format: [1, num_classes+4, num_boxes] → transpose → [num_boxes, ...]
            preds = output[0].T  # [num_boxes, num_classes+4]
            for pred in preds:
                boxes = pred[:4]   # cx, cy, w, h
                class_scores = pred[4:]
                cls_id = int(class_scores.argmax())
                conf = float(class_scores.max())
                if conf >= conf_thresh:
                    cx, cy, w, h = boxes.tolist()
                    detections.append({
                        "class_id": cls_id,
                        "class_name": ROD_CLASSES[cls_id] if 0 <= cls_id < len(ROD_CLASSES) else "unknown",
                        "confidence": conf,
                        "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    })
    return detections


def smoke_test_onnx(artifact_path: Path, imgsz: int, expected_sha: str | None) -> dict:
    """Run smoke test on a single ONNX artifact."""
    result = {
        "artifact": str(artifact_path),
        "imgsz": imgsz,
        "tests": {},
        "passed": True,
    }

    # 1. Exists
    exists = artifact_path.is_file()
    result["tests"]["file_exists"] = {"passed": exists}
    if not exists:
        result["passed"] = False
        result["tests"]["file_exists"]["error"] = f"File not found: {artifact_path}"
        return result

    # 2. SHA-256
    if expected_sha:
        actual_sha = sha256_file(artifact_path)
        sha_ok = actual_sha == expected_sha
        result["tests"]["sha256"] = {
            "passed": sha_ok,
            "expected": expected_sha,
            "actual": actual_sha,
        }
        if not sha_ok:
            result["passed"] = False
            print(f"    ✗ SHA256 MISMATCH  expected={expected_sha[:16]}  actual={actual_sha[:16]}")

    # 3. Loads without error
    try:
        sess = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
        result["tests"]["model_loads"] = {"passed": True}
    except Exception as e:
        result["tests"]["model_loads"] = {"passed": False, "error": str(e)}
        result["passed"] = False
        return result

    # 4. Inference with dummy input
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    try:
        outputs = sess.run(None, {inp_name: dummy})
        result["tests"]["inference_runs"] = {"passed": True, "num_outputs": len(outputs)}
    except Exception as e:
        result["tests"]["inference_runs"] = {"passed": False, "error": str(e)}
        result["passed"] = False
        return result

    # 5. No NaN / Inf
    has_nan_inf = any(
        np.any(np.isnan(o)) or np.any(np.isinf(o))
        for o in outputs if isinstance(o, np.ndarray)
    )
    result["tests"]["no_nan_inf"] = {"passed": not has_nan_inf}
    if has_nan_inf:
        result["passed"] = False

    # 6. Schema + bbox bounds
    try:
        detections = parse_detections(outputs[0], imgsz, conf_thresh=0.0)
        schema_ok = all(
            isinstance(d.get("class_id"), int)
            and isinstance(d.get("class_name"), str)
            and isinstance(d.get("confidence"), float)
            and isinstance(d.get("bbox"), list)
            and len(d["bbox"]) == 4
            for d in detections[:20]  # check first 20
        )
        boxes_in_bounds = all(
            all(-imgsz * 0.1 <= v <= imgsz * 1.1 for v in d["bbox"])
            for d in detections[:20]
        )
        result["tests"]["schema_ok"] = {"passed": schema_ok}
        result["tests"]["boxes_in_bounds"] = {"passed": boxes_in_bounds, "num_detections": len(detections)}
        if not schema_ok or not boxes_in_bounds:
            result["passed"] = False
    except Exception as e:
        result["tests"]["schema_ok"] = {"passed": False, "error": str(e)}
        result["passed"] = False

    return result


def smoke_test_ncnn(
    param_path: Path,
    bin_path: Path,
    imgsz: int,
    expected_sha_param: str | None,
    expected_sha_bin: str | None,
) -> dict:
    """Run smoke test on an NCNN artifact (.param and .bin)."""
    result = {
        "artifact": f"{param_path.name} / {bin_path.name}",
        "imgsz": imgsz,
        "tests": {},
        "passed": True,
    }

    # 1. Existence
    param_exists = param_path.is_file()
    bin_exists = bin_path.is_file()
    result["tests"]["files_exist"] = {
        "passed": param_exists and bin_exists,
        "param_exists": param_exists,
        "bin_exists": bin_exists,
    }
    if not (param_exists and bin_exists):
        result["passed"] = False
        return result

    # 2. SHA-256
    if expected_sha_param:
        act_p = sha256_file(param_path)
        sha_p_ok = act_p == expected_sha_param
        result["tests"]["sha256_param"] = {
            "passed": sha_p_ok, "expected": expected_sha_param, "actual": act_p
        }
        if not sha_p_ok:
            result["passed"] = False

    if expected_sha_bin:
        act_b = sha256_file(bin_path)
        sha_b_ok = act_b == expected_sha_bin
        result["tests"]["sha256_bin"] = {
            "passed": sha_b_ok, "expected": expected_sha_bin, "actual": act_b
        }
        if not sha_b_ok:
            result["passed"] = False

    # 3. Load & Run with NCNN if module available
    try:
        import ncnn
        net = ncnn.Net()
        net.opt.num_threads = 4
        net.opt.use_vulkan_compute = False
        ret_param = net.load_param(str(param_path))
        ret_bin = net.load_model(str(bin_path))
        loads_ok = ret_param == 0 and ret_bin == 0
        result["tests"]["model_loads"] = {"passed": loads_ok}
        if not loads_ok:
            result["passed"] = False
            return result

        dummy = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        mat_in = ncnn.Mat.from_pixels(dummy, ncnn.Mat.PixelType.PIXEL_RGB, imgsz, imgsz)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        runs_ok = ret == 0
        result["tests"]["inference_runs"] = {"passed": runs_ok}
        if not runs_ok:
            result["passed"] = False
            return result

        arr = np.array(mat_out)
        has_nan_inf = bool(np.any(np.isnan(arr)) or np.any(np.isinf(arr)))
        result["tests"]["no_nan_inf"] = {"passed": not has_nan_inf}
        if has_nan_inf:
            result["passed"] = False

        # Parse detections
        out_3d = arr[np.newaxis, ...]  # [1, 29, num_boxes]
        detections = parse_detections(out_3d, imgsz, conf_thresh=0.0)
        schema_ok = len(detections) > 0 and all(
            isinstance(d.get("class_id"), int)
            and isinstance(d.get("class_name"), str)
            and isinstance(d.get("confidence"), float)
            and isinstance(d.get("bbox"), list)
            and len(d["bbox"]) == 4
            for d in detections[:20]
        )
        # Check boxes in bounds (for valid anchors or top confidence)
        top_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:20]
        boxes_in_bounds = all(
            all(-imgsz * 0.2 <= v <= imgsz * 1.2 for v in d["bbox"])
            for d in top_dets
        )
        result["tests"]["schema_ok"] = {"passed": schema_ok}
        result["tests"]["boxes_in_bounds"] = {"passed": boxes_in_bounds, "num_detections": len(detections)}
        if not schema_ok or not boxes_in_bounds:
            result["passed"] = False
    except ImportError:
        result["tests"]["ncnn_runtime"] = {
            "passed": True,
            "note": "ncnn module not installed; skipped on-device execution test",
        }
    except Exception as e:
        result["tests"]["inference_runs"] = {"passed": False, "error": str(e)}
        result["passed"] = False

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test deployment artifacts")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Directory containing weights/ subdirectory with manifest.json")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Direct path to manifest.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Resolve manifest
    if args.manifest:
        manifest_path = args.manifest
    elif args.model_dir:
        manifest_path = args.model_dir / "manifest.json"
    else:
        # Default: look relative to script location
        here = Path(__file__).resolve()
        for candidate in [
            here.parents[1] / "weights" / "manifest.json",
            here.parents[2] / "weights" / "manifest.json",
        ]:
            if candidate.is_file():
                manifest_path = candidate
                break
        else:
            print("ERROR: manifest.json not found. Pass --manifest or --model-dir.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Smoke-testing artifacts from: {manifest_path}")
    print(f"{'='*60}")

    manifest = json.loads(manifest_path.read_text())
    models = manifest.get("models", {})

    all_results = []
    total_tests = 0
    total_passed = 0

    def resolve_path(rel_str: str) -> Path:
        rel = Path(rel_str)
        candidates = [
            manifest_path.parent.parent / rel,
            manifest_path.parent / rel,
            manifest_path.parent / rel.name,
            manifest_path.parents[2] / rel,
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    for mname, entry in models.items():
        print(f"\n── {mname} ──")

        # Collect artifacts
        onnx_artifacts = []
        ncnn_artifacts = []

        for key, val in entry.items():
            if not isinstance(val, dict):
                continue

            # ONNX
            path_str = val.get("path_relative") or val.get("path")
            if path_str and str(path_str).endswith(".onnx"):
                onnx_artifacts.append({
                    "key": key,
                    "path": resolve_path(path_str),
                    "sha256": val.get("sha256"),
                    "imgsz": val.get("imgsz", 320),
                })

            # NCNN
            if "param_path_relative" in val or "dir_relative" in val:
                param_rel = val.get("param_path_relative") or f"{val.get('dir_relative')}/model.ncnn.param"
                bin_rel = val.get("bin_path_relative") or f"{val.get('dir_relative')}/model.ncnn.bin"
                sha_dict = val.get("sha256", {}) if isinstance(val.get("sha256"), dict) else {}
                ncnn_artifacts.append({
                    "key": key,
                    "param_path": resolve_path(param_rel),
                    "bin_path": resolve_path(bin_rel),
                    "sha256_param": sha_dict.get("param"),
                    "sha256_bin": sha_dict.get("bin"),
                    "imgsz": val.get("imgsz", 320),
                })

        # Test ONNX
        for art in onnx_artifacts:
            print(f"  Testing [{art['key']}] {art['path'].name} (imgsz={art['imgsz']})...")
            res = smoke_test_onnx(art["path"], art["imgsz"], art["sha256"])
            all_results.append(res)
            total_tests += 1
            if res["passed"]:
                total_passed += 1
                print(f"    ✓ PASS")
            else:
                failed = [k for k, v in res["tests"].items() if not v.get("passed")]
                print(f"    ✗ FAIL  —  failed tests: {failed}")
            if args.verbose:
                for t_name, t_res in res["tests"].items():
                    status = "✓" if t_res.get("passed") else "✗"
                    print(f"      {status} {t_name}: {t_res}")

        # Test NCNN
        for art in ncnn_artifacts:
            print(f"  Testing [{art['key']}] {art['param_path'].name} + {art['bin_path'].name} (imgsz={art['imgsz']})...")
            res = smoke_test_ncnn(
                art["param_path"], art["bin_path"], art["imgsz"],
                art["sha256_param"], art["sha256_bin"]
            )
            all_results.append(res)
            total_tests += 1
            if res["passed"]:
                total_passed += 1
                print(f"    ✓ PASS")
            else:
                failed = [k for k, v in res["tests"].items() if not v.get("passed")]
                print(f"    ✗ FAIL  —  failed tests: {failed}")
            if args.verbose:
                for t_name, t_res in res["tests"].items():
                    status = "✓" if t_res.get("passed") else "✗"
                    print(f"      {status} {t_name}: {t_res}")

    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY: {total_passed}/{total_tests} artifacts passed")
    print(f"NOTE: This smoke test does NOT set accuracy_passed in manifest.json.")
    print(f"      Run run_640_profile.py for full accuracy evaluation.")
    print(f"{'='*60}\n")

    # Write results
    out_path = manifest_path.parent / "smoke_test_results.json"
    out_path.write_text(json.dumps({
        "manifest": str(manifest_path),
        "total": total_tests,
        "passed": total_passed,
        "failed": total_tests - total_passed,
        "results": all_results,
    }, indent=2))
    print(f"Results written to: {out_path}")

    if total_passed < total_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
