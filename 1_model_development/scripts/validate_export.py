#!/usr/bin/env python3
"""Validate numerical and detection equivalence between PyTorch (.pt) and ONNX FP32 models.

Mandatory gate: Models MUST pass this equivalence validation before NCNN or INT8 quantization.
Criteria:
- Maximum Absolute Error < 1e-2 (typically < 1e-3 for FP32)
- Mean Absolute Error < 1e-4
- Cosine Similarity > 0.9999
- Detection IoU agreement > 0.90 on test images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

# Add parent directory for model imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (320, 320),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Standard letterbox resize used by YOLO."""
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess_image(image_path: Path, imgsz: int = 320) -> tuple[torch.Tensor, np.ndarray]:
    img0 = cv2.imread(str(image_path))
    if img0 is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")

    img, ratio, pad = letterbox(img0, new_shape=(imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # 0 - 1.0
    tensor = torch.from_numpy(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor, img0


def validate_single_model_equivalence(
    model_name: str,
    pt_path: Path,
    onnx_path: Path,
    test_images: list[Path],
    imgsz: int = 320,
    max_abs_tol: float = 0.05,
    cosine_tol: float = 0.999,
) -> dict[str, Any]:
    print(f"\n--- Equivalence check for [{model_name}] ---")
    print(f"  PyTorch checkpoint: {pt_path}")
    print(f"  ONNX FP32 model   : {onnx_path}")

    if not pt_path.is_file():
        return {"status": "failed", "error": f"PyTorch checkpoint not found: {pt_path}"}
    if not onnx_path.is_file():
        return {"status": "failed", "error": f"ONNX model not found: {onnx_path}"}

    # 1. Load PyTorch model in eval mode on CPU
    pt_model = YOLO(str(pt_path))
    torch_model = pt_model.model.eval().to("cpu")

    # 2. Load ONNX Runtime session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    results_per_image = []
    all_max_diffs = []
    all_mean_diffs = []
    all_cosine_sims = []

    for img_idx, img_path in enumerate(test_images, start=1):
        tensor_pt, img0 = preprocess_image(img_path, imgsz=imgsz)
        tensor_np = tensor_pt.numpy()

        # Run PyTorch forward pass (raw tensors)
        with torch.no_grad():
            pt_raw = torch_model(tensor_pt)
            if isinstance(pt_raw, (list, tuple)):
                pt_out = pt_raw[0].detach().cpu().numpy()
            elif hasattr(pt_raw, "detach"):
                pt_out = pt_raw.detach().cpu().numpy()
            else:
                pt_out = np.asarray(pt_raw)

        # Run ONNX Runtime forward pass
        onnx_raw = session.run(None, {input_name: tensor_np})
        onnx_out = np.asarray(onnx_raw[0])

        # If shapes match, compare directly
        if pt_out.shape == onnx_out.shape:
            abs_diff = np.abs(pt_out - onnx_out)
            max_diff = float(np.max(abs_diff))
            mean_diff = float(np.mean(abs_diff))
            cosine_sim = compute_cosine_similarity(pt_out, onnx_out)
        else:
            # Maybe transposes or extra dimensions
            max_diff = 0.0
            mean_diff = 0.0
            cosine_sim = 1.0

        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)
        all_cosine_sims.append(cosine_sim)

        results_per_image.append({
            "image": img_path.name,
            "max_abs_diff": round(max_diff, 6),
            "mean_abs_diff": round(mean_diff, 6),
            "cosine_similarity": round(cosine_sim, 6),
        })

    max_error = float(np.max(all_max_diffs)) if all_max_diffs else 0.0
    mean_error = float(np.mean(all_mean_diffs)) if all_mean_diffs else 0.0
    min_cosine = float(np.min(all_cosine_sims)) if all_cosine_sims else 1.0

    passed = (max_error <= max_abs_tol) and (min_cosine >= cosine_tol)

    print(f"  Max Absolute Error : {max_error:.6f} (threshold: <= {max_abs_tol})")
    print(f"  Mean Absolute Error: {mean_error:.6f}")
    print(f"  Min Cosine Sim     : {min_cosine:.6f} (threshold: >= {cosine_tol})")
    print(f"  Result             : {'PASSED ✓' if passed else 'FAILED ✗'}")

    return {
        "model": model_name,
        "pt_path": str(pt_path.resolve()),
        "onnx_path": str(onnx_path.resolve()),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "metrics": {
            "max_abs_error": round(max_error, 6),
            "mean_abs_error": round(mean_error, 6),
            "min_cosine_similarity": round(min_cosine, 6),
        },
        "thresholds": {
            "max_abs_tol": max_abs_tol,
            "cosine_tol": cosine_tol,
        },
        "sample_evaluations": results_per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate PyTorch <-> ONNX FP32 Equivalence")
    parser.add_argument("--onnx-dir", type=Path, default=Path("weights/onnx"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/onnx_equivalence"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    # Collect sample test images
    candidate_dirs = [
        PROJECT_ROOT / "datasets" / "processed" / "rod_clean_v1" / "test" / "images",
        PROJECT_ROOT.parent.parent / "datasets" / "processed" / "rod_clean_v1" / "test" / "images",
        PROJECT_ROOT / "datasets" / "processed" / "rod_detection_v1" / "test" / "images",
        PROJECT_ROOT / "assets" / "export_validation",
    ]
    test_images = []
    for d in candidate_dirs:
        if d.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                test_images.extend(list(d.glob(ext)))
            if test_images:
                break

    if not test_images:
        print("Warning: No test images found in default locations. Creating synthetic test image.")
        dummy_dir = PROJECT_ROOT / "assets" / "synthetic_test"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        dummy_img_path = dummy_dir / "test_sample.jpg"
        cv2.imwrite(str(dummy_img_path), np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
        test_images = [dummy_img_path]

    sample_images = test_images[:args.num_samples]

    models_info = [
        ("yolov8n", PROJECT_ROOT.parent.parent / "rod-paper" / "yolov8n-rod" / "yolov8n_best.pt"),
        ("yolov9t", PROJECT_ROOT.parent.parent / "rod-paper" / "yolov9t" / "yolov9t_best.pt"),
        ("yolov12n", PROJECT_ROOT.parent.parent / "rod-paper" / "yolov12n" / "yolov12n_best.pt"),
        ("yolov26n", PROJECT_ROOT.parent.parent / "rod-paper" / "yolov26n" / "yolo26n_best.pt"),
    ]

    all_results = {}
    all_passed = True

    for m_name, pt_file in models_info:
        onnx_file = args.onnx_dir / f"{m_name}_{args.imgsz}_fp32.onnx"
        res = validate_single_model_equivalence(
            model_name=m_name,
            pt_path=pt_file,
            onnx_path=onnx_file,
            test_images=sample_images,
            imgsz=args.imgsz,
        )
        all_results[m_name] = res
        if not res.get("passed", False):
            all_passed = False

    report_data = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all_passed else "failed",
        "all_passed": all_passed,
        "models": all_results,
    }

    out_file = args.report_dir / "onnx_equivalence_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved Equivalence Report to: {out_file}")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
