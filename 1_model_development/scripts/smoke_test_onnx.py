#!/usr/bin/env python3
"""Smoke test for ONNX models.

Validates that ONNX models can:
- Load successfully
- Run inference
- Produce valid outputs (no NaN/Inf, valid bboxes, etc.)

Usage:
    python scripts/smoke_test_onnx.py --models yolov8n
    python scripts/smoke_test_onnx.py --models yolov8n yolo11n --source assets/smoke_test.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from models.onnx_detector import ONNXDetector


def smoke_test_onnx_model(
    onnx_path: Path,
    image_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict:
    """Smoke test a single ONNX model.
    
    Returns:
        dict with status, timings, detection count, and any errors
    """
    try:
        rel_path = onnx_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = onnx_path
    
    result = {
        "model": onnx_path.stem,
        "onnx_path": str(rel_path),
        "status": "failed",
        "error": None,
        "load_ms": 0,
        "inference_ms": 0,
        "detections": 0,
        "checks": {
            "load_successful": False,
            "inference_successful": False,
            "no_nan_inf": False,
            "valid_confidences": False,
            "valid_bboxes": False,
            "valid_class_ids": False,
        },
    }
    
    try:
        # Load model
        start_load = time.time()
        detector = ONNXDetector(str(onnx_path))
        load_time = (time.time() - start_load) * 1000
        result["load_ms"] = round(load_time, 2)
        result["checks"]["load_successful"] = True
        
        # Read image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        orig_h, orig_w = image.shape[:2]
        
        # Inference
        start_inference = time.time()
        detections = detector.predict(str(image_path), conf=conf, iou=iou)
        inference_time = (time.time() - start_inference) * 1000
        result["inference_ms"] = round(inference_time, 2)
        result["detections"] = len(detections)
        result["checks"]["inference_successful"] = True
        
        # Validate detections
        has_nan_inf = False
        valid_confidences = True
        valid_bboxes = True
        valid_class_ids = True
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            
            # Check for NaN/Inf
            if np.isnan([x1, y1, x2, y2, det.confidence]).any():
                has_nan_inf = True
            if np.isinf([x1, y1, x2, y2, det.confidence]).any():
                has_nan_inf = True
            
            # Check confidence range
            if not (0.0 <= det.confidence <= 1.0):
                valid_confidences = False
            
            # Check bbox validity
            if x2 <= x1 or y2 <= y1:
                valid_bboxes = False
            
            # Check bbox within image bounds
            if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h:
                # Allow small epsilon for floating point errors
                if not (x1 >= -1 and y1 >= -1 and x2 <= orig_w + 1 and y2 <= orig_h + 1):
                    valid_bboxes = False
            
            # Check class ID
            if det.class_id < 0 or det.class_id >= 80:  # COCO has 80 classes
                valid_class_ids = False
        
        result["checks"]["no_nan_inf"] = not has_nan_inf
        result["checks"]["valid_confidences"] = valid_confidences
        result["checks"]["valid_bboxes"] = valid_bboxes
        result["checks"]["valid_class_ids"] = valid_class_ids
        
        # Overall status
        all_checks_passed = all(result["checks"].values())
        
        if all_checks_passed:
            result["status"] = "passed"
        else:
            result["status"] = "failed"
            failed_checks = [k for k, v in result["checks"].items() if not v]
            result["error"] = f"Failed checks: {', '.join(failed_checks)}"
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke test ONNX models")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names (without _320_fp32.onnx suffix)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("assets/smoke_test.jpg"),
        help="Test image path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (optional)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("weights/onnx"),
        help="ONNX models directory",
    )
    
    args = parser.parse_args()
    
    # Check image exists
    if not args.source.exists():
        print(f"Error: Image not found: {args.source}")
        sys.exit(1)
    
    print(f"Smoke testing ONNX models")
    print(f"Image: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print(f"{'='*80}")
    
    # Test each model
    results = {}
    
    for model_name in args.models:
        # Build ONNX path: {model}_320_fp32.onnx
        onnx_filename = f"{model_name}_320_fp32.onnx"
        onnx_path = args.models_dir / onnx_filename
        
        if not onnx_path.exists():
            print(f"\n⚠ ONNX file not found: {onnx_path}, skipping")
            results[model_name] = {
                "model": model_name,
                "status": "not_found",
                "error": f"File not found: {onnx_path}",
            }
            continue
        
        print(f"\nTesting: {model_name}")
        print("-" * 80)
        
        result = smoke_test_onnx_model(onnx_path, args.source, args.conf, args.iou)
        results[model_name] = result
        
        # Print result
        if result["status"] == "passed":
            print(f"✓ Status: PASSED")
        else:
            print(f"✗ Status: FAILED")
            if result["error"]:
                print(f"  Error: {result['error']}")
        
        print(f"  Load time: {result['load_ms']} ms")
        print(f"  Inference time: {result['inference_ms']} ms")
        print(f"  Detections: {result['detections']}")
        
        print(f"  Checks:")
        for check_name, check_passed in result["checks"].items():
            status = "✓" if check_passed else "✗"
            print(f"    {status} {check_name}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    passed = [m for m, r in results.items() if r["status"] == "passed"]
    failed = [m for m, r in results.items() if r["status"] == "failed"]
    not_found = [m for m, r in results.items() if r["status"] == "not_found"]
    
    print(f"Passed:    {len(passed)}")
    print(f"Failed:    {len(failed)}")
    print(f"Not found: {len(not_found)}")
    
    if passed:
        print(f"\n✓ Passed models:")
        for m in passed:
            print(f"  - {m} ({results[m]['detections']} detections, {results[m]['inference_ms']} ms)")
    
    if failed:
        print(f"\n✗ Failed models:")
        for m in failed:
            print(f"  - {m}: {results[m].get('error', 'unknown')}")
    
    # Save results
    if args.output:
        output_data = {
            "test_image": str(args.source),
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "total_models": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "not_found": len(not_found),
            "results": results,
        }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output}")
    
    # Exit code
    if failed and not passed:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
