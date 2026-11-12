# PBL6 Blind Navigation — Integration Guide

> **Target audience**: Developers integrating ROD model artifacts into `2_edge_device/`.

> **320 best-model update:** This document contains legacy examples and must not be used as-is for the current 320 ROD artifacts. In particular, its old ONNX/NCNN runner examples do not match the current edge implementation. Use [BEST_MODEL_INFERENCE.md](../2_edge_device/BEST_MODEL_INFERENCE.md) and its checksum-verifying `best_model_inference.py` template for the benchmarked 320 configurations. The 640 guidance below is retained for reference.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step 1 — Select an Artifact from manifest.json](#2-step-1--select-an-artifact-from-manifestjson)
3. [Step 2 — Verify SHA-256 Integrity](#3-step-2--verify-sha-256-integrity)
4. [Step 3 — Copy/Link Model Files](#4-step-3--copylink-model-files)
5. [Step 4 — Configure Backend](#5-step-4--configure-backend)
6. [Step 5 — Run Smoke Test on Static Images](#6-step-5--run-smoke-test-on-static-images)
7. [Step 6 — Connect Webcam](#7-step-6--connect-webcam)
8. [Class Labels](#8-class-labels)
9. [Output Adapter Reference](#9-output-adapter-reference)
10. [Raspberry Pi Notes](#10-raspberry-pi-notes)

---

## 1. Overview

The model artifacts exported by `run_640_profile.py` and `execute_full_rod_pipeline.py` are:

| Format          | Naming Pattern                          | Location                     |
|-----------------|-----------------------------------------|------------------------------|
| ONNX FP32       | `{model}_rod_{imgsz}_onnx_fp32.onnx`   | `weights/onnx/`              |
| NCNN FP32       | `{model}_rod_{imgsz}_ncnn_fp32/`       | `weights/ncnn/`              |
| INT8 PTQ (ONNX) | `{model}_rod_{imgsz}_onnx_int8.onnx`   | `weights/int8/`              |

**Single source of truth**: `weights/manifest.json`  
Only use artifacts that have `accuracy_passed: true` in `manifest.json`.

---

## 2. Step 1 — Select an Artifact from manifest.json

```python
import json
from pathlib import Path

manifest = json.loads(Path("1_model_development/weights/manifest.json").read_text())

# Example: select yolov8n ONNX FP32 at 640x640
entry = manifest["models"]["yolov8n"]["onnx_fp32_640"]
print(entry["path_relative"])   # weights/onnx/yolov8n_rod_640_onnx_fp32.onnx
print(entry["sha256"])           # SHA-256 hex string
print(entry["accuracy_passed"])  # Must be True before use!
print(entry["imgsz"])            # 640
print(entry["output_adapter"])   # "raw" or "e2e_nms"
```

**Available artifact keys per model**:
- `onnx_fp32_320`, `onnx_fp32_640`
- `onnx_int8_320`, `onnx_int8_640`
- `ncnn_fp32_320`, `ncnn_fp32_640`

---

## 3. Step 2 — Verify SHA-256 Integrity

Always verify the SHA-256 of a downloaded or copied artifact before loading it.

```python
import hashlib
from pathlib import Path

def verify_sha256(artifact_path: str, expected: str) -> bool:
    h = hashlib.sha256()
    with open(artifact_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch!\n  Expected: {expected}\n  Actual  : {actual}")
    return True

# Usage
verify_sha256(
    "1_model_development/weights/onnx/yolov8n_rod_640_onnx_fp32.onnx",
    entry["sha256"]
)
print("✓ Integrity check passed")
```

You can also run the bundled smoke test:
```bash
python 1_model_development/scripts/smoke_test_deployment.py \
    --manifest 1_model_development/weights/manifest.json \
    --verbose
```

---

## 4. Step 3 — Copy/Link Model Files

Copy (or symlink) the selected artifacts into `2_edge_device/modules/vision_local/models/`:

```bash
# ONNX
mkdir -p 2_edge_device/modules/vision_local/models/onnx
cp 1_model_development/weights/onnx/yolov8n_rod_640_onnx_fp32.onnx \
   2_edge_device/modules/vision_local/models/onnx/

# NCNN (copy entire directory)
mkdir -p 2_edge_device/modules/vision_local/models/ncnn
cp -r 1_model_development/weights/ncnn/yolov8n_rod_640_ncnn_fp32 \
      2_edge_device/modules/vision_local/models/ncnn/
```

> **Do not rename files** after copying — the manifest and deployment_config.yaml reference exact filenames.

---

## 5. Step 4 — Configure Backend

### ONNX Backend (using existing `OnnxDetector`)

```python
# Do NOT rewrite OnnxDetector — use it as-is from 2_edge_device/modules/vision_local/
from modules.vision_local.onnx_detector import OnnxDetector

ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]

detector = OnnxDetector(
    onnx_path="modules/vision_local/models/onnx/yolov8n_rod_640_onnx_fp32.onnx",
    class_names=ROD_CLASSES,
)

# For yolov26n (e2e_nms output adapter), pass adapter="e2e_nms" if OnnxDetector supports it
# detector_26n = OnnxDetector(..., adapter="e2e_nms")
```

**Input spec**:
- RGB float32 tensor `[1, 3, H, W]` where H=W=640 (or 320)
- Normalize: `pixel / 255.0` (no ImageNet mean subtraction)
- Letterbox: preserve aspect ratio, pad with 114 (grey)

### NCNN Backend (using existing `YoloNcnnRunner`)

```python
# Do NOT rewrite YoloNcnnRunner — use it as-is
from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner

runner = YoloNcnnRunner(
    param_path="modules/vision_local/models/ncnn/yolov8n_rod_640_ncnn_fp32/model.ncnn.param",
    bin_path="modules/vision_local/models/ncnn/yolov8n_rod_640_ncnn_fp32/model.ncnn.bin",
    target_size=640,  # match imgsz from manifest
)
```

> **Note**: `YoloNcnnRunner` currently uses COCO class labels. Update `self.classes` to `ROD_CLASSES`
> in your initialization (without modifying the base class).

```python
runner.classes = ROD_CLASSES  # Override after init — do not rewrite YoloNcnnRunner
```

---

## 6. Step 5 — Run Smoke Test on Static Images

Before connecting a webcam, verify the detector works on a static test image:

```python
import cv2
import numpy as np

# Load sample image (bundled in pi_deployment_package.tar.gz → assets/sample_rod.jpg)
img = cv2.imread("assets/sample_rod.jpg")

# ONNX path
detections = detector.detect(img)

# Verify output schema
for d in detections:
    assert isinstance(d["class_id"], int), "class_id must be int"
    assert isinstance(d["class_name"], str), "class_name must be str"
    assert isinstance(d["confidence"], float), "confidence must be float"
    assert len(d["bbox"]) == 4, "bbox must be [x1, y1, x2, y2]"
    x1, y1, x2, y2 = d["bbox"]
    h, w = img.shape[:2]
    assert 0 <= x1 <= w and 0 <= x2 <= w, f"x coordinates out of bounds"
    assert 0 <= y1 <= h and 0 <= y2 <= h, f"y coordinates out of bounds"

print(f"✓ {len(detections)} detections, schema OK")
```

Or use the bundled script:
```bash
python 1_model_development/scripts/smoke_test_deployment.py \
    --manifest 1_model_development/weights/manifest.json
```

---

## 7. Step 6 — Connect Webcam

Only after Step 5 passes on static images:

```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    detections = detector.detect(frame)
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("ROD Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## 8. Class Labels

25 ROD-Dataset classes in exact index order (0–24):

| ID | Class              | ID | Class              |
|----|--------------------|----|-------------------|
| 0  | Bike               | 13 | Guard rail         |
| 1  | Building           | 14 | Pedestrian crosswalk |
| 2  | Car                | 15 | Truck              |
| 3  | Person             | 16 | Bus                |
| 4  | Stairs             | 17 | Bench              |
| 5  | Traffic sign       | 18 | Traffic Cone       |
| 6  | Electrical Pole    | 19 | Fire hydrant       |
| 7  | Road               | 20 | Teraffic Barrel    |
| 8  | Motorcycle         | 21 | Plant Pot          |
| 9  | Dustbin            | 22 | Electrical Box     |
| 10 | Dog                | 23 | Chair              |
| 11 | Manhole            | 24 | Bicycle Rack       |
| 12 | Tree               |    |                    |

---

## 9. Output Adapter Reference

| Model    | Adapter    | Output Shape           | NMS Needed? |
|----------|------------|------------------------|-------------|
| yolov8n  | `raw`      | `[1, 29, 8400]` @640   | Yes         |
| yolov9t  | `raw`      | `[1, 29, 8400]` @640   | Yes         |
| yolov12n | `raw`      | `[1, 29, 8400]` @640   | Yes         |
| yolov26n | `e2e_nms`  | `[1, 300, 6]`          | No          |

The `OnnxDetector` in `2_edge_device/modules/vision_local/onnx_detector.py` automatically
detects the output format. No changes needed.

---

## 10. Raspberry Pi Notes

> **Status: Pending**  
> No Pi 4 benchmark data has been collected. All `raspberry_pi_benchmark` fields in
> `manifest.json` are `"Pending"`.

**To run on Pi**:
1. Transfer `pi_deployment_package.tar.gz` to Raspberry Pi 4.
2. Extract and run:
   ```bash
   tar -xzf pi_deployment_package.tar.gz
   cd pi_deployment_package
   ./run_on_pi.sh
   ```
3. This will:
   - Install `requirements-pi.txt`
   - Run `smoke_test_deployment.py`
   - Run `benchmark_pi.py` (latency, FPS, temperature, throttling)
4. Copy the `benchmarks_pi/` output back to the repo and update `manifest.json`.

**Recommended first deployment** (before benchmark data): `320×320 ONNX FP32`
(smaller memory footprint, confirmed lower latency estimate from host).  
Switch to `640×640` if small-object recall is insufficient after testing.
