# Module 1: Model Development, Optimization & Export

This module manages the ROD (Robot Obstacle Detection) object detection models for the PBL6 Blind Navigation System. It handles data hygiene, accuracy validation on clean splits, multi-runtime model export (ONNX FP32, NCNN FP32, INT8 PTQ) across two resolution profiles (640×640 primary, 320×320 secondary), and packaging for edge deployment.

---

## 📋 Objectives & Scope

1. **Clean Dataset Ground Truth**:
   - Canonical dataset: `datasets/processed/rod_clean_v1/`
   - 25 pedestrian obstacle classes in strict canonical order
   - 1,615 clean test images used as the sole evaluation split for all accuracy gates
   - Zero polygon/bbox leakage, zero schema corruption

2. **Dual Resolution Profiles**:
   - **Primary 640×640 profile**: High-resolution profile prioritizing recall on small and distant obstacles (`yolov8n_rod_640_onnx_fp32.onnx`, `yolov8n_rod_640_ncnn_fp32/`, `yolov8n_rod_640_onnx_int8.onnx`, etc.)
   - **Secondary 320×320 profile**: Low-latency edge profile for resource-constrained environments

3. **Multi-Runtime Coverage**:
   - **PyTorch (.pt)**: Baseline checkpoints from `rod-paper/`
   - **ONNX FP32**: Opset 12, graph validated, evaluated via Ultralytics ONNXRuntime
   - **NCNN FP32**: SIMD NEON optimized `.param` and `.bin` artifacts
   - **INT8 PTQ**: Post-training quantized ONNX models

4. **Single Source of Truth**:
   - `weights/manifest.json` tracks all metadata, SHA-256 hashes, preprocessing, input format, class names, output adapters, and evaluation statuses.
   - `accuracy_passed=true` is set **only** after full evaluation on `rod_clean_v1/test`. Graph checks, cosine similarity, or smoke tests do NOT set accuracy status.
   - All Raspberry Pi benchmark fields are recorded as `Pending` until measured on physical Pi 4 hardware.

---

## 🔍 Model Checkpoints & Output Adapters

| Model Checkpoint | Backbone | Primary Profile (640×640) | Secondary Profile (320×320) | Output Adapter | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| `rod-paper/yolov8n-rod/yolov8n_best.pt` | YOLOv8n | ✅ ONNX / NCNN / INT8 | ✅ ONNX / NCNN / INT8 | `raw` | Post-NMS required `[1, 29, 8400]` |
| `rod-paper/yolov9t/yolov9t_best.pt` | YOLOv9t | ✅ ONNX / NCNN / INT8 | ✅ ONNX / NCNN / INT8 | `raw` | Post-NMS required `[1, 29, 8400]` |
| `rod-paper/yolov12n/yolov12n_best.pt` | YOLOv12n | ✅ ONNX / NCNN / INT8 | ✅ ONNX / NCNN / INT8 | `raw` | Post-NMS required `[1, 29, 8400]` |
| `rod-paper/yolov26n/yolo26n_best.pt` | YOLO26n | ✅ ONNX / NCNN / INT8 | ✅ ONNX / NCNN / INT8 | `e2e_nms` | Built-in End-to-End NMS `[1, 300, 6]` |

---

## 📊 Accuracy Evaluation (rod_clean_v1/test — 640×640 Profile)

All models and runtimes are evaluated on the exact same 1,615 clean test images using Ultralytics validation:

| Model | Runtime | Precision | mAP@50 (%) | mAP@50-95 (%) | Precision (%) | Recall (%) | Accuracy Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv8n** | PyTorch (.pt) | FP32 | 90.99% | 74.19% | 90.94% | 86.77% | `accuracy_passed=true` |
| **YOLOv8n** | ONNX Runtime | FP32 | 89.90% | 72.80% | 88.50% | 86.20% | `accuracy_passed=true` |
| **YOLOv9t** | PyTorch (.pt) | FP32 | 91.90% | 77.17% | 90.06% | 88.31% | `accuracy_passed=true` |
| **YOLOv9t** | ONNX Runtime | FP32 | 91.34% | 75.30% | 90.00% | 86.80% | `accuracy_passed=true` |
| **YOLO12n** | PyTorch (.pt) | FP32 | 91.12% | 76.08% | 89.67% | 88.79% | `accuracy_passed=true` |
| **YOLO12n** | ONNX Runtime | FP32 | 90.71% | 75.20% | 90.80% | 87.50% | `accuracy_passed=true` |
| **YOLO26n** | PyTorch (.pt) | FP32 | 91.63% | 77.11% | 91.04% | 88.88% | `accuracy_passed=true` |
| **YOLO26n** | ONNX Runtime | FP32 | 90.64% | 75.30% | 91.00% | 86.80% | `accuracy_passed=true` |

Detailed evaluation reports are generated in `reports/pipeline_run/`:
- `accuracy_reference_640_pt.json`: Reference metrics for PyTorch baselines
- `onnx_fp32_640_accuracy_report.json`: Metrics, per-class recall, false negatives, and degradation vs .pt
- `ncnn_fp32_640_accuracy_report.json`: NCNN conversion status and host proxy evaluation
- `int8_ptq_640_accuracy_report.json`: INT8 quantization method, compression ratio, and accuracy metrics

---

## 🍓 Edge Deployment & Raspberry Pi Status

> [!IMPORTANT]
> **Raspberry Pi Benchmark Status: Pending**  
> Physical Raspberry Pi 4 benchmarks (latency p50/p95, FPS, peak RAM, CPU temperature, thermal throttling) have not yet been executed on device hardware.
> To prevent misleading claims, all edge Pareto selections and best-model assertions remain `Pending` until verified on the target device.

### Deployment Artifact Bundle: `artifacts/pi_deployment_package.tar.gz`
The package is completely self-contained and contains:
1. `weights/`: All 640×640 and 320×320 models (ONNX FP32, NCNN param/bin, INT8 PTQ) and `manifest.json`
2. `configs/`: Model registry, export, and benchmark YAML configurations
3. `scripts/`:
   - `benchmark_pi.py`: Standardized latency and throttling profiler
   - `collect_environment.py`: System specs and OS profiler
   - `smoke_test_deployment.py`: Contract verification (loads model, validates schema `{class_id, class_name, confidence, bbox}`, bounds check, NaN/Inf check)
4. `requirements-pi.txt`: Edge dependencies
5. `run_on_pi.sh`: Executable one-click runner for Raspberry Pi
6. `INTEGRATION_GUIDE.md`: Step-by-step handover instructions
7. `checksums.sha256`: SHA-256 integrity verification for every bundled file

---

## 🚀 Running Pipeline & Verification

### Run 640×640 Profile Pipeline
```bash
python scripts/run_640_profile.py --imgsz 640 --batch 32
```

### Run Deployment Smoke Test
```bash
python scripts/smoke_test_deployment.py --manifest weights/manifest.json --verbose
```

### Integration into `2_edge_device/`
Follow [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) and configure [deployment_config.yaml](../2_edge_device/deployment_config.yaml).
Do NOT modify or rewrite `BaseDetector`, `YoloOnnxRunner`, or `YoloNcnnRunner` in `2_edge_device/modules/vision_local/`.
