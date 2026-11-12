# Best Model Inference Guide (320)

This guide uses the packaged 320x320 artifacts and the tested runner at `2_edge_device/best_model_inference.py`. Run commands from the workspace root (`/home/admin/Documents/pbl6`). The runner verifies manifest SHA-256 values before loading an artifact.

## Model Choices

| Purpose | Configuration | Test dangerous recall | Critical minimum | Pi main p50 | Size | Notes |
|---|---|---:|---:|---:|---:|---|
| Best measured recall / fastest | YOLO26n NCNN FP32 | 80.342% | 73.205% (Car) | 84.289 ms | 9.127 MB | Native NCNN 320 test exists. Accuracy differs from ONNX; validate the backend discrepancy before safety deployment. |
| Recommended edge balance | YOLO26n ONNX INT8 | 79.260% | 72.840% (Car) | 100.174 ms | 2.770 MB | Default runner configuration. Direct ONNX INT8 test evidence. |
| Highest ONNX FP32 recall | YOLO12n ONNX FP32 | 80.208% | 68.000% (Person) | 122.202 ms screening p50 | 9.973 MB | Strong macro recall, but lower worst critical-class recall and slower than the edge pick. |

Recall values are from the current full-test evaluation reports at the evaluator's F1-selected operating point. They are **not** results at a locked 320 deployment confidence threshold. The available operational-threshold report is 640px validation for YOLO12n and YOLO26n only. Therefore, these rankings are model-selection evidence, not a safety certification.

YOLO26n ONNX is exported with end-to-end NMS; the ONNX runner filters the model's detections by confidence and does not apply the CLI IoU value. Native NCNN YOLO26n returns raw `[29, 2100]` predictions and uses CPU NMS in the inference template. This backend/output-adapter difference can affect recall and explains why NCNN and ONNX results should not be assumed interchangeable.

## Setup

The model package and artifacts must be present at `pi_deployment_package/`. On a Raspberry Pi, install its dependencies in the active environment:

```bash
python -m pip install -r pi_deployment_package/requirements-pi.txt
```

If you copied the package elsewhere, pass its path with `--package-root /path/to/pi_deployment_package`.

## Run Inference

Default: YOLO26n ONNX INT8 on the bundled sample image. Confidence 0.25 and IoU 0.45 are the current deployment defaults, not a threshold calibrated on 320 validation data.

```bash
python blind-navigation/2_edge_device/best_model_inference.py \
  --source pi_deployment_package/assets/sample_rod.jpg
```

Use the native NCNN performance/recall candidate:

```bash
python blind-navigation/2_edge_device/best_model_inference.py \
  --source pi_deployment_package/assets/sample_rod.jpg \
  --model yolov26n --backend ncnn --precision fp32
```

Use the highest-dangerous-recall ONNX FP32 candidate from the 320 test metrics:

```bash
python blind-navigation/2_edge_device/best_model_inference.py \
  --source pi_deployment_package/assets/sample_rod.jpg \
  --model yolov12n --backend onnx --precision fp32
```

Process a video, optionally show annotated frames and save an annotated copy:

```bash
python blind-navigation/2_edge_device/best_model_inference.py \
  --source example.mp4 --display --save-video outputs/example_detected.mp4
```

Run a camera (source `0` is the first camera):

```bash
python blind-navigation/2_edge_device/best_model_inference.py \
  --source 0 --display --model yolov26n --backend onnx --precision int8
```

Set an explicit threshold while validating your deployment scene:

```bash
python blind-navigation/2_edge_device/best_model_inference.py \
  --source 0 --display --model yolov26n --backend onnx --precision int8 \
  --conf 0.25 --iou 0.45 --threads 4
```

The process prints the selected artifact/adapter and JSON detections. For streams, `--display` opens a window; `--save-video` writes annotated output. Without either option, it runs headless and prints frames with detections.

## Artifact Mapping

The runner reads files and SHA-256 values from `pi_deployment_package/weights/manifest.json`:

| CLI selection | Artifact |
|---|---|
| `--model yolov26n --backend onnx --precision int8` | `weights/int8/yolov26n_320_int8.onnx` |
| `--model yolov26n --backend onnx --precision fp32` | `weights/onnx/yolov26n_320_fp32.onnx` |
| `--model yolov12n --backend onnx --precision fp32` | `weights/onnx/yolov12n_320_fp32.onnx` |
| `--model yolov26n --backend ncnn --precision fp32` | `weights/ncnn/yolov26n_ncnn_fp32/model.ncnn.param` + `.bin` |

Other packaged 320 model/runtime combinations are also selectable with `--model`, `--backend`, and `--precision`; NCNN is FP32 only.

## Before Real Deployment

1. Select the confidence threshold on a 320 validation split using the specified precision constraint, then freeze it.
2. Evaluate the locked threshold once on the held-out test split and report dangerous recall, critical minimum, false negatives, and false positives per image.
3. For YOLO26n NCNN, compare its raw-output decode, NMS settings, boxes, and per-class metrics against ONNX. Do not copy ONNX thresholds/accuracy assumptions to NCNN.
4. Exercise the exact camera, lighting, and operating conditions, and verify thermal/power behavior on the target Pi.

The previous `modules/vision_local/yolo_ncnn_runner.py` is a legacy YOLOv8/COCO example. It uses different preprocessing/output assumptions and is **not** used by this runner or suitable for these ROD 320 artifacts without correction.
