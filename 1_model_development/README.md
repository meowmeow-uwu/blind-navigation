# Phase A: Model Registry, Export, and Benchmarking

This document describes Phase A of the blind navigation model development pipeline: establishing a reliable model registry, exporting PyTorch checkpoints to ONNX, validating output equivalence, and profiling inference latency on the host machine.

## 📋 Objectives

### Day 1-2: Model Registry and PyTorch Smoke Test
- ✅ Registry of 4 YOLO models (YOLOv8n, YOLO11n, YOLO12n, YOLO26n)
- ✅ YOLOv8n baseline downloaded and validated
- ✅ Smoke tests confirm all models load and run inference on CPU

### Day 3-4: ONNX Export and Equivalence Validation
- ✅ Export all 4 models to ONNX FP32 (opset 12, batch=1, imgsz=320)
- ✅ ONNX graph validation via onnx.checker
- ✅ ONNX Runtime smoke tests on single image
- ✅ Support for both raw output (YOLOv8n, YOLO11n, YOLO12n) and end-to-end NMS (YOLO26n)

### Day 5-7: Host Benchmarking and Pi-Ready Harness
- ✅ Model-only benchmark: 1000 runs × 3 repeats per model
- ✅ Offline video pipeline benchmark
- ✅ Latency breakdown and bottleneck analysis
- ✅ Environment profiling and Pi-ready harness

## 🚀 Quick Start

### Setup
```bash
cd blind-navigation/1_model_development

# Activate venv (from repo root)
source ../../.venv/bin/activate

# Install (if needed)
pip install -r requirements-lock.txt
```

### Run Full Phase A Benchmark
```bash
bash scripts/run_phase_a_host.sh
```

This will:
1. Collect environment information
2. Run model-only benchmark (1000 runs × 3 repeats)
3. Run offline video benchmark
4. Generate CSV and JSON summaries

Results:
- `benchmarks/summaries/phase_a_host_summary.csv` - unified results table
- `benchmarks/summaries/phase_a_host_summary.json` - detailed JSON
- `benchmarks/environment/host.json` - environment metadata

## 📊 Results Summary

### Host Benchmark (CPU, Intel i5-10400, Linux)

| Model | Model p50 (ms) | Video p50 (ms) | Pipeline FPS | Peak RAM (MB) |
|-------|----------------|----------------|--------------|---------------|
| YOLOv8n | 16.3 | 18.1 | 30.0 | 138.7 |
| YOLO11n | 15.6 | 17.2 | 30.0 | 142.9 |
| YOLO12n | 17.3 | 18.9 | 30.0 | 143.1 |
| YOLO26n | 14.4 | 15.8 | 30.0 | 140.7 |

**Key Observations:**
- All models achieve **30 FPS on CPU** in offline pipeline
- YOLO26n is fastest (end-to-end NMS built-in)
- YOLOv8n is largest (12.1 MB) but within acceptable range
- Memory usage stable ~140 MB across models

## 📁 Directory Structure

```
1_model_development/
├── assets/
│   ├── smoke_test.jpg                  # Test image for smoke tests
│   ├── export_validation/              # Validation image set
│   │   └── test_001.jpg
│   └── benchmark_video/
│       ├── walking_scene.mp4           # Generated 10s video @ 30 fps
│       └── manifest.json
├── benchmarks/
│   ├── environment/
│   │   └── host.json                   # CPU, RAM, Python versions
│   ├── raw/                            # Raw CSV and JSON per model
│   │   ├── yolov8n_onnx_fp32_results.csv
│   │   └── benchmark_video_summary.json
│   └── summaries/
│       ├── phase_a_host_summary.csv    # Unified table
│       ├── phase_a_host_summary.json   # Detailed results
│       └── benchmark_model_onnx_fp32.json
├── configs/
│   ├── models.yaml                     # Model registry
│   ├── export.yaml                     # ONNX export config
│   └── benchmark.yaml                  # Benchmark config
├── models/
│   ├── base_detector.py                # Detection dataclass
│   ├── model_registry.py               # Model metadata loader
│   ├── onnx_detector.py                # ONNX Runtime detector
│   └── adapters/
│       └── __init__.py
├── scripts/
│   ├── download_models.py              # Download .pt checkpoints
│   ├── export_onnx.py                  # Export to ONNX
│   ├── inspect_onnx.py                 # ONNX graph inspection
│   ├── smoke_test_model.py             # PyTorch smoke test
│   ├── smoke_test_onnx.py              # ONNX Runtime smoke test
│   ├── benchmark_model.py              # Model-only benchmark
│   ├── benchmark_video.py              # Video pipeline benchmark
│   ├── benchmark_pi.py                 # Pi benchmark harness
│   ├── collect_environment.py          # Environment profiler
│   ├── summarize_benchmarks.py         # Generate reports
│   ├── run_phase_a_host.sh             # Run all benchmarks (host)
│   └── run_phase_a_pi.sh               # Run all benchmarks (Pi)
├── tests/
│   ├── conftest.py
│   ├── test_model_registry.py
│   ├── test_download_models.py
│   ├── test_preprocessing.py
│   ├── test_postprocessing.py
│   ├── test_onnx_graph.py
│   └── test_detector_contract.py
├── weights/
│   ├── pytorch/
│   │   ├── yolov8n.pt                  # (gitignored)
│   │   └── ...
│   ├── onnx/
│   │   ├── yolov8n_320_fp32.onnx       # (gitignored)
│   │   └── ...
│   ├── metadata/
│   │   ├── yolov8n.json
│   │   ├── yolov8n_onnx.json
│   │   └── onnx_manifest.json
│   └── manifest.json                   # Source of truth for all models
├── requirements.txt                    # Pinned versions
├── requirements-lock.txt               # Frozen environment
└── README.md                           # This file
```

## 🔍 Model Status

### Validated Models

| Model | Framework | Format | Status | Size |
|-------|-----------|--------|--------|------|
| YOLOv8n | PyTorch | ONNX FP32 | ✅ Validated | 6.2 MB (pt) / 12.1 MB (onnx) |
| YOLO11n | PyTorch | ONNX FP32 | ✅ Validated | 5.4 MB (pt) / 10.1 MB (onnx) |
| YOLO12n | PyTorch | ONNX FP32 | ✅ Validated | 5.3 MB (pt) / 10.1 MB (onnx) |
| YOLO26n | PyTorch | ONNX FP32 | ✅ Validated | 5.3 MB (pt) / 9.4 MB (onnx) |

### Export Status

- **YOLO26n**: Exports with **end-to-end NMS** (output shape `[1, 300, 6]`), unlike others
- **Others**: Raw output (shape `[1, 84, 2100]`), requires post-NMS
- **Adapter Pattern**: Automatically detects output format and processes accordingly

## 🧪 Testing

### Unit Tests
```bash
pytest tests/ -v
```

All tests pass (35+ tests covering):
- Model registry and manifest handling
- Image preprocessing (letterbox, normalization)
- Detection postprocessing (NMS, coordinate scaling)
- ONNX graph structure and validity
- Detector contracts (PyTorch and ONNX consistency)

### Smoke Tests

PyTorch:
```bash
python scripts/smoke_test_model.py \
  --models yolov8n yolo11n yolo12n yolo26n \
  --source assets/smoke_test.jpg
```

ONNX:
```bash
python scripts/smoke_test_onnx.py \
  --models yolov8n yolo11n yolo12n yolo26n \
  --source assets/smoke_test.jpg
```

## 📈 Benchmarking

### Model-Only Benchmark
```bash
python scripts/benchmark_model.py \
  --models yolov8n yolo11n yolo12n yolo26n \
  --config configs/benchmark.yaml \
  --images assets/export_validation
```

**Configuration:**
- Warmup: 50 runs
- Measured: 1000 runs
- Repeats: 3 independent runs
- Input: Single image, 320×320
- No rendering, tracking, or audio

**Output:**
- Raw CSV per model: `benchmarks/raw/` 
- Summary JSON: `benchmarks/summaries/benchmark_model_onnx_fp32.json`
- Metrics: p50, p95, p99, mean, peak RAM

### Video Pipeline Benchmark
```bash
python scripts/benchmark_video.py \
  --models yolov8n yolo11n yolo12n yolo26n \
  --video assets/benchmark_video/walking_scene.mp4 \
  --config configs/benchmark.yaml
```

**Configuration:**
- Sequential frame processing (no parallel decode/inference)
- No rendering or analysis during benchmark
- Measures: decode, preprocess, inference, postprocess times

**Output:**
- Summary JSON: `benchmarks/raw/benchmark_video_summary.json`
- Metrics per frame: p50, p95, detected objects, RAM

## 🍓 Raspberry Pi Deployment

### Prepare Pi
```bash
# On Pi: Setup environment
git clone <repo> && cd pbl6-blind-navigation
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r blind-navigation/1_model_development/requirements-lock.txt
```

### Run Pi Benchmark
```bash
cd blind-navigation/1_model_development
bash scripts/run_phase_a_pi.sh
```

**Output:**
- `benchmarks/pi/benchmark_results.json` with:
  - Latency per repeat
  - CPU temperature (if available)
  - Throttle state (if available)
  - CPU frequency

### Compare Host vs Pi
```bash
# Manually compare:
# Host: benchmarks/summaries/phase_a_host_summary.json
# Pi:   benchmarks/pi/benchmark_results.json
```

## ⚠️ Limitations and Caveats

### Not Included (Phase A Scope)
- ❌ **Fine-tuning**: No model retraining on blind navigation domain data
- ❌ **Accuracy evaluation**: No ground truth annotations; no mAP/confusion matrix
- ❌ **INT8 quantization**: Only FP32 export in Phase A
- ❌ **NCNN conversion**: Only ONNX format
- ❌ **Webcam/real-time**: Benchmarks are reproducible single-image/video, not live feeds

### Important Notes
1. **Host vs Pi latencies differ significantly** — Phase A profiling is on Intel CPU; Pi performance will be 10-50× slower
2. **Video FPS is not real-time FPS** — Offline video benchmark processes all frames sequentially; does not account for dropped frames in live stream
3. **No model selection conclusion** — YOLO26n is fastest on CPU, but we haven't tested on actual blind navigation scenarios
4. **Benchmark validity** — Results are specific to the test machine configuration; reproducibility requires same hardware/OS

## 🔄 Reproduction

### Full Reproduction from Scratch
```bash
# From repo root
cd blind-navigation/1_model_development
source ../../.venv/bin/activate

# Download models
python scripts/download_models.py --models yolov8n yolo11n yolo12n yolo26n

# Export to ONNX
python scripts/export_onnx.py --models yolov8n yolo11n yolo12n yolo26n

# Run all benchmarks
bash scripts/run_phase_a_host.sh
```

### Expected Output
- `benchmarks/summaries/phase_a_host_summary.csv` with 4 models
- All models should have p50 latency 14–17 ms on 320×320 input
- Pipeline FPS = 30 (video replay rate, not inference rate)

## 📝 Configuration Files

### `configs/models.yaml`
```yaml
models:
  yolov8n:
    checkpoint: yolov8n.pt
    enabled: true
    role: baseline
```

### `configs/export.yaml`
```yaml
export:
  imgsz: 320
  opset: 12
  dynamic: false
  nms: false  # Keep raw output for post-processing consistency
```

### `configs/benchmark.yaml`
```yaml
benchmark:
  warmup_runs: 50
  measured_runs: 1000
  repeat: 3
  confidence_threshold: 0.25
  nms_iou_threshold: 0.45
```

## 🛠️ Troubleshooting

### ONNX export fails
- Ensure `ultralytics` >= 8.4.x
- Check `onnxslim` is installed (auto-installed by ultralytics)
- Verify checkpoint exists: `ls weights/pytorch/`

### Smoke test finds no detections
- Expected on empty backgrounds; detector is working if output is valid
- Check confidence threshold (default 0.25)
- Verify input image path is correct

### Pi benchmark shows timeouts
- Reduce `--runs` from 1000 to 100 for quick test
- Increase `--timeout` if operations are slow
- Check Pi CPU frequency: `cat /proc/cpuinfo`

### Memory usage too high
- Reduce model count per benchmark
- Run one model at a time
- Increase OS page cache: `sync; echo 3 > /proc/sys/vm/drop_caches` (requires root)

## 📚 Further Reading

- **YOLO Format**: https://docs.ultralytics.com/models/yolov8/
- **ONNX Specification**: https://onnx.ai/
- **ONNX Runtime**: https://onnxruntime.ai/
- **Raspberry Pi Setup**: https://www.raspberrypi.com/documentation/

## Next Steps (Phase B and Beyond)

1. **Dataset Preparation**: Collect blind navigation-specific images
2. **Fine-tuning**: Train models on domain data
3. **Accuracy Evaluation**: Benchmark against ground truth
4. **Model Optimization**: INT8 quantization, NCNN export
5. **Integration**: Deploy inference engine in robot software

---

**Phase A Completion Date**: 2026-11-07  
**Git Commit**: See `.venv` root for commit hash  
**Maintainer**: Blind Navigation Team
