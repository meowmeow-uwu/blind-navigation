#!/bin/bash
# Run full Phase A benchmark suite on host machine

set -e

echo "Phase A Host Benchmark Suite"
echo "============================"

# Activate venv
VENV_PATH="../../.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Python: $(python --version)"
echo ""

# 1. Collect environment
echo "[1/5] Collecting environment information..."
python scripts/collect_environment.py || echo "⚠ Warning: Failed to collect environment"
echo ""

# 2. Check models
echo "[2/5] Verifying model manifest..."
if [ ! -f "weights/manifest.json" ]; then
    echo "Error: weights/manifest.json not found"
    exit 1
fi
python -c "import json; m=json.load(open('weights/manifest.json')); print(f'Found {len(m[\"models\"])} models')"
echo ""

# 3. Model-only benchmark
echo "[3/5] Running model-only benchmark..."
python scripts/benchmark_model.py \
    --models yolov8n yolo11n yolo12n yolo26n \
    --config configs/benchmark.yaml \
    --images assets/export_validation || echo "⚠ Model benchmark failed"
echo ""

# 4. Video benchmark
echo "[4/5] Running offline video benchmark..."
python scripts/benchmark_video.py \
    --models yolov8n yolo11n yolo12n yolo26n \
    --video assets/benchmark_video/walking_scene.mp4 \
    --config configs/benchmark.yaml || echo "⚠ Video benchmark failed"
echo ""

# 5. Summarize results
echo "[5/5] Generating summary reports..."
python scripts/summarize_benchmarks.py
echo ""

echo "============================"
echo "✓ Phase A benchmark complete"
echo "Results in:"
echo "  - benchmarks/summaries/phase_a_host_summary.csv"
echo "  - benchmarks/summaries/phase_a_host_summary.json"
echo "  - benchmarks/environment/host.json"
