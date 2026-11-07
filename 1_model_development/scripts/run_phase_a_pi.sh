#!/bin/bash
# Run Phase A benchmark on Raspberry Pi

set -e

echo "Phase A Raspberry Pi Benchmark"
echo "=============================="

# Activate venv (if available)
VENV_PATH="../../.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Python: $(python --version)"
echo "Hostname: $(hostname)"
echo ""

# Collect environment
echo "[1/2] Collecting environment information..."
python scripts/collect_environment.py || echo "⚠ Warning: Failed to collect environment"
echo ""

# Run benchmark
echo "[2/2] Running Pi benchmark..."
python scripts/benchmark_pi.py \
    --models yolov8n yolo11n yolo12n yolo26n \
    --images assets/export_validation \
    --warmup 50 \
    --runs 1000 \
    --repeat 3 \
    --output benchmarks/pi
echo ""

echo "=============================="
echo "✓ Pi benchmark complete"
echo "Results in: benchmarks/pi/benchmark_results.json"
