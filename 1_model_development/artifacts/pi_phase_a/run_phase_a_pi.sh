#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "Raspberry Pi Phase A Standalone Benchmark Runner"
echo "=================================================="

# Check python
PYTHON_BIN="${PYTHON:-python3}"
echo "Using Python: $($PYTHON_BIN --version 2>&1)"

# Step 1: Preflight & environment collection
echo ""
echo "[1/3] Collecting host/device environment..."
$PYTHON_BIN scripts/collect_environment.py

# Step 2: Run benchmark
echo ""
echo "[2/3] Running ONNX models benchmark..."
$PYTHON_BIN scripts/benchmark_pi.py \
    --models yolov8n yolo11n yolo12n yolo26n \
    --images assets \
    --onnx-dir weights/onnx \
    --manifest weights/manifest.json \
    --warmup 50 \
    --runs 1000 \
    --repeat 3 \
    --output benchmarks/pi

echo ""
echo "[3/3] Packaging results..."
cat << 'EOF' > summarize_pi.py
import json, csv, sys
from pathlib import Path

raw_file = Path("benchmarks/pi/benchmark_results.json")
if raw_file.exists():
    with open(raw_file) as f:
        data = json.load(f)
    rows = []
    for m, d in data.items():
        reps = d.get("repeats", [])
        p50s = [r["latency"]["p50_ms"] for r in reps if r.get("status") == "valid"]
        p95s = [r["latency"]["p95_ms"] for r in reps if r.get("status") == "valid"]
        temps = [r["environment_sample"]["end_temperature_c"] for r in reps if r["environment_sample"].get("end_temperature_c")]
        rows.append({
            "model": m,
            "status": d.get("status", "unknown"),
            "valid_repeats": len(p50s),
            "p50_ms": round(sum(p50s)/len(p50s), 2) if p50s else "N/A",
            "p95_ms": round(sum(p95s)/len(p95s), 2) if p95s else "N/A",
            "avg_temp_c": round(sum(temps)/len(temps), 1) if temps else "N/A"
        })
    out_csv = Path("benchmarks/pi/phase_a_pi_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "status", "valid_repeats", "p50_ms", "p95_ms", "avg_temp_c"])
        w.writeheader()
        w.writerows(rows)
    print("Summary written to " + str(out_csv))
EOF
$PYTHON_BIN summarize_pi.py
rm -f summarize_pi.py

echo "=================================================="
echo "✓ Pi benchmark completed successfully."
echo "Results:"
echo "  - benchmarks/pi/benchmark_results.json"
echo "  - benchmarks/pi/phase_a_pi_summary.csv"
echo "=================================================="
