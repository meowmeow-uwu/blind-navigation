#!/usr/bin/env python3
"""Package standalone, minimal Phase A benchmark harness for Raspberry Pi."""

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def package_pi_harness(root_dir: Path, target_dir: Path):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Packaging standalone Pi harness to: {target_dir}")

    # 1. Subdirectories
    (target_dir / "configs").mkdir(parents=True, exist_ok=True)
    (target_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (target_dir / "models").mkdir(parents=True, exist_ok=True)
    (target_dir / "weights" / "onnx").mkdir(parents=True, exist_ok=True)
    (target_dir / "assets").mkdir(parents=True, exist_ok=True)
    (target_dir / "benchmarks" / "pi").mkdir(parents=True, exist_ok=True)

    # 2. Copy configs
    shutil.copy(root_dir / "configs" / "benchmark.yaml", target_dir / "configs" / "benchmark.yaml")
    shutil.copy(root_dir / "configs" / "models.yaml", target_dir / "configs" / "models.yaml")

    # 3. Copy scripts
    shutil.copy(root_dir / "scripts" / "benchmark_pi.py", target_dir / "scripts" / "benchmark_pi.py")
    shutil.copy(root_dir / "scripts" / "collect_environment.py", target_dir / "scripts" / "collect_environment.py")

    # 4. Copy models module
    shutil.copy(root_dir / "models" / "__init__.py", target_dir / "models" / "__init__.py")
    shutil.copy(root_dir / "models" / "base_detector.py", target_dir / "models" / "base_detector.py")
    shutil.copy(root_dir / "models" / "onnx_detector.py", target_dir / "models" / "onnx_detector.py")

    # 5. Copy ONNX models & manifest
    for onnx_file in (root_dir / "weights" / "onnx").glob("*.onnx"):
        shutil.copy(onnx_file, target_dir / "weights" / "onnx" / onnx_file.name)
    shutil.copy(root_dir / "weights" / "manifest.json", target_dir / "weights" / "manifest.json")

    # 6. Copy test asset
    sample_img = root_dir / "assets" / "smoke_test.jpg"
    shutil.copy(sample_img, target_dir / "assets" / "sample.jpg")

    # 7. Create requirements-pi.txt (minimal, NO torch/ultralytics)
    req_pi_content = """# Minimal headless dependencies for Raspberry Pi ONNX Benchmark
numpy>=1.24.0
onnx>=1.14.0
onnxruntime>=1.15.0
opencv-python-headless>=4.7.0
PyYAML>=6.0
psutil>=5.9.0
"""
    with open(target_dir / "requirements-pi.txt", "w") as f:
        f.write(req_pi_content)
    with open(root_dir / "requirements-pi.txt", "w") as f:
        f.write(req_pi_content)

    # 8. Create standalone run script
    runner_script = """#!/bin/bash
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
$PYTHON_BIN scripts/benchmark_pi.py \\
    --models yolov8n yolo11n yolo12n yolo26n \\
    --images assets \\
    --onnx-dir weights/onnx \\
    --manifest weights/manifest.json \\
    --warmup 50 \\
    --runs 1000 \\
    --repeat 3 \\
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
"""
    run_sh = target_dir / "run_phase_a_pi.sh"
    with open(run_sh, "w") as f:
        f.write(runner_script)
    run_sh.chmod(0o755)

    # 9. Create deployment README
    readme_content = """# Standalone Raspberry Pi Phase A Harness

## Quick Deployment (via SCP / USB)
1. Copy this folder to Raspberry Pi:
   ```bash
   scp -r artifacts/pi_phase_a/ pi@<PI_IP>:~/pi_phase_a
   ```
2. On Raspberry Pi:
   ```bash
   cd ~/pi_phase_a
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-pi.txt
   bash run_phase_a_pi.sh
   ```
3. Retrieve results:
   ```bash
   scp pi@<PI_IP>:~/pi_phase_a/benchmarks/pi/phase_a_pi_summary.csv ./
   ```
"""
    with open(target_dir / "README.md", "w") as f:
        f.write(readme_content)

    # 10. Generate run manifest and compute checksums
    file_checksums = {}
    for p in sorted(target_dir.rglob("*")):
        if p.is_file():
            file_checksums[str(p.relative_to(target_dir))] = sha256_file(p)

    package_manifest = {
        "package_name": "pi_phase_a",
        "description": "Minimal standalone Pi benchmark package",
        "files_count": len(file_checksums),
        "checksums": file_checksums,
    }
    with open(target_dir / "run_manifest.json", "w") as f:
        json.dump(package_manifest, f, indent=2)

    # 11. Create tar.gz archive
    archive_path = target_dir.parent / "pi_phase_a.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(target_dir, arcname="pi_phase_a")

    tar_sha = sha256_file(archive_path)
    print(f"✓ Created {archive_path} (SHA-256: {tar_sha[:16]}...)")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    target = root / "artifacts" / "pi_phase_a"
    package_pi_harness(root, target)
