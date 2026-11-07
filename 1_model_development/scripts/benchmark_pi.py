#!/usr/bin/env python3
"""Hardened Raspberry Pi and edge benchmark harness for Phase A.

Performs thorough preflight checks, system metrics logging (throttling, temperature,
frequency, undervoltage), invalid-run rules enforcement, and raw metric preservation.
Gracefully runs on non-Pi development hosts without failure.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import onnx
import onnx.checker
import onnxruntime as ort
import psutil

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.onnx_detector import ONNXDetector


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return res.stdout.strip()[:8]
    except Exception:
        return None


def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().lower()
            return "raspberry pi" in model
    except Exception:
        return False


def get_pi_throttling_and_undervoltage():
    """Query vcgencmd get_throttled on Raspberry Pi."""
    if not is_raspberry_pi():
        return None, None

    try:
        res = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            val_str = res.stdout.strip().split("=")[-1]
            code = int(val_str, 16)
            undervoltage = bool(code & 0x1 or code & 0x10000)
            throttled = bool(code & 0x2 or code & 0x20000)
            return code, (undervoltage or throttled)
    except Exception:
        pass
    return None, None


def get_system_temperature():
    """Read thermal zone temperature if on Linux / Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp_mc = int(f.read().strip())
            return round(temp_mc / 1000.0, 2)
    except Exception:
        return None


def get_cpu_governor():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except Exception:
        return "unknown"


def run_preflight(
    onnx_path: Path,
    expected_size: int,
    manifest_path: Path | None,
    num_threads: int,
) -> dict:
    preflight = {
        "status": "passed",
        "errors": [],
        "warnings": [],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "is_64bit": sys.maxsize > 2**32,
        "is_raspberry_pi": is_raspberry_pi(),
        "cpu_governor": get_cpu_governor(),
        "threads_configured": num_threads,
        "initial_temperature_c": get_system_temperature(),
        "free_disk_gb": round(shutil.disk_usage(onnx_path.parent).free / (1024**3), 2),
    }

    if not preflight["is_64bit"]:
        preflight["warnings"].append("System is not 64-bit; potential memory and SIMD limits")

    # 1. Model file existence
    if not onnx_path.exists():
        preflight["errors"].append(f"Model file not found: {onnx_path}")
        preflight["status"] = "failed"
        return preflight

    # 2. Check model SHA256 against manifest
    model_sha = sha256_file(onnx_path)
    preflight["model_sha256"] = model_sha

    if manifest_path and manifest_path.exists():
        try:
            with open(manifest_path) as f:
                man = json.load(f)
            model_stem = onnx_path.stem.replace(f"_{expected_size}_fp32", "")
            if model_stem in man["models"]:
                exp_sha = man["models"][model_stem].get(f"onnx_fp32_{expected_size}", {}).get("sha256")
                if exp_sha and exp_sha != model_sha:
                    preflight["errors"].append(f"SHA-256 mismatch: expected {exp_sha[:12]}, got {model_sha[:12]}")
                    preflight["status"] = "failed"
        except Exception as e:
            preflight["warnings"].append(f"Could not verify manifest SHA: {e}")

    # 3. Graph verification
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        inp = model.graph.input[0]
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        if shape != [1, 3, expected_size, expected_size]:
            preflight["errors"].append(f"Unexpected input shape: expected [1,3,{expected_size},{expected_size}], got {shape}")
            preflight["status"] = "failed"
    except Exception as e:
        preflight["errors"].append(f"ONNX check failed: {e}")
        preflight["status"] = "failed"

    # 4. Check available provider
    available_providers = ort.get_available_providers()
    preflight["available_providers"] = available_providers
    if "CPUExecutionProvider" not in available_providers:
        preflight["errors"].append("CPUExecutionProvider not available in ONNX Runtime")
        preflight["status"] = "failed"

    # 5. Check throttling
    code, throttled = get_pi_throttling_and_undervoltage()
    if throttled:
        preflight["warnings"].append(f"Active throttling/undervoltage detected prior to run (code={hex(code)})")

    return preflight


def run_benchmark_for_model(
    model_name: str,
    onnx_path: Path,
    image_path: Path,
    warmup_runs: int,
    measured_runs: int,
    repeat: int,
    num_threads: int,
    input_size: int,
    manifest_path: Path | None,
    output_dir: Path,
) -> dict:
    preflight = run_preflight(onnx_path, input_size, manifest_path, num_threads)
    if preflight["status"] == "failed":
        return {
            "model": model_name,
            "status": "invalid",
            "preflight": preflight,
            "error": "Preflight checks failed",
            "repeats": [],
        }

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1

    detector = ONNXDetector(str(onnx_path))
    git_commit = get_git_commit()

    repeat_summaries = []
    repeat_p50s = []

    for rep in range(1, repeat + 1):
        raw_samples = []

        # Warmup
        for _ in range(warmup_runs):
            try:
                detector.predict(str(image_path), conf=0.25, iou=0.45)
            except Exception:
                pass

        latencies = []
        cpu_utils = []
        mem_rss = []

        start_time = time.time()
        start_temp = get_system_temperature()
        throttle_code_start, _ = get_pi_throttling_and_undervoltage()

        for iter_idx in range(measured_runs):
            t0 = time.perf_counter()
            try:
                dets = detector.predict(str(image_path), conf=0.25, iou=0.45)
            except Exception:
                continue
            lat = (time.perf_counter() - t0) * 1000.0

            # Guard against NaN / Inf
            if np and (np.isnan(lat) or np.isinf(lat)):
                continue

            latencies.append(lat)

            if iter_idx % max(1, (measured_runs // 10)) == 0:
                cpu_utils.append(psutil.cpu_percent(interval=None))
                mem_rss.append(psutil.Process().memory_info().rss / (1024 * 1024))

        end_temp = get_system_temperature()
        throttle_code_end, throttled_flag = get_pi_throttling_and_undervoltage()

        # Check invalid run criteria
        is_invalid = False
        reasons = []

        if len(latencies) < measured_runs:
            is_invalid = True
            reasons.append(f"Incomplete measured runs: {len(latencies)}/{measured_runs}")

        if throttled_flag:
            is_invalid = True
            reasons.append(f"Thermal or undervoltage throttling triggered (code={throttle_code_end})")

        if end_temp and end_temp > 82.0:
            is_invalid = True
            reasons.append(f"Temperature exceeded 82C safe cutoff ({end_temp}C)")

        # Save raw CSV
        raw_csv_path = output_dir / f"{model_name}_pi_rep{rep}_raw.csv"
        with open(raw_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "latency_ms"])
            for idx, l in enumerate(latencies):
                w.writerow([idx, round(l, 3)])

        p50 = float(np.percentile(latencies, 50)) if (np and latencies) else 0.0
        p95 = float(np.percentile(latencies, 95)) if (np and latencies) else 0.0
        p99 = float(np.percentile(latencies, 99)) if (np and latencies) else 0.0
        mean_lat = float(np.mean(latencies)) if (np and latencies) else 0.0

        repeat_p50s.append(p50)

        repeat_summaries.append({
            "repeat": rep,
            "status": "invalid" if is_invalid else "valid",
            "invalid_reasons": reasons,
            "measured_runs": len(latencies),
            "latency": {
                "mean_ms": round(mean_lat, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
            },
            "environment_sample": {
                "cpu_util_avg": round(float(np.mean(cpu_utils)), 1) if (np and cpu_utils) else None,
                "ram_peak_mb": round(max(mem_rss), 1) if mem_rss else None,
                "start_temperature_c": start_temp,
                "end_temperature_c": end_temp,
                "throttling_code": throttle_code_end,
            },
            "raw_csv": str(raw_csv_path.name),
        })

    # Check stability between repeats
    stability_warning = None
    if len(repeat_p50s) >= 2:
        max_diff = max(repeat_p50s) - min(repeat_p50s)
        base = min(repeat_p50s)
        if base > 0 and (max_diff / base) > 0.10:
            stability_warning = f"p50 deviation across repeats > 10% ({round((max_diff / base)*100, 1)}%)"

    overall_status = "valid" if any(r["status"] == "valid" for r in repeat_summaries) else "invalid"

    return {
        "model": model_name,
        "status": overall_status,
        "stability_warning": stability_warning,
        "preflight": preflight,
        "git_commit": git_commit,
        "repeats": repeat_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Hardened Raspberry Pi ONNX Benchmark Harness")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n", "yolo12n", "yolo26n"])
    parser.add_argument("--images", type=Path, default=Path("assets/export_validation"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("weights/onnx"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/pi"))
    parser.add_argument("--manifest", type=Path, default=Path("weights/manifest.json"))
    parser.add_argument("--no-render", action="store_true", default=True)

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Find test image
    images = sorted(list(args.images.glob("*.jpg")) + list(args.images.glob("*.png")))
    if not images:
        print(f"Error: No images found in {args.images}")
        sys.exit(1)
    test_image = images[0]

    all_results = {}

    print(f"Starting Pi Benchmark Harness (Host dry-run compatible)")
    print(f"Platform is Raspberry Pi: {is_raspberry_pi()}")
    print(f"Threads: {args.threads}, Warmup: {args.warmup}, Runs: {args.runs}, Repeat: {args.repeat}\n")

    for model_name in args.models:
        onnx_file = args.onnx_dir / f"{model_name}_{args.input_size}_fp32.onnx"
        print(f"Running model: {model_name}...")
        res = run_benchmark_for_model(
            model_name,
            onnx_file,
            test_image,
            args.warmup,
            args.runs,
            args.repeat,
            args.threads,
            args.input_size,
            args.manifest,
            args.output,
        )
        all_results[model_name] = res

        print(f"  Status: {res['status']}")
        for rep in res["repeats"]:
            lat = rep["latency"]
            print(f"  Rep {rep['repeat']}: p50={lat['p50_ms']}ms, p95={lat['p95_ms']}ms [{rep['status']}]")
        if res.get("stability_warning"):
            print(f"  ⚠ {res['stability_warning']}")

    summary_file = args.output / "benchmark_results.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Completed benchmark run. Results saved to: {summary_file}")


if __name__ == "__main__":
    main()
