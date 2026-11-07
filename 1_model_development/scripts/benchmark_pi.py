#!/usr/bin/env python3
"""Benchmark script for Raspberry Pi.

This script is designed to run on Pi with minimal dependencies.
Temperature and throttle detection are best-effort and may fail gracefully.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.onnx_detector import ONNXDetector


def get_pi_temperature():
    """Get Pi CPU temperature."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp_mc = int(f.read().strip())
            return round(temp_mc / 1000.0, 2)
    except Exception:
        return None


def get_pi_throttled_state():
    """Get Pi throttle state."""
    try:
        with open("/proc/device-tree/throttled", "rb") as f:
            return f.read().decode().strip()
    except Exception:
        return None


def get_pi_cpu_frequency():
    """Get Pi CPU frequency."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    return round(float(line.split(":")[1].strip()), 2)
    except Exception:
        return None


def benchmark_model_pi(
    model_name: str,
    onnx_path: Path,
    image_path: Path,
    warmup_runs: int,
    measured_runs: int,
) -> dict:
    """Benchmark on Pi."""
    
    result = {
        "model": model_name,
        "status": "failed",
        "error": None,
        "measured_runs": 0,
        "latency_ms": [],
        "temperature_c": None,
        "throttled": None,
        "cpu_frequency_mhz": None,
    }
    
    try:
        # Load model
        detector = ONNXDetector(str(onnx_path))
        
        # Warmup
        for _ in range(warmup_runs):
            try:
                detector.predict(str(image_path), conf=0.25, iou=0.45)
            except Exception:
                pass
        
        # Measure
        latencies = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            try:
                detector.predict(str(image_path), conf=0.25, iou=0.45)
            except Exception:
                continue
            latencies.append((time.perf_counter() - start) * 1000)
        
        if latencies:
            result["status"] = "passed"
            result["measured_runs"] = len(latencies)
            result["latency_ms"] = [round(x, 3) for x in latencies]
            
            # Get Pi-specific metrics
            result["temperature_c"] = get_pi_temperature()
            result["throttled"] = get_pi_throttled_state()
            result["cpu_frequency_mhz"] = get_pi_cpu_frequency()
        else:
            result["error"] = "No successful runs"
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models on Raspberry Pi")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, default=Path("weights/onnx"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/pi"))
    
    args = parser.parse_args()
    
    # Get test images
    if not args.images.exists():
        print(f"Image directory not found: {args.images}")
        sys.exit(1)
    
    image_files = sorted(list(args.images.glob("*.jpg")) + list(args.images.glob("*.png")))
    if not image_files:
        print(f"No images found in {args.images}")
        sys.exit(1)
    
    test_image = image_files[0]
    
    # Create output dir
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each model
    all_results = {}
    
    for model_name in args.models:
        onnx_filename = f"{model_name}_320_fp32.onnx"
        onnx_path = args.onnx_dir / onnx_filename
        
        if not onnx_path.exists():
            print(f"⚠ ONNX not found: {onnx_path}")
            continue
        
        print(f"Benchmarking {model_name}...")
        
        # Run repeats
        repeat_results = []
        for rep in range(args.repeat):
            result = benchmark_model_pi(
                model_name,
                onnx_path,
                test_image,
                args.warmup,
                args.runs,
            )
            repeat_results.append(result)
            
            if result["status"] == "passed":
                if np:
                    latencies = result["latency_ms"]
                    p50 = round(float(np.percentile(latencies, 50)), 3)
                    print(f"  Repeat {rep + 1}: {len(latencies)} runs, p50={p50}ms, "
                          f"temp={result['temperature_c']}°C")
                else:
                    print(f"  Repeat {rep + 1}: {len(result['latency_ms'])} runs")
            else:
                print(f"  Repeat {rep + 1}: {result['error']}")
        
        all_results[model_name] = repeat_results
    
    # Write results JSON
    output_path = args.output / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
