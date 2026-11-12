#!/usr/bin/env python3
"""Benchmark ONNX models - model-only, no video."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.onnx_detector import ONNXDetector


def get_git_commit():
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return None


def get_onnx_sha256(onnx_path: Path) -> str:
    """Compute SHA256 of ONNX file."""
    h = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_process_ram_mb() -> float:
    """Get current process RAM in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_cpu_percent() -> float:
    """Get CPU usage percentage."""
    try:
        return psutil.cpu_percent(interval=0.01)
    except Exception:
        return 0.0


def benchmark_model(
    model_name: str,
    onnx_path: Path,
    image_path: Path,
    config: dict,
    run_num: int,
) -> list[dict[str, Any]]:
    """Benchmark a single ONNX model on one image."""
    results = []
    
    try:
        # Load model
        detector = ONNXDetector(str(onnx_path))
        
        # Read image
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        

        
        # Get benchmark config
        bench_cfg = config.get("benchmark", config)
        
        # Warmup
        for _ in range(bench_cfg["warmup_runs"]):
            try:
                detector.predict(
                    str(image_path),
                    conf=bench_cfg["confidence_threshold"],
                    iou=bench_cfg["nms_iou_threshold"],
                )
            except Exception:
                pass
        
        # Measured runs
        measured_runs = bench_cfg["measured_runs"]
        
        for iteration in range(measured_runs):
            start = time.perf_counter()
            
            try:
                detections = detector.predict(
                    str(image_path),
                    conf=bench_cfg["confidence_threshold"],
                    iou=bench_cfg["nms_iou_threshold"],
                )
            except Exception as e:
                continue
            
            total_ms = (time.perf_counter() - start) * 1000
            
            # Get system stats
            ram_mb = get_process_ram_mb()
            cpu_pct = get_cpu_percent()
            num_dets = len(detections)
            
            results.append({
                "model": model_name,
                "run": run_num,
                "iteration": iteration,
                "preprocess_ms": 0.0,  # Approximate - not separated
                "inference_ms": 0.0,   # Approximate - not separated
                "postprocess_ms": 0.0, # Approximate - not separated
                "total_ms": round(total_ms, 3),
                "cpu_percent": round(cpu_pct, 1),
                "ram_mb": round(ram_mb, 1),
                "detections": num_dets,
            })
    
    except Exception as e:
        print(f"Error benchmarking {model_name}: {e}")
    
    return results


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute percentile and mean statistics."""
    values = sorted(values)
    n = len(values)
    
    return {
        "mean": round(np.mean(values), 3),
        "median": round(np.median(values), 3),
        "p50": round(np.percentile(values, 50), 3),
        "p95": round(np.percentile(values, 95), 3),
        "p99": round(np.percentile(values, 99), 3),
        "min": round(np.min(values), 3),
        "max": round(np.max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--images", type=Path, default=Path("assets/export_validation"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("weights/onnx"))
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get test images
    if not args.images.exists():
        print(f"Image directory not found: {args.images}")
        sys.exit(1)
    
    image_files = sorted(list(args.images.glob("*.jpg")) + list(args.images.glob("*.png")))
    if not image_files:
        print(f"No images found in {args.images}")
        sys.exit(1)
    
    # Use first image for consistency
    test_image = image_files[0]
    print(f"Benchmarking with: {test_image}")
    
    # Create output dirs
    raw_dir = Path(config["output"]["raw_dir"])
    summary_dir = Path(config["output"]["summary_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each model
    all_results = {}
    git_commit = get_git_commit()
    
    for model_name in args.models:
        onnx_filename = f"{model_name}_320_fp32.onnx"
        onnx_path = args.onnx_dir / onnx_filename
        
        if not onnx_path.exists():
            print(f"⚠ ONNX not found: {onnx_path}")
            continue
        
        print(f"\nBenchmarking {model_name}...")
        
        onnx_sha = get_onnx_sha256(onnx_path)
        
        # Run repeats
        all_model_results = []
        for rep in range(config["benchmark"]["repeat"]):
            print(f"  Repeat {rep + 1}/{config['benchmark']['repeat']}...")
            results = benchmark_model(
                model_name,
                onnx_path,
                test_image,
                config,
                rep + 1,
            )
            all_model_results.extend(results)
        
        if not all_model_results:
            print(f"✗ No results for {model_name}")
            continue
        
        # Write raw CSV
        csv_path = raw_dir / f"{model_name}_onnx_fp32_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_model_results[0].keys())
            writer.writeheader()
            writer.writerows(all_model_results)
        
        print(f"  Raw results: {csv_path}")
        
        # Compute statistics
        total_ms_values = [r["total_ms"] for r in all_model_results]
        latency_stats = compute_stats(total_ms_values)
        
        # Summary
        summary = {
            "model": model_name,
            "runtime": "onnxruntime",
            "precision": "fp32",
            "input_size": config["benchmark"]["input_size"],
            "runs": config["benchmark"]["measured_runs"],
            "repeats": config["benchmark"]["repeat"],
            "latency": {
                "total_mean_ms": latency_stats["mean"],
                "total_median_ms": latency_stats["median"],
                "total_p50_ms": latency_stats["p50"],
                "total_p95_ms": latency_stats["p95"],
                "total_p99_ms": latency_stats["p99"],
                "total_min_ms": latency_stats["min"],
                "total_max_ms": latency_stats["max"],
            },
            "peak_ram_mb": round(max(r["ram_mb"] for r in all_model_results), 1),
            "onnx_sha256": onnx_sha,
            "git_commit": git_commit,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        all_results[model_name] = summary
        
        print(f"  ✓ p50={summary['latency']['total_p50_ms']}ms, "
              f"p95={summary['latency']['total_p95_ms']}ms, "
              f"p99={summary['latency']['total_p99_ms']}ms")
    
    # Write summary JSON
    summary_path = summary_dir / "benchmark_model_onnx_fp32.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Summary: {summary_path}")
    
    # Print table
    print(f"\n{'='*100}")
    print(f"{'Model':<15} {'Mean':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'Peak RAM':<12}")
    print(f"{'='*100}")
    for name, summary in sorted(all_results.items()):
        lat = summary["latency"]
        print(f"{name:<15} "
              f"{lat['total_mean_ms']:<10.2f} "
              f"{lat['total_p50_ms']:<10.2f} "
              f"{lat['total_p95_ms']:<10.2f} "
              f"{lat['total_p99_ms']:<10.2f} "
              f"{summary['peak_ram_mb']:<12.1f}")


if __name__ == "__main__":
    main()
