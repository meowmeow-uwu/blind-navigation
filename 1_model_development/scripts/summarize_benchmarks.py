#!/usr/bin/env python3
"""Summarize benchmark results and generate reports."""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    """Load CSV file."""
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    raw_dir = Path("benchmarks/raw")
    summary_dir = Path("benchmarks/summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model-only benchmark
    model_summary_path = summary_dir / "benchmark_model_onnx_fp32.json"
    if not model_summary_path.exists():
        print(f"⚠ Model benchmark not found: {model_summary_path}")
        model_results = {}
    else:
        model_results = load_json(model_summary_path)
    
    # Load video benchmark
    video_summary_path = raw_dir / "benchmark_video_summary.json"
    if not video_summary_path.exists():
        print(f"⚠ Video benchmark not found: {video_summary_path}")
        video_results = {}
    else:
        video_results = load_json(video_summary_path)
    
    # Create unified CSV
    rows = []
    
    for model_name, model_data in model_results.items():
        video_data = video_results.get(model_name, {})
        
        row = {
            "model": model_name,
            "backend": "onnxruntime",
            "precision": "fp32",
            "input_size": model_data.get("input_size", 320),
            "model_only_p50_ms": model_data.get("latency", {}).get("total_p50_ms", "N/A"),
            "model_only_p95_ms": model_data.get("latency", {}).get("total_p95_ms", "N/A"),
            "video_p50_ms": video_data.get("processing", {}).get("p50_frame_ms", "N/A"),
            "video_p95_ms": video_data.get("processing", {}).get("p95_frame_ms", "N/A"),
            "pipeline_fps": video_data.get("pipeline_fps", "N/A"),
            "peak_ram_mb": model_data.get("peak_ram_mb", "N/A"),
            "status": "validated",
        }
        rows.append(row)
    
    # Write summary CSV
    if rows:
        csv_path = summary_dir / "phase_a_host_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ Summary CSV: {csv_path}")
    
    # Write JSON summary
    summary_json = {
        "generated_at": None,
        "model_benchmarks": model_results,
        "video_benchmarks": video_results,
        "unified_summary": rows,
    }
    
    from datetime import datetime
    summary_json["generated_at"] = datetime.utcnow().isoformat() + "Z"
    
    json_path = summary_dir / "phase_a_host_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"✓ Summary JSON: {json_path}")
    
    # Print table
    print(f"\n{'='*140}")
    print(f"{'Model':<15} {'Model p50':<12} {'Video p50':<12} {'Video p95':<12} {'Pipeline FPS':<15} {'Peak RAM':<12}")
    print(f"{'='*140}")
    for row in sorted(rows, key=lambda x: x["model"]):
        print(f"{row['model']:<15} "
              f"{str(row['model_only_p50_ms']):<12} "
              f"{str(row['video_p50_ms']):<12} "
              f"{str(row['video_p95_ms']):<12} "
              f"{str(row['pipeline_fps']):<15} "
              f"{str(row['peak_ram_mb']):<12}")


if __name__ == "__main__":
    main()
