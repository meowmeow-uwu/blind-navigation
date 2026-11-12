#!/usr/bin/env python3
"""Benchmark ONNX models on offline video pipeline."""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.onnx_detector import ONNXDetector


def get_cpu_percent() -> float:
    """Get CPU usage percentage."""
    try:
        return psutil.cpu_percent(interval=0.01)
    except Exception:
        return 0.0


def get_process_ram_mb() -> float:
    """Get current process RAM in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def benchmark_video(
    model_name: str,
    onnx_path: Path,
    video_path: Path,
    config: dict,
) -> dict[str, Any]:
    """Benchmark model on video frames."""
    
    bench_cfg = config.get("benchmark", config)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "model": model_name,
            "status": "failed",
            "error": f"Failed to open video: {video_path}",
        }
    
    # Load detector
    try:
        detector = ONNXDetector(str(onnx_path))
    except Exception as e:
        cap.release()
        return {
            "model": model_name,
            "status": "failed",
            "error": f"Failed to load detector: {e}",
        }
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Process frames
    frame_times = []
    detections_per_frame = []
    peak_ram = 0
    processed_frames = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip if beyond limit
        if processed_frames >= bench_cfg.get("max_frames", 300):
            break
        
        try:
            # Create temp file for inference (OpenCV frame to bytes)
            temp_path = Path(f"/tmp/frame_{os.getpid()}.jpg")
            cv2.imwrite(str(temp_path), frame)
            
            start = time.perf_counter()
            detections = detector.predict(
                str(temp_path),
                conf=bench_cfg["confidence_threshold"],
                iou=bench_cfg["nms_iou_threshold"],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            frame_times.append(elapsed_ms)
            detections_per_frame.append(len(detections))
            peak_ram = max(peak_ram, get_process_ram_mb())
            
            temp_path.unlink()
            processed_frames += 1
            
        except Exception as e:
            continue
    
    cap.release()
    
    if not frame_times:
        return {
            "model": model_name,
            "status": "failed",
            "error": "No frames processed",
        }
    
    # Compute statistics
    frame_times = np.array(frame_times)
    
    result = {
        "model": model_name,
        "status": "passed",
        "video": str(video_path),
        "video_info": {
            "total_frames": total_frames,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
        },
        "processing": {
            "processed_frames": processed_frames,
            "mean_frame_ms": round(float(np.mean(frame_times)), 3),
            "median_frame_ms": round(float(np.median(frame_times)), 3),
            "p50_frame_ms": round(float(np.percentile(frame_times, 50)), 3),
            "p95_frame_ms": round(float(np.percentile(frame_times, 95)), 3),
            "p99_frame_ms": round(float(np.percentile(frame_times, 99)), 3),
            "min_frame_ms": round(float(np.min(frame_times)), 3),
            "max_frame_ms": round(float(np.max(frame_times)), 3),
        },
        "detections": {
            "mean_per_frame": round(float(np.mean(detections_per_frame)), 2),
            "max_per_frame": int(np.max(detections_per_frame)),
            "min_per_frame": int(np.min(detections_per_frame)),
        },
        "resources": {
            "peak_ram_mb": round(peak_ram, 1),
        },
        "pipeline_fps": round(processed_frames * fps / total_frames if total_frames > 0 else 0, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models on video")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("weights/onnx"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/raw"))
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Check video
    if not args.video.exists():
        print(f"Video not found: {args.video}")
        
        # Create dummy video from image if available
        test_img = Path("assets/export_validation/test_001.jpg")
        if test_img.exists():
            print(f"Creating test video from {test_img}...")
            img = cv2.imread(str(test_img))
            h, w = img.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(args.video), fourcc, 30, (w, h))
            
            # Write 300 frames (10 sec @ 30 fps)
            for _ in range(300):
                out.write(img)
            out.release()
            
            print(f"✓ Created test video: {args.video}")
        else:
            sys.exit(1)
    
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
        
        print(f"Benchmarking {model_name} on video...")
        result = benchmark_video(model_name, onnx_path, args.video, config)
        all_results[model_name] = result
        
        if result["status"] == "passed":
            proc = result["processing"]
            print(f"  ✓ {result['processing']['processed_frames']} frames, "
                  f"p50={proc['p50_frame_ms']}ms, "
                  f"pipeline_fps={result['pipeline_fps']}")
        else:
            print(f"  ✗ {result['error']}")
    
    # Write summary JSON
    summary_path = args.output / "benchmark_video_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Video benchmark results: {summary_path}")
    
    # Print table
    print(f"\n{'='*120}")
    print(f"{'Model':<15} {'Frames':<10} {'Mean':<10} {'P50':<10} {'P95':<10} {'Pipeline FPS':<15}")
    print(f"{'='*120}")
    for name, result in sorted(all_results.items()):
        if result["status"] == "passed":
            proc = result["processing"]
            print(f"{name:<15} "
                  f"{result['processing']['processed_frames']:<10} "
                  f"{proc['mean_frame_ms']:<10.2f} "
                  f"{proc['p50_frame_ms']:<10.2f} "
                  f"{proc['p95_frame_ms']:<10.2f} "
                  f"{result['pipeline_fps']:<15.2f}")


if __name__ == "__main__":
    main()
