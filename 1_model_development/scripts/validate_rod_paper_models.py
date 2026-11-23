#!/usr/bin/env python3
"""Validate fine-tuned ROD models (YOLOv8n, YOLOv9t, YOLOv12n, YOLO26n) on ROD test set.

Compares measured performance directly against the paper / benchmark results
reported in `rod-paper/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # blind-navigation/1_model_development
REPO_ROOT = PROJECT_ROOT.parent.parent  # pbl6-blind-navigation
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Models configuration matching folder names and checkpoints in rod-paper/
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "yolov8n": {
        "folder": "yolov8n-rod",
        "checkpoint": "yolov8n_best.pt",
        "paper_results": "yolov8n_results.txt",
        "paper_summary": "yolov8n_summary.json",
    },
    "yolov9t": {
        "folder": "yolov9t",
        "checkpoint": "yolov9t_best.pt",
        "paper_results": "yolov9t_results.txt",
        "paper_summary": "yolov9t_summary.json",
    },
    "yolov12n": {
        "folder": "yolov12n",
        "checkpoint": "yolov12n_best.pt",
        "paper_results": "yolov12n_results.txt",
        "paper_summary": "yolov12n_summary.json",
    },
    "yolov26n": {
        "folder": "yolov26n",
        "checkpoint": "yolo26n_best.pt",
        "paper_results": "yolo26n_results.txt",
        "paper_summary": "yolo26n_summary.json",
    },
}

ROD_NAMES = [
    "Bike", "Building", "Car", "Person", "Stairs",
    "Traffic sign", "Electrical Pole", "Road", "Motorcycle", "Dustbin",
    "Dog", "Manhole", "Tree", "Guard rail", "Pedestrian crosswalk",
    "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant",
    "Teraffic Barrel", "Plant Pot", "Electrical Box", "Chair", "Bicycle Rack",
]


def create_eval_yaml(test_dir: Path, output_yaml_path: Path) -> Path:
    """Create a temporary data.yaml pointing to the local test set directory."""
    images_dir = test_dir if test_dir.name == "images" else test_dir / "images"
    if not images_dir.is_dir():
        # Maybe test_dir is the dataset root
        if (test_dir / "test" / "images").is_dir():
            images_dir = test_dir / "test" / "images"
        elif (test_dir / "test").is_dir():
            images_dir = test_dir / "test"
        else:
            raise FileNotFoundError(f"Could not find test images directory at: {test_dir}")

    # Ultralytics requires 'train' and 'val' keys even when only evaluating.
    # Pointing both 'train', 'val', and 'test' to the test images directory satisfies validation.
    content = {
        "path": str(images_dir.parent.resolve()),
        "train": str(images_dir.resolve()),
        "val": str(images_dir.resolve()),
        "test": str(images_dir.resolve()),
        "nc": len(ROD_NAMES),
        "names": ROD_NAMES,
    }

    import yaml
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)

    return output_yaml_path


def load_paper_metrics(paper_dir: Path, model_key: str) -> dict[str, Any]:
    """Load baseline results from paper's summary.json or results.txt."""
    cfg = MODEL_CONFIGS[model_key]
    model_folder = paper_dir / cfg["folder"]
    summary_path = model_folder / cfg["paper_summary"]
    results_path = model_folder / cfg["paper_results"]

    metrics: dict[str, Any] = {}

    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                metrics = {
                    "mAP50": data.get("mAP50"),
                    "mAP50_95": data.get("mAP50_95"),
                    "precision": data.get("precision"),
                    "recall": data.get("recall"),
                    "params": data.get("params"),
                    "avg_ms_per_img": data.get("avg_ms_per_img"),
                    "fps": data.get("fps"),
                    "test_images": data.get("test_images"),
                }
                return metrics
        except Exception as e:
            print(f"Warning: Failed reading {summary_path}: {e}")

    if results_path.is_file():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        k, v = k.strip(), v.strip()
                        try:
                            val = float(v) if "." in v or "e" in v.lower() else int(v)
                        except ValueError:
                            val = v
                        metrics[k] = val
        except Exception as e:
            print(f"Warning: Failed reading {results_path}: {e}")

    return metrics


def validate_model(
    model_name: str,
    weights_path: Path,
    data_yaml_path: Path,
    imgsz: int = 640,
    batch: int = 32,
    device: str = "cuda",
    project_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Ultralytics validation on the model weights."""
    import torch

    # Check device availability and resolve safe device
    chosen_device = device.strip() if device else ""
    if not chosen_device or chosen_device.lower() in ("cuda", "0", "cuda:0") or chosen_device.isdigit():
        if not torch.cuda.is_available():
            if chosen_device and chosen_device.lower() != "cpu":
                print(f"[{model_name}] [NOTICE] CUDA device '{chosen_device}' requested but PyTorch CUDA is not available on this system.")
                print(f"[{model_name}] [NOTICE] Automatically falling back to 'cpu'...")
            chosen_device = "cpu"
        elif not chosen_device:
            chosen_device = "0"

    from ultralytics import YOLO

    print(f"\n[{model_name}] Loading weights: {weights_path} (target device: {chosen_device})")
    model = YOLO(str(weights_path))
    if hasattr(model, "to") and chosen_device:
        model.to(chosen_device)

    save_dir = project_dir or (PROJECT_ROOT / "runs" / "rod_validation")
    save_dir.mkdir(parents=True, exist_ok=True)

    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml_path),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "device": chosen_device,
        "plots": False,
        "save_json": False,
        "project": str(save_dir),
        "name": model_name,
        "exist_ok": True,
        "verbose": True,
    }

    start_time = time.time()
    metrics = model.val(**val_kwargs)
    eval_duration = time.time() - start_time

    # Extract metrics from Ultralytics results object
    # metrics.box.map50, metrics.box.map (mAP50-95), metrics.box.mp (mean precision), metrics.box.mr (mean recall)
    box = metrics.box
    speed = metrics.speed  # dict: {'preprocess': ms, 'inference': ms, 'loss': ms, 'postprocess': ms}
    inference_ms = speed.get("inference", 0.0)
    total_ms = sum(speed.values())
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    # Calculate parameter count
    param_count = sum(p.numel() for p in model.model.parameters())

    return {
        "model": model_name,
        "weights": str(weights_path),
        "params": param_count,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "inference_ms": round(inference_ms, 2),
        "total_ms": round(total_ms, 2),
        "fps": round(fps, 2),
        "eval_duration_sec": round(eval_duration, 2),
    }


def format_diff(measured: float | None, paper: float | None, is_pct: bool = False) -> str:
    """Format difference between measured metric and paper metric."""
    if measured is None or paper is None:
        return "N/A"
    diff = measured - paper
    sign = "+" if diff >= 0 else ""
    if is_pct:
        return f"{sign}{diff*100:+.2f}%"
    return f"{sign}{diff:+.4f}"


def print_comparison_table(results: list[dict[str, Any]]):
    """Print an ASCII comparison table matching measured results against paper baseline."""
    header = (
        f"{'Model':<10} | {'Metric':<10} | {'Measured':<12} | {'Paper':<12} | {'Diff':<10}\n"
        + "-" * 62
    )
    print("\n" + "=" * 62)
    print("COMPARISON: MEASURED ON TEST SET vs. PAPER BASELINE")
    print("=" * 62)
    print(header)

    for item in results:
        m = item["model"]
        meas = item.get("measured", {})
        paper = item.get("paper", {})

        metrics_to_show = [
            ("mAP@0.50", meas.get("mAP50"), paper.get("mAP50"), False),
            ("mAP@50-95", meas.get("mAP50_95"), paper.get("mAP50_95"), False),
            ("Precision", meas.get("precision"), paper.get("precision"), False),
            ("Recall", meas.get("recall"), paper.get("recall"), False),
            ("Params", meas.get("params"), paper.get("params"), False),
            ("FPS", meas.get("fps"), paper.get("fps"), False),
        ]

        for idx, (label, val_m, val_p, is_pct) in enumerate(metrics_to_show):
            prefix = m if idx == 0 else ""
            str_m = f"{val_m:.4f}" if isinstance(val_m, float) else str(val_m if val_m is not None else "N/A")
            str_p = f"{val_p:.4f}" if isinstance(val_p, float) else str(val_p if val_p is not None else "N/A")
            diff_str = (
                format_diff(val_m, val_p, is_pct=is_pct)
                if isinstance(val_m, (int, float)) and isinstance(val_p, (int, float))
                else "-"
            )
            print(f"{prefix:<10} | {label:<10} | {str_m:<12} | {str_p:<12} | {diff_str:<10}")
        print("-" * 62)


def main():
    parser = argparse.ArgumentParser(
        description="Validate fine-tuned ROD models on test set and compare with paper."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "yolov8n", "yolov9t", "yolov12n", "yolov26n"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )
    # Default test directory: prefer real dataset if available, fallback to processed
    real_raw_test = PROJECT_ROOT / "datasets" / "raw" / "ROD-Dataset" / "dataset" / "test"
    fallback_test = PROJECT_ROOT / "datasets" / "processed" / "rod_v1" / "test"
    default_test_dir = real_raw_test if real_raw_test.is_dir() else fallback_test

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=default_test_dir,
        help=f"Path to ROD test set directory containing images/ and labels/ (default: {default_test_dir})",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=REPO_ROOT / "rod-paper",
        help="Path to rod-paper directory containing model weights and paper results (default: rod-paper)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "validation_rod_paper",
        help="Directory to save comparison reports and JSON outputs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for validation (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for validation (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device, e.g. '0', 'cpu', 'cuda' (default: auto)",
    )

    args = parser.parse_args()

    # Determine models to run
    models_to_run = (
        ["yolov8n", "yolov9t", "yolov12n", "yolov26n"]
        if "all" in args.models
        else args.models
    )

    paper_dir = args.paper_dir.resolve()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("ROD MODELS TEST SET VALIDATION & PAPER BENCHMARK COMPARISON")
    print("=" * 62)
    print(f"Paper folder: {paper_dir}")
    print(f"Test data:    {test_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Models:       {', '.join(models_to_run)}")
    print(f"Image size:   {args.imgsz} | Batch: {args.batch} | Device: {args.device or 'auto'}")
    print("=" * 62)

    # 1. Create a validation yaml pointing to test set
    temp_yaml_path = output_dir / "rod_test_val.yaml"
    create_eval_yaml(test_dir, temp_yaml_path)
    print(f"Generated validation config: {temp_yaml_path}")

    all_comparisons = []

    for model_key in models_to_run:
        cfg = MODEL_CONFIGS[model_key]
        weights_file = paper_dir / cfg["folder"] / cfg["checkpoint"]

        if not weights_file.is_file():
            print(f"\n[SKIP] Checkpoint not found: {weights_file}")
            continue

        # Load paper results for comparison
        paper_metrics = load_paper_metrics(paper_dir, model_key)

        # Run validation
        try:
            measured_metrics = validate_model(
                model_name=model_key,
                weights_path=weights_file,
                data_yaml_path=temp_yaml_path,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project_dir=output_dir / "val_runs",
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to validate {model_key}: {e}")
            measured_metrics = {"error": str(e)}

        comparison_entry = {
            "model": model_key,
            "weights_path": str(weights_file),
            "measured": measured_metrics,
            "paper": paper_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        all_comparisons.append(comparison_entry)

    # Print summary table
    print_comparison_table(all_comparisons)

    # Save results to JSON
    json_path = output_dir / "rod_paper_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nSaved detailed comparison JSON to: {json_path}")

    # Save results to Markdown report
    md_path = output_dir / "rod_paper_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ROD Models: Test Set Validation vs. Paper Baseline\n\n")
        f.write(f"- **Evaluated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Test Dataset**: `{test_dir}`\n")
        f.write(f"- **Image Size**: {args.imgsz}\n\n")
        f.write("| Model | Metric | Measured (Local) | Paper / Checkpoint | Difference |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for item in all_comparisons:
            m = item["model"]
            meas = item.get("measured", {})
            paper = item.get("paper", {})
            metrics_to_show = [
                ("mAP@0.50", meas.get("mAP50"), paper.get("mAP50")),
                ("mAP@50-95", meas.get("mAP50_95"), paper.get("mAP50_95")),
                ("Precision", meas.get("precision"), paper.get("precision")),
                ("Recall", meas.get("recall"), paper.get("recall")),
                ("Params", meas.get("params"), paper.get("params")),
                ("FPS", meas.get("fps"), paper.get("fps")),
            ]
            for idx, (label, val_m, val_p) in enumerate(metrics_to_show):
                prefix = f"**{m}**" if idx == 0 else ""
                str_m = f"{val_m:.4f}" if isinstance(val_m, float) else str(val_m if val_m is not None else "N/A")
                str_p = f"{val_p:.4f}" if isinstance(val_p, float) else str(val_p if val_p is not None else "N/A")
                diff_str = (
                    format_diff(val_m, val_p)
                    if isinstance(val_m, (int, float)) and isinstance(val_p, (int, float))
                    else "-"
                )
                f.write(f"| {prefix} | {label} | {str_m} | {str_p} | {diff_str} |\n")
        f.write("\n")
    print(f"Saved Markdown report to: {md_path}\n")


if __name__ == "__main__":
    main()
