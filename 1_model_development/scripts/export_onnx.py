#!/usr/bin/env python3
"""Export validated PyTorch checkpoints to ONNX format.

Reads weights/manifest.json and exports only models with status='validated'.
Creates standardized ONNX files with naming convention: {model}_{imgsz}_fp32.onnx

Usage:
    python scripts/export_onnx.py --models yolov8n yolo11n
    python scripts/export_onnx.py --models yolov8n --config configs/export.yaml
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def export_model(
    model_name: str,
    checkpoint_path: Path,
    output_dir: Path,
    export_config: dict,
    metadata_dir: Path,
) -> dict:
    """Export a single model to ONNX.
    
    Returns:
        dict with status, path, sha256, error, etc.
    """
    result = {
        "model": model_name,
        "status": "failed",
        "error": None,
        "export_time_sec": 0,
        "onnx_path": None,
        "onnx_sha256": None,
        "onnx_size_mb": 0,
    }
    
    start_time = time.time()
    
    try:
        print(f"\n{'='*60}")
        print(f"Exporting {model_name}")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load model
        model = YOLO(str(checkpoint_path))
        
        # Build output filename: {model}_{imgsz}_fp32.onnx
        imgsz = export_config["imgsz"]
        onnx_filename = f"{model_name}_{imgsz}_fp32.onnx"
        
        # Export
        export_args = {
            "format": export_config["format"],
            "imgsz": imgsz,
            "batch": export_config["batch"],
            "dynamic": export_config["dynamic"],
            "simplify": export_config["simplify"],
            "opset": export_config["opset"],
            "half": export_config["half"],
            "int8": export_config["int8"],
            "nms": export_config["nms"],
        }
        
        print(f"Export args: {export_args}")
        
        # Ultralytics export returns path
        exported_path = model.export(**export_args)
        
        # Move to standardized location
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / onnx_filename
        
        # Ultralytics may return string or Path
        exported_path = Path(exported_path)
        
        if not exported_path.is_absolute():
            exported_path = Path.cwd() / exported_path
        if not final_path.is_absolute():
            final_path = Path.cwd() / final_path
        
        if exported_path != final_path:
            if final_path.exists():
                final_path.unlink()
            exported_path.rename(final_path)
        
        # Compute hash and size
        sha256 = sha256_file(final_path)
        size_mb = final_path.stat().st_size / (1024 * 1024)
        
        elapsed = time.time() - start_time
        
        # Make path relative for JSON
        try:
            rel_path = final_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = final_path
        
        result.update({
            "status": "success",
            "export_time_sec": round(elapsed, 2),
            "onnx_path": str(rel_path),
            "onnx_sha256": sha256,
            "onnx_size_mb": round(size_mb, 2),
        })
        
        print(f"✓ Export successful")
        print(f"  Path: {final_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  SHA256: {sha256[:16]}...")
        print(f"  Time: {elapsed:.2f}s")
        
        # Write metadata
        try:
            rel_checkpoint = checkpoint_path.relative_to(Path.cwd())
        except ValueError:
            rel_checkpoint = checkpoint_path
        
        metadata = {
            "model": model_name,
            "format": "onnx",
            "precision": "fp32",
            "input_size": imgsz,
            "batch_size": export_config["batch"],
            "dynamic": export_config["dynamic"],
            "opset": export_config["opset"],
            "nms_included": export_config["nms"],
            "simplify": export_config["simplify"],
            "onnx_path": str(rel_path),
            "onnx_sha256": sha256,
            "onnx_size_bytes": final_path.stat().st_size,
            "onnx_size_mb": round(size_mb, 2),
            "source_checkpoint": str(rel_checkpoint),
            "export_config": export_args,
            "export_time_sec": round(elapsed, 2),
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "status": "success",
        }
        
        metadata_path = metadata_dir / f"{model_name}_onnx.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata: {metadata_path}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        result.update({
            "status": "failed",
            "error": error_msg,
            "export_time_sec": round(elapsed, 2),
        })
        
        print(f"✗ Export failed: {error_msg}")
        
        # Write failure metadata
        metadata = {
            "model": model_name,
            "format": "onnx",
            "status": "failed",
            "error": error_msg,
            "export_time_sec": round(elapsed, 2),
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }
        
        metadata_path = metadata_dir / f"{model_name}_onnx.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to export (must exist in manifest)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/export.yaml"),
        help="Export configuration file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("weights/manifest.json"),
        help="Model manifest path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights/onnx"),
        help="Output directory for ONNX files",
    )
    
    args = parser.parse_args()
    
    # Load export config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    export_config = config["export"]
    
    # Load manifest
    if not args.manifest.exists():
        print(f"Error: Manifest not found: {args.manifest}")
        sys.exit(1)
    
    with open(args.manifest) as f:
        manifest = json.load(f)
    
    # Prepare directories
    metadata_dir = Path("weights/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export models
    results = []
    
    for model_name in args.models:
        if model_name not in manifest["models"]:
            print(f"\n⚠ Model '{model_name}' not found in manifest, skipping")
            results.append({
                "model": model_name,
                "status": "not_in_manifest",
                "error": "Model not found in manifest.json",
            })
            continue
        
        model_entry = manifest["models"][model_name]
        
        # Check PyTorch status (support both schemas)
        if "pytorch" in model_entry:
            # New schema: {"pytorch": {"status": ..., "path": ...}}
            pytorch_status = model_entry["pytorch"].get("status")
            pytorch_path = model_entry["pytorch"].get("path")
        else:
            # Old schema: {"status": ..., "checkpoint": ...}
            pytorch_status = model_entry.get("status")
            pytorch_path = model_entry.get("checkpoint")
        
        if pytorch_status != "validated":
            print(f"\n⚠ Model '{model_name}' has status '{pytorch_status}', skipping")
            results.append({
                "model": model_name,
                "status": "skipped",
                "error": f"PyTorch checkpoint not validated (status={pytorch_status})",
            })
            continue
        
        # Get checkpoint path
        if not pytorch_path:
            print(f"\n⚠ Model '{model_name}' has no checkpoint path in manifest, skipping")
            results.append({
                "model": model_name,
                "status": "skipped",
                "error": "No checkpoint path in manifest",
            })
            continue
        
        checkpoint_path = Path(pytorch_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path("weights") / checkpoint_path
        if not checkpoint_path.exists():
            print(f"\n⚠ Checkpoint not found: {checkpoint_path}, skipping")
            results.append({
                "model": model_name,
                "status": "checkpoint_missing",
                "error": f"Checkpoint file not found: {checkpoint_path}",
            })
            continue
        
        # Export
        result = export_model(
            model_name,
            checkpoint_path,
            args.output_dir,
            export_config,
            metadata_dir,
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] in ("skipped", "not_in_manifest", "checkpoint_missing")]
    
    print(f"Success: {len(success)}")
    print(f"Failed:  {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    
    if success:
        print("\n✓ Successfully exported:")
        for r in success:
            print(f"  - {r['model']}: {r['onnx_path']} ({r['onnx_size_mb']} MB)")
    
    if failed:
        print("\n✗ Failed exports:")
        for r in failed:
            print(f"  - {r['model']}: {r['error']}")
    
    if skipped:
        print("\n⚠ Skipped:")
        for r in skipped:
            print(f"  - {r['model']}: {r.get('error', 'unknown reason')}")
    
    # Update manifest with ONNX info
    print(f"\nUpdating manifest: {args.manifest}")
    for result in results:
        model_name = result["model"]
        if model_name in manifest["models"]:
            imgsz = export_config["imgsz"]
            onnx_key = f"onnx_fp32_{imgsz}"
            
            if result["status"] == "success":
                manifest["models"][model_name][onnx_key] = {
                    "status": "exported",
                    "validation": "pending",
                    "path": result["onnx_path"],
                    "sha256": result["onnx_sha256"],
                    "metadata": f"weights/metadata/{model_name}_onnx.json",
                }
            else:
                manifest["models"][model_name][onnx_key] = {
                    "status": result["status"],
                    "error": result.get("error"),
                    "metadata": f"weights/metadata/{model_name}_onnx.json",
                }
    
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Manifest updated")
    
    # Exit code
    if failed and not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
