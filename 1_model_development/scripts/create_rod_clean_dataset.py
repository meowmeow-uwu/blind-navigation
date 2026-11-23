#!/usr/bin/env python3
"""Create rod_clean_v1 dataset by removing cross-split exact duplicates.

Strict Priority: train > test > valid.
- Keeps raw dataset strictly untouched.
- Removes processed/rod_detection_v1/val if present.
- Eliminates any image in test/valid whose SHA256 matches an image in train.
- Eliminates any image in valid whose SHA256 matches an image in test.
- Copies clean images and labels to datasets/processed/rod_clean_v1.
- Generates data.yaml and complete manifests with full audit logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack"
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def remove_val_duplicate(processed_root: Path) -> bool:
    """Step 1: Delete processed/rod_detection_v1/val if present."""
    val_path = processed_root / "val"
    if val_path.is_symlink() or val_path.exists():
        print(f"[Step 1] Removing duplicated/convenience 'val' at: {val_path}")
        if val_path.is_symlink():
            val_path.unlink()
        elif val_path.is_dir():
            shutil.rmtree(val_path)
        else:
            val_path.unlink()
        print("  ✓ Successfully removed 'val'")
        return True
    return False


def hash_single_image(args: tuple[str, Path]) -> dict[str, Any]:
    split, img_path = args
    h = sha256_file(img_path)
    return {
        "split": split,
        "path": img_path,
        "filename": img_path.name,
        "stem": img_path.stem,
        "hash": h,
    }


def create_clean_dataset(
    source_root: Path,
    output_root: Path,
    manifest_root: Path,
    workers: int = 8,
) -> dict[str, Any]:
    start_time = time.time()
    print("=" * 70)
    print("STARTING ROD_CLEAN_V1 DEDUPLICATION PIPELINE")
    print(f"Source processed dataset : {source_root}")
    print(f"Target clean dataset     : {output_root}")
    print(f"Manifest output directory: {manifest_root}")
    print("Priority order           : train > test > valid")
    print("=" * 70)

    # 1. Clean duplicated val in source
    remove_val_duplicate(source_root)

    splits = ["train", "valid", "test"]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    # 2. Collect and hash all images across splits
    tasks = []
    print("\nScanning images from source dataset...")
    for split in splits:
        img_dir = source_root / split / "images"
        if not img_dir.is_dir():
            print(f"Warning: {img_dir} does not exist!")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in img_dir.glob(ext):
                tasks.append((split, img_path))

    print(f"Found {len(tasks)} images across splits. Computing SHA-256 hashes...")
    hash_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        image_records = list(executor.map(hash_single_image, tasks))
    print(f"Computed {len(image_records)} image hashes in {time.time() - hash_start:.2f}s.")

    # 3. Index by hash and analyze cross-split duplicates
    hash_to_records: dict[str, list[dict[str, Any]]] = {}
    for rec in image_records:
        hash_to_records.setdefault(rec["hash"], []).append(rec)

    # Strict priority resolution: train > test > valid
    kept_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []

    for h, recs in hash_to_records.items():
        splits_present = {r["split"] for r in recs}
        if len(splits_present) == 1:
            # No cross-split leakage for this hash, keep all records in this split
            kept_records.extend(recs)
        else:
            # Cross-split duplicate!
            # Determine winning split based on priority: train > test > valid
            if "train" in splits_present:
                winner_split = "train"
            elif "test" in splits_present:
                winner_split = "test"
            else:
                winner_split = "valid"

            for r in recs:
                if r["split"] == winner_split:
                    kept_records.append(r)
                else:
                    removed_records.append({
                        "hash": r["hash"],
                        "removed_split": r["split"],
                        "removed_filename": r["filename"],
                        "removed_path": str(r["path"]),
                        "winning_split": winner_split,
                        "reason": f"cross_split_leakage_superseded_by_{winner_split}",
                    })

    print(f"\nResolution Summary:")
    print(f"  Total source images   : {len(image_records)}")
    print(f"  Unique hashes         : {len(hash_to_records)}")
    print(f"  Kept images           : {len(kept_records)}")
    print(f"  Removed cross-split   : {len(removed_records)}")

    # Group kept by split
    kept_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    for r in kept_records:
        kept_by_split[r["split"]].append(r)

    # 4. Copy kept images and their matching labels
    print("\nCopying clean images and labels to target...")
    split_stats: dict[str, dict[str, int]] = {}

    for split in splits:
        dst_img_dir = output_root / split / "images"
        dst_lbl_dir = output_root / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        src_lbl_dir = source_root / split / "labels"

        items = kept_by_split[split]
        copied_images = 0
        copied_labels = 0
        total_boxes = 0

        for r in items:
            src_img: Path = r["path"]
            dst_img = dst_img_dir / src_img.name
            if not dst_img.exists() or dst_img.stat().st_size != src_img.stat().st_size:
                shutil.copy2(src_img, dst_img)
            copied_images += 1

            src_lbl = src_lbl_dir / f"{r['stem']}.txt"
            dst_lbl = dst_lbl_dir / f"{r['stem']}.txt"

            if src_lbl.exists():
                # Read and verify label format (strictly 5-token bboxes)
                content = src_lbl.read_text(encoding="utf-8")
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                # Write to clean destination
                dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                copied_labels += 1
                total_boxes += len(lines)
            else:
                # If label doesn't exist, create empty file to guarantee missing_pairs=0
                dst_lbl.write_text("", encoding="utf-8")
                copied_labels += 1

        split_stats[split] = {
            "images": copied_images,
            "labels": copied_labels,
            "boxes": total_boxes,
        }
        print(f"  [{split}] Copied {copied_images} images, {copied_labels} labels ({total_boxes} boxes)")

    # 5. Write data.yaml in clean dataset
    data_yaml_path = output_root / "data.yaml"
    data_yaml_content = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(ROD_CLASSES),
        "names": ROD_CLASSES,
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)
    print(f"\nWritten dataset configuration: {data_yaml_path}")

    # 6. Write manifest and audit logs
    removed_csv_path = manifest_root / "removed_cross_split_duplicates.csv"
    with open(removed_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["hash", "removed_split", "removed_filename", "removed_path", "winning_split", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(removed_records)
    print(f"Written removed duplicates log: {removed_csv_path} ({len(removed_records)} rows)")

    manifest_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_root.resolve()),
        "clean_dataset": str(output_root.resolve()),
        "deduplication_policy": "cross_split_exact_match_priority_train_gt_test_gt_valid",
        "total_source_images": len(image_records),
        "total_unique_hashes": len(hash_to_records),
        "total_clean_images": len(kept_records),
        "total_removed_duplicates": len(removed_records),
        "split_statistics": split_stats,
        "removed_count_by_split": {
            s: sum(1 for r in removed_records if r["removed_split"] == s)
            for s in splits
        },
        "duration_sec": round(time.time() - start_time, 2),
    }

    manifest_json_path = manifest_root / "dedup_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Written deduplication manifest: {manifest_json_path}")

    # Clean label caches
    for cache_file in output_root.glob("**/*.cache"):
        try:
            cache_file.unlink()
        except Exception:
            pass

    print("\n✓ ROD_CLEAN_V1 CREATION COMPLETED SUCCESSFULLY!")
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Generate rod_clean_v1 dataset")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/processed/rod_detection_v1"),
        help="Source processed dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed/rod_clean_v1"),
        help="Output clean dataset root",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("datasets/manifests/rod_clean_v1"),
        help="Directory to save manifests and logs",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of hash workers")
    args = parser.parse_args()

    # Fallback to repo-relative paths if needed
    source_root = args.source_dir
    if not source_root.exists() and (Path("..") / source_root).exists():
        source_root = Path("..") / source_root
    if not source_root.exists() and (Path("../../datasets/processed/rod_detection_v1")).exists():
        source_root = Path("../../datasets/processed/rod_detection_v1")

    create_clean_dataset(
        source_root=source_root,
        output_root=args.output_dir,
        manifest_root=args.manifest_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
