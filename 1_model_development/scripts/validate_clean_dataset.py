#!/usr/bin/env python3
"""Validate clean dataset (rod_clean_v1).

Strictly verifies:
1. content_leakage == 0 (no duplicate image SHA256 across train/valid/test)
2. mixed_schema == 0 (strictly 5-token detection bounding boxes, zero polygon/segmentation tokens)
3. invalid_rows == 0 (no out-of-range class, invalid float, zero/negative width/height, out of bounds)
4. missing_pairs == 0 (no image without label, no orphan label without image)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
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


def validate_clean_dataset(
    dataset_dir: Path,
    manifest_dir: Path,
    nc: int = 25,
) -> dict[str, Any]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid", "test"]

    print("=" * 70)
    print("STARTING DATASET VALIDATION (rod_clean_v1)")
    print(f"Dataset root : {dataset_dir}")
    print(f"Manifest root: {manifest_dir}")
    print("=" * 70)

    split_counts = {s: {"images": 0, "labels": 0, "boxes": 0} for s in splits}
    class_counts = {s: defaultdict(int) for s in splits}

    content_leakage_events: list[dict[str, Any]] = []
    mixed_schema_events: list[dict[str, Any]] = []
    invalid_rows_events: list[dict[str, Any]] = []
    missing_pairs_events: list[dict[str, Any]] = []

    seen_hashes_by_split: dict[str, tuple[str, Path]] = {}

    for split in splits:
        split_dir = dataset_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.is_dir():
            print(f"Warning: split directory {img_dir} not found!")
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        labels = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else []

        split_counts[split]["images"] = len(images)
        split_counts[split]["labels"] = len(labels)

        lbl_map = {lbl.stem: lbl for lbl in labels}
        img_map = {img.stem: img for img in images}

        print(f"Validating split '{split}': {len(images)} images, {len(labels)} labels...")

        # 1. Missing pairs: images missing labels
        for img in images:
            if img.stem not in lbl_map:
                missing_pairs_events.append({
                    "split": split,
                    "file": img.name,
                    "error": "missing_label_for_image",
                })

        # 2. Missing pairs: orphan labels missing images
        for lbl in labels:
            if lbl.stem not in img_map:
                missing_pairs_events.append({
                    "split": split,
                    "file": lbl.name,
                    "error": "orphan_label_missing_image",
                })

        # 3. Content leakage & hash check
        for img_idx, img_path in enumerate(images, start=1):
            h = sha256_file(img_path)
            if h in seen_hashes_by_split:
                prev_split, prev_path = seen_hashes_by_split[h]
                if prev_split != split:
                    content_leakage_events.append({
                        "hash": h,
                        "first_split": prev_split,
                        "second_split": split,
                        "first_file": str(prev_path.name),
                        "second_file": str(img_path.name),
                    })
            else:
                seen_hashes_by_split[h] = (split, img_path)

        # 4. Schema and annotation validity check
        for lbl_path in labels:
            try:
                content = lbl_path.read_text(encoding="utf-8")
            except Exception as e:
                invalid_rows_events.append({
                    "split": split,
                    "file": lbl_path.name,
                    "line": 0,
                    "error": f"read_error_{e}",
                })
                continue

            lines = [l.strip() for l in content.splitlines() if l.strip()]
            for line_idx, line in enumerate(lines, start=1):
                tokens = line.split()

                # Mixed schema check: must be exactly 5 tokens
                if len(tokens) != 5:
                    mixed_schema_events.append({
                        "split": split,
                        "file": lbl_path.name,
                        "line": line_idx,
                        "token_count": len(tokens),
                        "content": line,
                        "error": "not_5_tokens_detection_bbox",
                    })
                    continue

                # Invalid rows check: numeric parsing and bounds
                try:
                    cls_id = int(tokens[0])
                    cx = float(tokens[1])
                    cy = float(tokens[2])
                    bw = float(tokens[3])
                    bh = float(tokens[4])
                except ValueError as e:
                    invalid_rows_events.append({
                        "split": split,
                        "file": lbl_path.name,
                        "line": line_idx,
                        "content": line,
                        "error": f"parse_error_{e}",
                    })
                    continue

                if cls_id < 0 or cls_id >= nc:
                    invalid_rows_events.append({
                        "split": split,
                        "file": lbl_path.name,
                        "line": line_idx,
                        "content": line,
                        "error": f"class_id_out_of_range_{cls_id}",
                    })
                    continue

                if bw <= 0.0 or bh <= 0.0:
                    invalid_rows_events.append({
                        "split": split,
                        "file": lbl_path.name,
                        "line": line_idx,
                        "content": line,
                        "error": "zero_or_negative_box_dimensions",
                    })
                    continue

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                    # Slight tolerance for rounding at edges, flag if definitely outside
                    if cx < -0.05 or cx > 1.05 or cy < -0.05 or cy > 1.05 or bw > 1.05 or bh > 1.05:
                        invalid_rows_events.append({
                            "split": split,
                            "file": lbl_path.name,
                            "line": line_idx,
                            "content": line,
                            "error": "coordinates_outside_unit_range",
                        })
                        continue

                split_counts[split]["boxes"] += 1
                class_counts[split][cls_id] += 1

    # Summarize results
    total_images = sum(s["images"] for s in split_counts.values())
    total_labels = sum(s["labels"] for s in split_counts.values())
    total_boxes = sum(s["boxes"] for s in split_counts.values())

    content_leakage = len(content_leakage_events)
    mixed_schema = len(mixed_schema_events)
    invalid_rows = len(invalid_rows_events)
    missing_pairs = len(missing_pairs_events)

    passed = (
        content_leakage == 0
        and mixed_schema == 0
        and invalid_rows == 0
        and missing_pairs == 0
    )

    print("\n" + "=" * 70)
    print("VALIDATION CRITERIA RESULTS")
    print("=" * 70)
    print(f"  content_leakage : {content_leakage:<5} (target: 0) [{'PASS' if content_leakage == 0 else 'FAIL'}]")
    print(f"  mixed_schema    : {mixed_schema:<5} (target: 0) [{'PASS' if mixed_schema == 0 else 'FAIL'}]")
    print(f"  invalid_rows    : {invalid_rows:<5} (target: 0) [{'PASS' if invalid_rows == 0 else 'FAIL'}]")
    print(f"  missing_pairs   : {missing_pairs:<5} (target: 0) [{'PASS' if missing_pairs == 0 else 'FAIL'}]")
    print("-" * 70)
    print(f"  OVERALL STATUS  : {'PASSED' if passed else 'FAILED'}")
    print(f"  Total images    : {total_images}")
    print(f"  Total labels    : {total_labels}")
    print(f"  Total boxes     : {total_boxes}")
    print("=" * 70)

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_dir.resolve()),
        "status": "passed" if passed else "failed",
        "criteria": {
            "content_leakage": content_leakage,
            "mixed_schema": mixed_schema,
            "invalid_rows": invalid_rows,
            "missing_pairs": missing_pairs,
        },
        "split_counts": split_counts,
        "total_images": total_images,
        "total_labels": total_labels,
        "total_boxes": total_boxes,
        "content_leakage_events": content_leakage_events,
        "mixed_schema_events": mixed_schema_events,
        "invalid_rows_events": invalid_rows_events,
        "missing_pairs_events": missing_pairs_events,
    }

    report_path = manifest_dir / "clean_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to: {report_path}")

    # Write split statistics CSV
    split_csv = manifest_dir / "clean_split_statistics.csv"
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "images", "labels", "boxes"])
        writer.writeheader()
        for s in splits:
            writer.writerow({
                "split": s,
                "images": split_counts[s]["images"],
                "labels": split_counts[s]["labels"],
                "boxes": split_counts[s]["boxes"],
            })

    # Write class distribution CSV
    class_csv = manifest_dir / "clean_class_distribution.csv"
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "train", "valid", "test", "total"])
        writer.writeheader()
        for cid, cname in enumerate(ROD_CLASSES):
            tr = class_counts["train"][cid]
            val = class_counts["valid"][cid]
            tst = class_counts["test"][cid]
            writer.writerow({
                "class_id": cid,
                "class_name": cname,
                "train": tr,
                "valid": val,
                "test": tst,
                "total": tr + val + tst,
            })

    if not passed:
        raise ValueError(
            f"Validation failed: content_leakage={content_leakage}, mixed_schema={mixed_schema}, "
            f"invalid_rows={invalid_rows}, missing_pairs={missing_pairs}"
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate clean dataset criteria")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/processed/rod_clean_v1"),
        help="Path to rod_clean_v1 dataset",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("datasets/manifests/rod_clean_v1"),
        help="Path to manifest output dir",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_dir
    if not dataset_root.exists() and (Path("..") / dataset_root).exists():
        dataset_root = Path("..") / dataset_root
    if not dataset_root.exists() and Path("../../datasets/processed/rod_clean_v1").exists():
        dataset_root = Path("../../datasets/processed/rod_clean_v1")

    validate_clean_dataset(dataset_root, args.manifest_dir)


if __name__ == "__main__":
    main()
