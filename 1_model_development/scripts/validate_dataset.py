#!/usr/bin/env python3
"""Validate dataset integrity, labels, bounding boxes, and split leakage.

Generates validation artifacts, manifests, and class statistics for Phase A.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

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
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def classify_size(pixel_w: float, pixel_h: float) -> str:
    area = pixel_w * pixel_h
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    return "large"


def validate_dataset(dataset_dir: Path, manifest_dir: Path, nc: int = 25) -> dict:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid", "test"]

    split_counts = {s: {"images": 0, "labels": 0, "boxes": 0} for s in splits}
    class_counts = {s: defaultdict(int) for s in splits}
    size_counts = {s: {"small": 0, "medium": 0, "large": 0} for s in splits}

    invalid_annotations = []
    duplicate_images = []
    images_manifest_rows = []
    checksum_lines = []

    seen_filenames = {}
    seen_hashes = {}

    leakage_detected = []

    for split in splits:
        split_dir = dataset_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        labels = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []

        split_counts[split]["images"] = len(images)
        split_counts[split]["labels"] = len(labels)

        lbl_stems = {lbl.stem: lbl for lbl in labels}

        for img_path in images:
            stem = img_path.stem

            # Check filename collision across splits
            if stem in seen_filenames:
                leakage_detected.append({
                    "stem": stem,
                    "first_split": seen_filenames[stem],
                    "second_split": split,
                    "type": "filename_collision"
                })
            else:
                seen_filenames[stem] = split

            # Content hash
            img_hash = sha256_file(img_path)
            checksum_lines.append(f"{img_hash}  {img_path.relative_to(dataset_dir.parent)}")

            if img_hash in seen_hashes:
                prev_split, prev_path = seen_hashes[img_hash]
                duplicate_images.append({
                    "hash": img_hash,
                    "split_1": prev_split,
                    "path_1": str(prev_path),
                    "split_2": split,
                    "path_2": str(img_path),
                })
                if prev_split != split:
                    leakage_detected.append({
                        "hash": img_hash,
                        "first_split": prev_split,
                        "second_split": split,
                        "type": "content_leakage"
                    })
            else:
                seen_hashes[img_hash] = (split, img_path)

            # Read image validity & resolution
            img = cv2.imread(str(img_path))
            if img is None:
                invalid_annotations.append({
                    "split": split,
                    "file": str(img_path.name),
                    "error": "corrupted_or_unreadable_image"
                })
                continue

            h, w = img.shape[:2]

            # Label pairing
            if stem not in lbl_stems:
                invalid_annotations.append({
                    "split": split,
                    "file": str(img_path.name),
                    "error": "missing_label_file"
                })
                continue

            lbl_path = lbl_stems[stem]
            box_count_img = 0

            with open(lbl_path) as f:
                lines = [line.strip() for line in f if line.strip()]

            for line_idx, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_annotations.append({
                        "split": split,
                        "file": str(lbl_path.name),
                        "line": line_idx,
                        "error": f"invalid_format_expected_5_tokens_got_{len(parts)}"
                    })
                    continue

                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError as e:
                    invalid_annotations.append({
                        "split": split,
                        "file": str(lbl_path.name),
                        "line": line_idx,
                        "error": f"parse_error_{e}"
                    })
                    continue

                # Check class id
                if cls_id < 0 or cls_id >= nc:
                    invalid_annotations.append({
                        "split": split,
                        "file": str(lbl_path.name),
                        "line": line_idx,
                        "error": f"class_id_out_of_range_{cls_id}"
                    })
                    continue

                # Check dimensions > 0
                if bw <= 0.0 or bh <= 0.0:
                    invalid_annotations.append({
                        "split": split,
                        "file": str(lbl_path.name),
                        "line": line_idx,
                        "error": "zero_or_negative_box_dimensions"
                    })
                    continue

                # Check coords in [0, 1]
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                    invalid_annotations.append({
                        "split": split,
                        "file": str(lbl_path.name),
                        "line": line_idx,
                        "error": "coordinates_outside_unit_range"
                    })
                    continue

                # Check boundary overflow
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    invalid_annotations.append({
                        "split": split,
                        "file": str(lbl_path.name),
                        "line": line_idx,
                        "error": "box_exceeds_image_boundaries"
                    })
                    continue

                pixel_w = bw * w
                pixel_h = bh * h
                size_cat = classify_size(pixel_w, pixel_h)

                split_counts[split]["boxes"] += 1
                class_counts[split][cls_id] += 1
                size_counts[split][size_cat] += 1
                box_count_img += 1

            images_manifest_rows.append({
                "split": split,
                "image_name": img_path.name,
                "width": w,
                "height": h,
                "sha256": img_hash,
                "box_count": box_count_img,
            })

        # Check for orphan labels
        img_stems = {img.stem for img in images}
        for lbl_path in labels:
            if lbl_path.stem not in img_stems:
                invalid_annotations.append({
                    "split": split,
                    "file": str(lbl_path.name),
                    "error": "orphan_label_missing_image"
                })

    # Write manifests and reports
    # 1. Checksums
    checksums_path = manifest_dir / "checksums.sha256"
    with open(checksums_path, "w") as f:
        f.write("\n".join(checksum_lines) + "\n")

    # 2. Images manifest CSV
    images_manifest_path = manifest_dir / "images_manifest.csv"
    with open(images_manifest_path, "w", newline="") as f:
        if images_manifest_rows:
            writer = csv.DictWriter(f, fieldnames=images_manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(images_manifest_rows)

    # 3. Invalid annotations CSV
    invalid_csv = manifest_dir / "invalid_annotations.csv"
    with open(invalid_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "line", "error"])
        writer.writeheader()
        for item in invalid_annotations:
            writer.writerow({
                "split": item.get("split", ""),
                "file": item.get("file", ""),
                "line": item.get("line", ""),
                "error": item.get("error", "")
            })

    # 4. Duplicate images CSV
    dup_csv = manifest_dir / "duplicate_images.csv"
    with open(dup_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hash", "split_1", "path_1", "split_2", "path_2"])
        writer.writeheader()
        writer.writerows(duplicate_images)

    # 5. Split statistics CSV
    split_stat_csv = manifest_dir / "split_statistics.csv"
    with open(split_stat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "images", "labels", "boxes"])
        writer.writeheader()
        for split in splits:
            writer.writerow({
                "split": split,
                "images": split_counts[split]["images"],
                "labels": split_counts[split]["labels"],
                "boxes": split_counts[split]["boxes"]
            })

    # 6. Class distribution CSV
    class_dist_csv = manifest_dir / "class_distribution.csv"
    with open(class_dist_csv, "w", newline="") as f:
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
                "total": tr + val + tst
            })

    # 7. Object size distribution CSV
    size_dist_csv = manifest_dir / "object_size_distribution.csv"
    with open(size_dist_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "small", "medium", "large", "total"])
        writer.writeheader()
        for split in splits:
            sm = size_counts[split]["small"]
            md = size_counts[split]["medium"]
            lg = size_counts[split]["large"]
            writer.writerow({
                "split": split,
                "small": sm,
                "medium": md,
                "large": lg,
                "total": sm + md + lg
            })

    # 8. Validation report JSON
    report = {
        "validated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_root": str(dataset_dir),
        "total_images": sum(s["images"] for s in split_counts.values()),
        "total_boxes": sum(s["boxes"] for s in split_counts.values()),
        "split_counts": split_counts,
        "object_size_counts": size_counts,
        "invalid_annotations_count": len(invalid_annotations),
        "duplicate_images_count": len(duplicate_images),
        "split_leakage_detected": len(leakage_detected) > 0,
        "leakage_events": leakage_detected,
        "status": "passed" if len(invalid_annotations) == 0 and len(leakage_detected) == 0 else "failed"
    }

    report_json_path = manifest_dir / "validation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)

    # 9. Dataset manifest JSON
    dataset_manifest = {
        "dataset_name": "ROD-Dataset",
        "version": "rod_v1",
        "upstream_source": "https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO",
        "paper": "arXiv:2611.00681",
        "license": "MIT",
        "taxonomy": {
            "num_classes": nc,
            "classes": ROD_CLASSES
        },
        "splits": split_counts,
        "manifests": {
            "images_manifest": str(images_manifest_path.name),
            "checksums": str(checksums_path.name),
            "validation_report": str(report_json_path.name),
            "class_distribution": str(class_dist_csv.name)
        },
        "validation_status": report["status"],
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    manifest_json_path = manifest_dir / "dataset_manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset and generate manifests")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/processed/rod_v1"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("datasets/manifests"))
    args = parser.parse_args()

    report = validate_dataset(args.dataset_dir, args.manifest_dir)
    print(f"Validation status: {report['status']}")
    print(f"Images: {report['total_images']}, Boxes: {report['total_boxes']}")
    print(f"Invalid annotations: {report['invalid_annotations_count']}")
    print(f"Leakage detected: {report['split_leakage_detected']}")

    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
