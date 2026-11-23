#!/usr/bin/env python3
"""Validate taxonomy, class mapping, and class distribution for Phase A."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def validate_taxonomy_and_mapping(
    rod_cfg_path: Path,
    mapping_cfg_path: Path,
    danger_cfg_path: Path,
    class_dist_csv: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    warnings = []

    # 1. Load configs
    with open(rod_cfg_path) as f:
        rod_cfg = yaml.safe_load(f)
    with open(mapping_cfg_path) as f:
        mapping_cfg = yaml.safe_load(f)
    with open(danger_cfg_path) as f:
        danger_cfg = yaml.safe_load(f)

    rod_classes = rod_cfg["classes"]
    mapping = mapping_cfg["mapping"]
    danger_groups = danger_cfg["groups"]

    # 2. Check uniqueness of ROD IDs & names
    if len(rod_classes) != rod_cfg["dataset"]["num_classes"]:
        errors.append(f"num_classes mismatch: declared {rod_cfg['dataset']['num_classes']}, found {len(rod_classes)}")

    seen_names = set()
    for cid, cname in rod_classes.items():
        if cname in seen_names:
            errors.append(f"Duplicate class name in ROD taxonomy: {cname}")
        seen_names.add(cname)

    # 3. Check mapping completeness
    for cid, cname in rod_classes.items():
        if cid not in mapping:
            errors.append(f"Missing mapping entry for class ID {cid} ({cname})")
            continue
        entry = mapping[cid]
        if entry["rod_name"] != cname:
            errors.append(f"Name mismatch in mapping for ID {cid}: expected '{cname}', got '{entry['rod_name']}'")

        coco_id = entry.get("coco_id")
        coco_name = entry.get("coco_name")
        if coco_id is not None:
            if coco_id < 0 or coco_id >= len(COCO_80):
                errors.append(f"Invalid COCO ID {coco_id} for ROD class {cname}")
            elif COCO_80[coco_id] != coco_name:
                errors.append(f"COCO ID {coco_id} name mismatch: expected '{COCO_80[coco_id]}', got '{coco_name}'")

    # 4. Check danger group consistency
    all_grouped_classes = set()
    for gname, gdata in danger_groups.items():
        for c in gdata["classes"]:
            if c not in seen_names:
                errors.append(f"Unknown class '{c}' listed in danger group '{gname}'")
            all_grouped_classes.add(c)

    # 5. Load class statistics if available
    class_stats = []
    test_instances = {}
    if class_dist_csv.exists():
        with open(class_dist_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                cid = int(r["class_id"])
                cname = r["class_name"]
                tst = int(r["test"])
                test_instances[cname] = tst

                is_shared = mapping[cid]["coco_id"] is not None
                is_critical = cname in danger_groups["critical"]["classes"]

                if tst < 50:
                    warnings.append(f"Class '{cname}' has <50 test instances ({tst})")
                if is_critical and tst == 0:
                    warnings.append(f"Critical class '{cname}' has 0 instances in test split")

                class_stats.append({
                    "class_id": cid,
                    "class_name": cname,
                    "shared_with_coco": is_shared,
                    "coco_id": mapping[cid]["coco_id"],
                    "coco_name": mapping[cid]["coco_name"],
                    "is_critical": is_critical,
                    "train_instances": int(r["train"]),
                    "valid_instances": int(r["valid"]),
                    "test_instances": tst,
                    "total_instances": int(r["total"]),
                })

        # Save class_statistics.csv
        stat_out_path = output_dir / "class_statistics.csv"
        with open(stat_out_path, "w", newline="") as f:
            if class_stats:
                writer = csv.DictWriter(f, fieldnames=class_stats[0].keys())
                writer.writeheader()
                writer.writerows(class_stats)

    report = {
        "status": "passed" if len(errors) == 0 else "failed",
        "taxonomy": "ROD-Dataset-25",
        "total_classes": len(rod_classes),
        "shared_coco_classes_count": sum(1 for v in mapping.values() if v["coco_id"] is not None),
        "domain_only_classes_count": sum(1 for v in mapping.values() if v["coco_id"] is None),
        "critical_classes_count": len(danger_groups["critical"]["classes"]),
        "errors": errors,
        "warnings": warnings,
    }

    report_path = output_dir / "class_mapping_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate class mapping and taxonomy")
    parser.add_argument("--rod-config", type=Path, default=Path("configs/datasets/rod.yaml"))
    parser.add_argument("--mapping-config", type=Path, default=Path("configs/datasets/rod_to_coco.yaml"))
    parser.add_argument("--danger-config", type=Path, default=Path("configs/danger_classes.yaml"))
    parser.add_argument("--class-dist", type=Path, default=Path("datasets/manifests/class_distribution.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/summaries"))

    args = parser.parse_args()

    report = validate_taxonomy_and_mapping(
        args.rod_config,
        args.mapping_config,
        args.danger_config,
        args.class_dist,
        args.output_dir,
    )

    print(f"Taxonomy validation: {report['status']}")
    print(f"Total classes: {report['total_classes']}")
    print(f"Shared COCO: {report['shared_coco_classes_count']}, Domain-only: {report['domain_only_classes_count']}")
    if report["errors"]:
        print(f"Errors: {report['errors']}")
        sys.exit(1)
    if report["warnings"]:
        print(f"Warnings ({len(report['warnings'])}): {report['warnings'][:5]}...")


if __name__ == "__main__":
    main()
