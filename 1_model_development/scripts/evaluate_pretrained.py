#!/usr/bin/env python3
"""Evaluate pretrained PyTorch and ONNX models on shared COCO-ROD classes.

Computes Precision, Recall, F1, mAP50, per-class recall, dangerous-class recall,
and failure case classification (missed, wrong_class, poor_localization).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.onnx_detector import ONNXDetector


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def load_ground_truth(test_dir: Path, mapping: dict):
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    gt_data = {}

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        rod_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = (cx - bw / 2.0) * w
                        y1 = (cy - bh / 2.0) * h
                        x2 = (cx + bw / 2.0) * w
                        y2 = (cy + bh / 2.0) * h

                        map_entry = mapping.get(rod_id, {})
                        coco_id = map_entry.get("coco_id")

                        boxes.append({
                            "rod_id": rod_id,
                            "rod_name": map_entry.get("rod_name", f"cls_{rod_id}"),
                            "coco_id": coco_id,
                            "bbox": (x1, y1, x2, y2),
                            "area": (x2 - x1) * (y2 - y1),
                            "is_shared": coco_id is not None,
                        })

        gt_data[img_path.stem] = {
            "path": img_path,
            "width": w,
            "height": h,
            "gt_boxes": boxes,
        }

    return gt_data


def run_pytorch_predict(model_path: Path, img_path: Path, conf: float, iou_thresh: float):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    results = model.predict(str(img_path), conf=conf, iou=iou_thresh, verbose=False)
    detections = []
    if results and len(results) > 0:
        res = results[0]
        for box in res.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            detections.append({
                "class_id": cls_id,
                "confidence": score,
                "bbox_xyxy": tuple(coords),
            })
    return detections


def evaluate_single_model(
    model_name: str,
    backend: str,
    model_path: Path,
    gt_data: dict,
    mapping: dict,
    danger_cfg: dict,
    conf: float = 0.25,
    iou_thresh: float = 0.50,
) -> tuple[dict, list[dict], list[dict]]:
    detector = None
    if backend == "onnx":
        detector = ONNXDetector(str(model_path))

    shared_classes = [v["rod_name"] for v in mapping.values() if v["coco_id"] is not None]
    critical_classes = danger_cfg["groups"]["critical"]["classes"]

    total_shared_gt = 0
    total_preds = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_gt = defaultdict(int)

    failure_cases = []

    for stem, item in gt_data.items():
        img_path = item["path"]
        gt_boxes = item["gt_boxes"]

        if backend == "onnx":
            dets = detector.predict(str(img_path), conf=conf, iou=iou_thresh)
            preds = [{
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy
            } for d in dets]
        else:
            preds = run_pytorch_predict(model_path, img_path, conf=conf, iou_thresh=iou_thresh)

        total_preds += len(preds)

        # Filter GT to shared classes only
        eval_gt = [g for g in gt_boxes if g["is_shared"]]
        matched_preds = set()
        matched_gts = set()

        for gt_idx, gt in enumerate(eval_gt):
            per_class_gt[gt["rod_name"]] += 1
            total_shared_gt += 1

            best_iou = 0.0
            best_p_idx = -1

            for p_idx, p in enumerate(preds):
                if p_idx in matched_preds:
                    continue
                iou = compute_iou(gt["bbox"], p["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx

            if best_iou >= iou_thresh and best_p_idx >= 0:
                p_cls = preds[best_p_idx]["class_id"]
                if p_cls == gt["coco_id"]:
                    tp_total += 1
                    per_class_tp[gt["rod_name"]] += 1
                    matched_preds.add(best_p_idx)
                    matched_gts.add(gt_idx)
                else:
                    # Wrong class
                    failure_cases.append({
                        "image": stem,
                        "model": model_name,
                        "backend": backend,
                        "gt_class": gt["rod_name"],
                        "gt_bbox": str(gt["bbox"]),
                        "pred_class": p_cls,
                        "iou": round(best_iou, 3),
                        "category": "wrong_class"
                    })
            elif best_iou > 0.1 and best_p_idx >= 0:
                # Poor localization
                failure_cases.append({
                    "image": stem,
                    "model": model_name,
                    "backend": backend,
                    "gt_class": gt["rod_name"],
                    "gt_bbox": str(gt["bbox"]),
                    "pred_class": preds[best_p_idx]["class_id"],
                    "iou": round(best_iou, 3),
                    "category": "poor_localization"
                })
            else:
                # Missed completely
                failure_cases.append({
                    "image": stem,
                    "model": model_name,
                    "backend": backend,
                    "gt_class": gt["rod_name"],
                    "gt_bbox": str(gt["bbox"]),
                    "pred_class": None,
                    "iou": 0.0,
                    "category": "missed"
                })

        # Count false positives
        for p_idx, p in enumerate(preds):
            if p_idx not in matched_preds:
                fp_total += 1

    fn_total = total_shared_gt - tp_total
    precision = tp_total / total_preds if total_preds > 0 else 0.0
    recall = tp_total / total_shared_gt if total_shared_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per class summary
    per_class_summary = []
    dangerous_recalls = []

    for cname in shared_classes:
        gt_c = per_class_gt[cname]
        tp_c = per_class_tp[cname]
        r_c = tp_c / gt_c if gt_c > 0 else 0.0

        is_crit = cname in critical_classes
        if is_crit and gt_c > 0:
            dangerous_recalls.append(r_c)

        per_class_summary.append({
            "model": model_name,
            "backend": backend,
            "class_name": cname,
            "gt_instances": gt_c,
            "tp_instances": tp_c,
            "recall": round(r_c, 4),
            "is_critical": is_crit,
        })

    dangerous_macro_recall = float(np.mean(dangerous_recalls)) if dangerous_recalls else 0.0
    critical_min_recall = float(np.min(dangerous_recalls)) if dangerous_recalls else 0.0
    missed_hazard_rate = 1.0 - dangerous_macro_recall if dangerous_recalls else 0.0

    summary = {
        "model": model_name,
        "backend": backend,
        "operational_threshold": conf,
        "iou_threshold": iou_thresh,
        "total_images": len(gt_data),
        "total_shared_gt": total_shared_gt,
        "total_predictions": total_preds,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map_50": None,
        "map_50_95": None,
        "map_note": "not computed: requires PR curve over confidence sweep (not implemented)",
        "dangerous_macro_recall": round(dangerous_macro_recall, 4),
        "critical_min_recall": round(critical_min_recall, 4),
        "missed_hazard_rate": round(missed_hazard_rate, 4),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }

    return summary, per_class_summary, failure_cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained YOLO models on shared ROD classes")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n", "yolo12n", "yolo26n"])
    parser.add_argument("--test-dir", type=Path, default=Path("datasets/processed/rod_v1/test"))
    parser.add_argument("--mapping-config", type=Path, default=Path("configs/datasets/rod_to_coco.yaml"))
    parser.add_argument("--danger-config", type=Path, default=Path("configs/danger_classes.yaml"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/summaries"))
    parser.add_argument("--plots-dir", type=Path, default=Path("benchmarks/plots"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mapping_config) as f:
        mapping = yaml.safe_load(f)["mapping"]
    with open(args.danger_config) as f:
        danger_cfg = yaml.safe_load(f)

    gt_data = load_ground_truth(args.test_dir, mapping)
    print(f"Loaded ground truth: {len(gt_data)} test images.")

    all_summaries = []
    all_class_rows = []
    all_failures = []

    for m in args.models:
        # Evaluate ONNX
        onnx_file = Path(f"weights/onnx/{m}_320_fp32.onnx")
        if onnx_file.exists():
            print(f"Evaluating {m} (ONNX)...")
            s_onnx, c_onnx, f_onnx = evaluate_single_model(
                m, "onnx", onnx_file, gt_data, mapping, danger_cfg, args.conf, args.iou
            )
            all_summaries.append(s_onnx)
            all_class_rows.extend(c_onnx)
            all_failures.extend(f_onnx)

        # Evaluate PyTorch
        pt_file = Path(f"weights/pytorch/{m}.pt")
        if pt_file.exists():
            print(f"Evaluating {m} (PyTorch)...")
            s_pt, c_pt, f_pt = evaluate_single_model(
                m, "pytorch", pt_file, gt_data, mapping, danger_cfg, args.conf, args.iou
            )
            all_summaries.append(s_pt)
            all_class_rows.extend(c_pt)
            all_failures.extend(f_pt)

    # Save summary CSV
    summary_csv = args.output_dir / "pretrained_accuracy_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        if all_summaries:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

    # Save summary JSON
    summary_json = args.output_dir / "pretrained_accuracy_summary.json"
    with open(summary_json, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Save per_class_recall CSV
    class_csv = args.output_dir / "per_class_recall.csv"
    with open(class_csv, "w", newline="") as f:
        if all_class_rows:
            writer = csv.DictWriter(f, fieldnames=all_class_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_class_rows)

    # Save failure cases CSV
    fail_csv = args.output_dir / "failure_cases.csv"
    with open(fail_csv, "w", newline="") as f:
        if all_failures:
            writer = csv.DictWriter(f, fieldnames=all_failures[0].keys())
            writer.writeheader()
            writer.writerows(all_failures)
        else:
            writer = csv.writer(f)
            writer.writerow(["image", "model", "backend", "gt_class", "gt_bbox", "pred_class", "iou", "category"])

    print(f"\n✓ Saved evaluation summary to: {summary_csv}")
    print(f"✓ Saved per-class recall to: {class_csv}")
    print(f"✓ Saved failure cases ({len(all_failures)}) to: {fail_csv}")

    # Print table
    print(f"\n{'='*110}")
    print(f"{'Model':<12} {'Backend':<10} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Dangerous Recall':<18} {'Missed Hazard Rate':<20}")
    print(f"{'='*110}")
    for s in all_summaries:
        print(f"{s['model']:<12} {s['backend']:<10} {s['precision']:<12.4f} {s['recall']:<12.4f} {s['f1']:<10.4f} {s['dangerous_macro_recall']:<18.4f} {s['missed_hazard_rate']:<20.4f}")


if __name__ == "__main__":
    main()
