#!/usr/bin/env python3
"""Select 640px operational confidence thresholds on rod_clean_v1/valid."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from ultralytics.data.augment import LetterBox
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DEV_ROOT = SCRIPT_DIR.parent
DATASET_DIR = MODEL_DEV_ROOT / "datasets" / "processed" / "rod_clean_v1"
VALID_IMAGES_DIR = DATASET_DIR / "valid" / "images"
VALID_LABELS_DIR = DATASET_DIR / "valid" / "labels"
WEIGHTS_DIR = MODEL_DEV_ROOT / "weights"
REPORTS_DIR = MODEL_DEV_ROOT / "reports" / "pipeline_run"
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"
DEPLOYMENT_CONFIG_PATH = MODEL_DEV_ROOT.parent / "2_edge_device" / "deployment_config.yaml"
IMAGE_SIZE = 640
MIN_NMS_CONFIDENCE = 0.05
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300
THRESHOLDS = np.arange(5, 91, 5, dtype=np.int32) / 100.0
ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox_with_pad(image: np.ndarray, letterbox: LetterBox) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    result = letterbox(image=image)
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    pad_x = (IMAGE_SIZE - round(width * scale)) / 2
    pad_y = (IMAGE_SIZE - round(height * scale)) / 2
    return result, (round(pad_x - 0.1), round(pad_y - 0.1))


def read_targets(path: Path, shape: tuple[int, int], pad: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file() or not path.stat().st_size:
        return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)
    rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if rows.shape[1] != 5:
        raise ValueError(f"Invalid YOLO label row in {path}")
    height, width = shape
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    boxes = rows[:, 1:5].copy()
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes *= scale
    boxes[:, [0, 2]] += pad[0]
    boxes[:, [1, 3]] += pad[1]
    return torch.from_numpy(rows[:, 0].copy()), torch.from_numpy(boxes)


def metric_at_threshold(
    threshold: float,
    tp_by_class: np.ndarray,
    predictions_by_class: np.ndarray,
    target_counts: np.ndarray,
    critical_ids: list[int],
    image_count: int,
) -> dict[str, Any]:
    present_classes = np.flatnonzero(target_counts)
    per_class_recall: dict[str, float | None] = {name: None for name in ROD_CLASSES}
    per_class_precision: dict[str, float | None] = {name: None for name in ROD_CLASSES}
    per_class_fn: dict[str, int] = {}
    recalls = []
    precisions = []
    for class_id in present_classes:
        tp = int(tp_by_class[class_id])
        fp = int(predictions_by_class[class_id] - tp)
        recall = tp / int(target_counts[class_id])
        precision = tp / (tp + fp) if tp + fp else 0.0
        class_name = ROD_CLASSES[class_id]
        per_class_recall[class_name] = round(recall, 6)
        per_class_precision[class_name] = round(precision, 6)
        per_class_fn[class_name] = int(target_counts[class_id] - tp)
        recalls.append(recall)
        precisions.append(precision)

    critical_recalls = [
        per_class_recall[ROD_CLASSES[class_id]]
        for class_id in critical_ids
        if per_class_recall[ROD_CLASSES[class_id]] is not None
    ]
    total_tp = int(tp_by_class.sum())
    total_predictions = int(predictions_by_class.sum())
    return {
        "confidence_threshold": round(float(threshold), 2),
        "precision": round(float(np.mean(precisions)), 6) if precisions else 0.0,
        "recall": round(float(np.mean(recalls)), 6) if recalls else 0.0,
        "dangerous_class_recall": round(float(np.mean(critical_recalls)), 6) if critical_recalls else None,
        "critical_min_recall": round(float(min(critical_recalls)), 6) if critical_recalls else None,
        "false_negatives": int(sum(per_class_fn.values())),
        "false_positives": total_predictions - total_tp,
        "false_positives_per_image": round((total_predictions - total_tp) / image_count, 6),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_false_negatives": per_class_fn,
    }


def evaluate_model(model_name: str, artifact_key: str, critical_ids: list[int], images: list[Path], manifest: dict) -> dict[str, Any]:
    artifact = manifest["models"][model_name][artifact_key]
    onnx_path = (MODEL_DEV_ROOT / artifact["path_relative"]).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX artifact missing: {onnx_path}")
    expected_hash = artifact.get("sha256")
    if expected_hash and sha256_file(onnx_path) != expected_hash:
        raise ValueError(f"Manifest hash mismatch: {onnx_path}")
    if artifact.get("imgsz") != IMAGE_SIZE:
        raise ValueError(f"Expected a {IMAGE_SIZE}px artifact, got {artifact.get('imgsz')}: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    letterbox = LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False, stride=32)
    target_counts = np.zeros(len(ROD_CLASSES), dtype=np.int64)
    tp_by_threshold = np.zeros((len(THRESHOLDS), len(ROD_CLASSES)), dtype=np.int64)
    predictions_by_threshold = np.zeros_like(tp_by_threshold)
    matching = SimpleNamespace(iouv=torch.tensor([0.5]))

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read validation image: {image_path}")
        boxed, pad = letterbox_with_pad(image, letterbox)
        input_tensor = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        raw_output = session.run(None, {input_name: input_tensor})[0]
        if raw_output.shape != (1, len(ROD_CLASSES) + 4, (IMAGE_SIZE // 32) ** 2 * 21):
            raise ValueError(f"Unexpected ONNX output shape for {model_name}: {raw_output.shape}")
        candidates = non_max_suppression(
            torch.from_numpy(raw_output.copy()),
            conf_thres=MIN_NMS_CONFIDENCE,
            iou_thres=NMS_IOU_THRESHOLD,
            multi_label=True,
            max_det=MAX_DETECTIONS,
        )[0]
        target_cls, target_boxes = read_targets(
            VALID_LABELS_DIR / f"{image_path.stem}.txt", image.shape[:2], pad
        )
        if target_cls.numel():
            target_ids = target_cls.numpy().astype(int)
            if np.any((target_ids < 0) | (target_ids >= len(ROD_CLASSES))):
                raise ValueError(f"Invalid class ID in {image_path.stem}.txt")
            target_counts += np.bincount(target_ids, minlength=len(ROD_CLASSES))
        if target_cls.numel() and candidates.shape[0]:
            correct = BaseValidator.match_predictions(
                matching,
                candidates[:, 5],
                target_cls,
                box_iou(target_boxes, candidates[:, :4]),
            ).numpy()[:, 0]
        else:
            correct = np.zeros(candidates.shape[0], dtype=bool)

        confidences = candidates[:, 4].numpy()
        class_ids = candidates[:, 5].numpy().astype(int)
        for threshold_index, threshold in enumerate(THRESHOLDS):
            keep = confidences >= threshold
            if not np.any(keep):
                continue
            predictions_by_threshold[threshold_index] += np.bincount(
                class_ids[keep], minlength=len(ROD_CLASSES)
            )
            tp_by_threshold[threshold_index] += np.bincount(
                class_ids[keep & correct], minlength=len(ROD_CLASSES)
            )
        if index % 500 == 0 or index == len(images):
            print(f"  [{model_name}] {index}/{len(images)} validation images", flush=True)

    sweep = [
        metric_at_threshold(float(threshold), tp_by_threshold[i], predictions_by_threshold[i], target_counts, critical_ids, len(images))
        for i, threshold in enumerate(THRESHOLDS)
    ]
    eligible = [row for row in sweep if row["precision"] >= 0.70]
    selected = max(eligible, key=lambda row: (row["recall"], row["precision"], row["confidence_threshold"])) if eligible else None
    return {
        "model": model_name,
        "backend": artifact.get("runtime", "onnx"),
        "precision": artifact.get("precision"),
        "artifact": artifact["path_relative"],
        "sha256": sha256_file(onnx_path),
        "imgsz": IMAGE_SIZE,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "selection_rule": "maximum macro recall subject to macro precision >= 0.70; ties prefer precision then higher confidence",
        "selected": selected,
        "threshold_sweep": sweep,
    }


def main() -> None:
    images = sorted(
        path for path in VALID_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) != 3448:
        raise RuntimeError(f"Expected 3448 validation images; found {len(images)}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    critical_names = set(
        (yaml.safe_load((MODEL_DEV_ROOT / "configs" / "danger_classes.yaml").read_text()) or {})
        .get("groups", {}).get("critical", {}).get("classes", [])
    )
    critical_ids = [ROD_CLASSES.index(name) for name in critical_names]
    if len(critical_ids) != len(critical_names):
        raise ValueError("Critical class list does not match ROD_CLASSES")

    specs = (
        ("yolov12n", "onnx_fp32_640"),
        ("yolov26n", "onnx_int8_640"),
    )
    print(f"Threshold sweep on rod_clean_v1/valid: {len(images)} images, {len(THRESHOLDS)} thresholds")
    model_results = {
        model: evaluate_model(model, key, critical_ids, images, manifest)
        for model, key in specs
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "rod_clean_v1/valid",
        "split": "valid",
        "validation_images": len(images),
        "imgsz": IMAGE_SIZE,
        "confidence_thresholds": [round(float(x), 2) for x in THRESHOLDS],
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "metrics_definition": {
            "precision_recall": "macro mean over classes with validation labels at IoU 0.50 and the selected confidence threshold",
            "dangerous_class_recall": "macro mean recall over configs/danger_classes.yaml groups.critical",
            "critical_min_recall": "minimum recall across the configured critical classes",
            "false_negatives": "ground-truth boxes unmatched at IoU >= 0.50",
            "false_positives_per_image": "unmatched detections divided by validation image count",
        },
        "models": model_results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "operational_threshold_report.json"
    md_path = REPORTS_DIR / "operational_threshold_report.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# Operational Threshold Selection",
        "",
        f"- Dataset: `{report['dataset']}` ({len(images)} images)",
        f"- Input: {IMAGE_SIZE}x{IMAGE_SIZE}; NMS IoU: {NMS_IOU_THRESHOLD:.2f}",
        "- Selection: maximum macro recall with macro precision >= 0.70; ties prefer precision, then higher confidence.",
        "- Metrics use IoU 0.50 for TP/FP/FN matching; dangerous recall is macro recall over configured critical classes.",
        "",
        "| Model | Threshold | Precision | Recall | Dangerous Recall | Critical Min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in model_results.values():
        selected = result["selected"]
        if selected is None:
            lines.append(f"| {result['model']} | no eligible threshold | - | - | - | - |")
        else:
            lines.append(
                f"| {result['model']} | {selected['confidence_threshold']:.2f} | {selected['precision']:.4f} | "
                f"{selected['recall']:.4f} | {selected['dangerous_class_recall']:.4f} | {selected['critical_min_recall']:.4f} |"
            )
    lines.extend(["", "## Selected operating metrics", "", "| Model | False negatives | False positives/image |", "|---|---:|---:|"])
    for result in model_results.values():
        selected = result["selected"]
        if selected is not None:
            lines.append(f"| {result['model']} | {selected['false_negatives']} | {selected['false_positives_per_image']:.4f} |")
    lines.extend(["", "Full metrics for all thresholds, including per-class values, are in the JSON report.", ""])
    md_path.write_text("\n".join(lines))

    manifest["operational_thresholds"] = manifest.get("operational_thresholds", {})
    manifest["operational_thresholds"].update({
        model_name: {
            "backend": result["backend"],
            "imgsz": IMAGE_SIZE,
            "artifact": result["artifact"],
            "confidence_threshold": result["selected"]["confidence_threshold"] if result["selected"] else None,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "selected_on": "rod_clean_v1/valid",
            "validation_images": len(images),
            "selection_rule": result["selection_rule"],
        }
        for model_name, result in model_results.items()
    })
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    for model_name, result in model_results.items():
        selected = result["selected"]
        if selected is None:
            raise RuntimeError(f"No threshold with precision >= 0.70 for {model_name}; not locking a threshold")
        key = "onnx_fp32_640" if model_name == "yolov12n" else "onnx_int8_640"
        artifact_entry = manifest["models"][model_name][key]
        artifact_entry["confidence_threshold"] = selected["confidence_threshold"]
        artifact_entry["nms_iou_threshold"] = NMS_IOU_THRESHOLD
        artifact_entry["operational_threshold_source"] = "rod_clean_v1/valid"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    for result in model_results.values():
        selected = result["selected"]
        print(
            f"{result['model']}: selected={selected['confidence_threshold'] if selected else None}, "
            f"precision={selected['precision'] if selected else None}, recall={selected['recall'] if selected else None}"
        )
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
