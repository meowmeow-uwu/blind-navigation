#!/usr/bin/env python3
"""Evaluate cached NCNN FP32 320x320 models directly on rod_clean_v1/test."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import ncnn
import numpy as np
import torch
import yaml
from ultralytics.data.augment import LetterBox
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DEV_ROOT = SCRIPT_DIR.parent
REPO_ROOT = MODEL_DEV_ROOT.parent.parent
WEIGHTS_DIR = MODEL_DEV_ROOT / "weights"
DATASET_DIR = MODEL_DEV_ROOT / "datasets" / "processed" / "rod_clean_v1"
SPLIT = "test"
TEST_IMAGES_DIR = DATASET_DIR / SPLIT / "images"
TEST_LABELS_DIR = DATASET_DIR / SPLIT / "labels"
REPORTS_DIR = MODEL_DEV_ROOT / "reports" / "pipeline_run"

ROD_MODELS = ("yolov8n", "yolov9t", "yolov12n", "yolov26n")
ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]
IMAGE_SIZE = 320
CONF_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_test_images() -> list[Path]:
    return sorted(
        path for path in TEST_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def read_targets(label_path: Path, original_shape: tuple[int, int], pad: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not label_path.is_file() or not label_path.stat().st_size:
        return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)

    rows = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if rows.shape[1] != 5:
        raise ValueError(f"Expected YOLO labels with 5 columns in {label_path}, got {rows.shape[1]}")
    height, width = original_shape
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    boxes = rows[:, 1:5].copy()
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    boxes *= scale
    boxes[:, [0, 2]] += pad[0]
    boxes[:, [1, 3]] += pad[1]
    return torch.from_numpy(rows[:, 0].copy()), torch.from_numpy(boxes)


def letterbox_image(image: np.ndarray, letterbox: LetterBox) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    resized = letterbox(image=image)
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    pad_x = (IMAGE_SIZE - round(width * scale)) / 2
    pad_y = (IMAGE_SIZE - round(height * scale)) / 2
    left = round(pad_x - 0.1)
    top = round(pad_y - 0.1)
    return resized, (left, top)


def evaluate_model(model_name: str, images: list[Path], critical_classes: set[str]) -> dict[str, Any]:
    model_dir_name = f"{model_name}_ncnn_fp32" if IMAGE_SIZE == 320 else f"{model_name}_rod_{IMAGE_SIZE}_ncnn_fp32"
    model_dir = WEIGHTS_DIR / "ncnn" / model_dir_name
    param_path = model_dir / "model.ncnn.param"
    bin_path = model_dir / "model.ncnn.bin"
    if not param_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"NCNN model files are missing for {model_name}: {model_dir}")

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    if net.load_param(str(param_path)) != 0 or net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"Could not load NCNN model: {model_dir}")

    letterbox = LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False, stride=32)
    stats: dict[str, list[np.ndarray]] = {key: [] for key in ("tp", "conf", "pred_cls", "target_cls")}
    target_counts = np.zeros(len(ROD_CLASSES), dtype=np.int64)
    matching = SimpleNamespace(iouv=IOU_THRESHOLDS)
    started = time.time()

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read test image: {image_path}")
        original_shape = image.shape[:2]
        boxed, pad = letterbox_image(image, letterbox)
        rgb_chw = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0

        with net.create_extractor() as extractor:
            if extractor.input("in0", ncnn.Mat(rgb_chw)) != 0:
                raise RuntimeError(f"NCNN input failed for {image_path.name}")
            status, output = extractor.extract("out0")
        if status != 0:
            raise RuntimeError(f"NCNN inference failed for {image_path.name}: status={status}")

        raw_output = np.asarray(output, dtype=np.float32)
        if raw_output.shape != (len(ROD_CLASSES) + 4, (IMAGE_SIZE // 32) ** 2 * 21):
            raise ValueError(f"Unexpected NCNN output shape for {model_name}: {raw_output.shape}")
        prediction = non_max_suppression(
            torch.from_numpy(raw_output.copy()[None]),
            conf_thres=CONF_THRESHOLD,
            iou_thres=NMS_IOU_THRESHOLD,
            multi_label=True,
            max_det=MAX_DETECTIONS,
        )[0]
        pred = {
            "bboxes": prediction[:, :4],
            "conf": prediction[:, 4],
            "cls": prediction[:, 5],
            "extra": prediction[:, 6:],
        }

        target_cls, target_boxes = read_targets(
            TEST_LABELS_DIR / f"{image_path.stem}.txt", original_shape, pad
        )
        if target_cls.numel() and (
            (target_cls < 0).any() or (target_cls >= len(ROD_CLASSES)).any()
        ):
            raise ValueError(f"Out-of-range class id in {image_path.stem}.txt")
        if target_cls.numel():
            target_counts += np.bincount(target_cls.numpy().astype(int), minlength=len(ROD_CLASSES))

        if target_cls.numel() == 0 or pred["cls"].numel() == 0:
            correct = np.zeros((pred["cls"].shape[0], len(IOU_THRESHOLDS)), dtype=bool)
        else:
            overlaps = box_iou(target_boxes, pred["bboxes"])
            correct = BaseValidator.match_predictions(
                matching, pred["cls"], target_cls, overlaps
            ).cpu().numpy()
        stats["tp"].append(correct)
        stats["conf"].append(pred["conf"].cpu().numpy())
        stats["pred_cls"].append(pred["cls"].cpu().numpy())
        stats["target_cls"].append(target_cls.numpy())

        if index % 200 == 0 or index == len(images):
            print(f"  [{model_name}] {index}/{len(images)} images", flush=True)

    net = None
    concatenated = {
        key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        for key, values in stats.items()
    }
    if concatenated["tp"].size == 0:
        concatenated["tp"] = np.empty((0, len(IOU_THRESHOLDS)), dtype=bool)
    results = ap_per_class(
        concatenated["tp"],
        concatenated["conf"],
        concatenated["pred_cls"],
        concatenated["target_cls"],
        names=dict(enumerate(ROD_CLASSES)),
        prefix="Box",
    )
    true_positive, false_positive, precision, recall, _, ap, classes = results[:7]
    del true_positive, false_positive

    per_class_recall: dict[str, float | None] = {name: None for name in ROD_CLASSES}
    per_class_false_negatives: dict[str, int] = {}
    for index, class_id in enumerate(classes):
        class_name = ROD_CLASSES[int(class_id)]
        class_recall = float(recall[index])
        per_class_recall[class_name] = round(class_recall, 6)
        estimated_tp = round(class_recall * int(target_counts[class_id]))
        per_class_false_negatives[class_name] = max(0, int(target_counts[class_id]) - estimated_tp)

    critical_recalls = [
        value for name, value in per_class_recall.items()
        if name in critical_classes and value is not None
    ]
    critical_min_recall = min(critical_recalls) if critical_recalls else None
    return {
        "artifact": f"weights/ncnn/{model_dir_name}/model.ncnn.param + model.ncnn.bin",
        "sha256": {"param": sha256_file(param_path), "bin": sha256_file(bin_path)},
        "imgsz": IMAGE_SIZE,
        "runtime": "ncnn",
        "precision": "fp32",
        "evaluator": "Ultralytics DetectionValidator NMS, match_predictions, and ap_per_class",
        "confidence_threshold": CONF_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "mAP50": round(float(ap[:, 0].mean()), 6) if len(ap) else 0.0,
        "mAP50_95": round(float(ap.mean()), 6) if len(ap) else 0.0,
        "precision": round(float(precision.mean()), 6) if len(precision) else 0.0,
        "recall": round(float(recall.mean()), 6) if len(recall) else 0.0,
        "per_class_recall": per_class_recall,
        "per_class_false_negatives": per_class_false_negatives,
        "false_negatives": sum(per_class_false_negatives.values()),
        "false_negative_definition": "ground-truth count minus round(class recall at Ultralytics selected F1 confidence times ground-truth count)",
        "dangerous_class_macro_recall": round(float(np.mean(critical_recalls)), 6) if critical_recalls else None,
        "critical_min_recall": round(float(critical_min_recall), 6) if critical_min_recall is not None else None,
        "dangerous_classes": sorted(critical_classes),
        "target_instances_per_class": {
            name: int(target_counts[index]) for index, name in enumerate(ROD_CLASSES)
        },
        "test_images": len(images),
        "eval_duration_sec": round(time.time() - started, 2),
        "accuracy_passed": len(images) == 1615 and bool(len(ap)),
    }


def main() -> None:
    global IMAGE_SIZE, SPLIT, TEST_IMAGES_DIR, TEST_LABELS_DIR
    parser = argparse.ArgumentParser(description="Evaluate cached native NCNN model artifacts")
    parser.add_argument("--imgsz", type=int, choices=(320, 640), default=320)
    parser.add_argument("--split", choices=("test",), default="test")
    args = parser.parse_args()
    IMAGE_SIZE = args.imgsz
    SPLIT = args.split
    TEST_IMAGES_DIR = DATASET_DIR / SPLIT / "images"
    TEST_LABELS_DIR = DATASET_DIR / SPLIT / "labels"

    images = list_test_images()
    if len(images) != 1615:
        raise RuntimeError(f"Expected 1615 test images, found {len(images)} in {TEST_IMAGES_DIR}")
    eval_yaml = DATASET_DIR / "eval_clean_test.yaml"
    if not eval_yaml.is_file():
        raise FileNotFoundError(f"Canonical test evaluator config is missing: {eval_yaml}")
    critical_classes = set(
        (yaml.safe_load((MODEL_DEV_ROOT / "configs" / "danger_classes.yaml").read_text())
         or {}).get("groups", {}).get("critical", {}).get("classes", [])
    )
    if not critical_classes:
        raise ValueError("No dangerous/critical classes configured")

    print(f"Direct NCNN FP32 evaluation: {len(images)} test images at {IMAGE_SIZE}x{IMAGE_SIZE}")
    models = {}
    for model_name in ROD_MODELS:
        print(f"\nEvaluating {model_name} NCNN FP32 {IMAGE_SIZE} with ncnn.Net", flush=True)
        models[model_name] = evaluate_model(model_name, images, critical_classes)
        print(
            f"  mAP50={models[model_name]['mAP50']:.4f} "
            f"mAP50-95={models[model_name]['mAP50_95']:.4f} "
            f"P={models[model_name]['precision']:.4f} R={models[model_name]['recall']:.4f}",
            flush=True,
        )

    manifest_path = None
    manifest = None
    if IMAGE_SIZE == 640:
        reference_path = REPORTS_DIR / "onnx_fp32_640_accuracy_report.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"ONNX FP32 640 reference report missing: {reference_path}")
        reference = json.loads(reference_path.read_text())["models"]
        manifest_path = WEIGHTS_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for model_name, result in models.items():
            onnx_metrics = reference[model_name]["accuracy"]
            deltas = {
                key: round(result[key] - onnx_metrics[key], 6)
                for key in ("mAP50", "mAP50_95", "recall")
            }
            gates = {key: abs(value) <= 0.005 for key, value in deltas.items()}
            passed = all(gates.values())
            result["comparison_vs_onnx_fp32_640"] = {
                "metrics": {key: onnx_metrics[key] for key in ("mAP50", "mAP50_95", "recall")},
                "deltas_ncnn_minus_onnx": deltas,
                "gate_tolerances": {"mAP50": 0.005, "mAP50_95": 0.005, "recall": 0.005},
                "gate_passed": gates,
            }
            result["accuracy_status"] = "verified" if passed else "failed_gate"
            result["accuracy_passed"] = passed
            manifest_entry = manifest["models"][model_name]["ncnn_fp32_640"]
            manifest_entry["accuracy"] = result
            manifest_entry["accuracy_status"] = result["accuracy_status"]
            manifest_entry["accuracy_passed"] = passed
            manifest_entry["accuracy_note"] = (
                "Native NCNN FP32 full-test evaluation passed all gates against ONNX FP32 640."
                if passed else "Native NCNN FP32 full-test evaluation did not pass all ONNX comparison gates."
            )
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

    report_path = REPORTS_DIR / f"ncnn_fp32_{IMAGE_SIZE}_accuracy_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imgsz": IMAGE_SIZE,
        "dataset": f"rod_clean_v1/{SPLIT}",
        "test_images": len(images),
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "confidence_threshold": CONF_THRESHOLD,
        "preprocessing": {
            "letterbox": f"Ultralytics LetterBox, {IMAGE_SIZE}x{IMAGE_SIZE}, fill=114, aspect ratio preserved",
            "color_order": "RGB",
            "normalization": "divide_by_255",
        },
        "models": models,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nNCNN full-test report: {report_path}")

    if IMAGE_SIZE == 640:
        lines = [
            "# Native NCNN FP32 640 Accuracy Verification", "",
            "- Dataset: `rod_clean_v1/test` (1,615 images)",
            "- Runtime: direct `ncnn.Net()` `.param/.bin` inference; Ultralytics NMS/matching/AP evaluator",
            "- Preprocessing: letterbox 640, RGB, divide by 255; confidence 0.001; NMS IoU 0.7", "",
            "| Model | mAP50 | mAP50-95 | Precision | Recall | Dangerous recall | Critical min | FN | ΔmAP50 | ΔmAP50-95 | ΔRecall | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for model_name, result in models.items():
            delta = result["comparison_vs_onnx_fp32_640"]["deltas_ncnn_minus_onnx"]
            lines.append(
                f"| {model_name} | {result['mAP50']:.4f} | {result['mAP50_95']:.4f} | "
                f"{result['precision']:.4f} | {result['recall']:.4f} | {result['dangerous_class_macro_recall']:.4f} | "
                f"{result['critical_min_recall']:.4f} | {result['false_negatives']} | "
                f"{delta['mAP50']:+.4f} | {delta['mAP50_95']:+.4f} | {delta['recall']:+.4f} | "
                f"{'PASS' if result['accuracy_passed'] else 'FAIL'} |"
            )
        lines.extend(["", "Per-class recall, class counts, hashes, and exact gate results are recorded in the JSON report.", ""])
        md_path = REPORTS_DIR / "ncnn_fp32_640_accuracy_report.md"
        md_path.write_text("\n".join(lines))
        if manifest_path is not None and manifest is not None:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"NCNN accuracy Markdown report: {md_path}")
        print("NCNN 640 accuracy_status updated per model based on the stated comparison gates.")


if __name__ == "__main__":
    main()
