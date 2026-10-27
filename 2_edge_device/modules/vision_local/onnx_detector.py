from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .base_detector import BaseDetector

LOGGER = logging.getLogger(__name__)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class OnnxDetector(BaseDetector):
    """YOLOv8 COCO ONNX detector using ONNX Runtime's CPU provider."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        session: Any | None = None,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = COCO_CLASSES
        self.session = session
        self.input_name: str | None = None
        self.output_names: list[str] = []
        self.last_timings_ms = {
            "preprocessing": 0.0,
            "inference": 0.0,
            "postprocessing": 0.0,
        }
        if self.session is None:
            self.load_model()
        else:
            self._set_session_names()

    def load_model(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required; install onnxruntime on Raspberry Pi"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._set_session_names()
        LOGGER.info("Loaded ONNX model %s with CPUExecutionProvider", self.model_path)

    def _set_session_names(self) -> None:
        if self.session is None:
            raise RuntimeError("ONNX Runtime session was not initialized")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        detections, _ = self.detect_with_metrics(frame)
        return detections

    def detect_with_metrics(
        self, frame: np.ndarray
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image with 3 channels")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("frame dimensions must be non-zero")
        if self.session is None or self.input_name is None:
            raise RuntimeError("ONNX Runtime session is not loaded")

        started = time.perf_counter()
        tensor, scale, pad_x, pad_y = self._preprocess(frame)
        preprocessing_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        detections = self._postprocess(
            outputs[0], frame.shape[1], frame.shape[0], scale, pad_x, pad_y
        )
        postprocessing_ms = (time.perf_counter() - started) * 1000
        self.last_timings_ms = {
            "preprocessing": preprocessing_ms,
            "inference": inference_ms,
            "postprocessing": postprocessing_ms,
        }
        return detections, self.last_timings_ms.copy()

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        import cv2

        frame_height, frame_width = frame.shape[:2]
        scale = min(self.input_size / frame_width, self.input_size / frame_height)
        resized_width = int(round(frame_width * scale))
        resized_height = int(round(frame_height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height))

        left = int(round((self.input_size - resized_width) / 2 - 0.1))
        top = int(round((self.input_size - resized_height) / 2 - 0.1))
        right = self.input_size - resized_width - left
        bottom = self.input_size - resized_height - top
        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = letterboxed[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(tensor), scale, float(left), float(top)

    def _postprocess(
        self,
        output: np.ndarray,
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[dict[str, Any]]:
        rows = np.asarray(output, dtype=np.float32)
        rows = np.squeeze(rows)
        if rows.ndim == 1 and rows.size in (6, 84, 85):
            rows = rows.reshape(1, -1)
        if rows.ndim != 2:
            raise ValueError(f"Unsupported YOLO output shape: {np.shape(output)}")

        if rows.shape[0] == 6 and rows.shape[1] > 6:
            rows = rows.T
        if rows.shape[1] == 6:
            candidates = self._decode_nms_output(rows)
        else:
            if rows.shape[0] in (84, 85) or (
                rows.shape[0] <= 256 and rows.shape[1] > rows.shape[0]
            ):
                rows = rows.T
            candidates = self._decode_yolo_output(rows)

        candidates = self._nms(candidates)
        detections: list[dict[str, Any]] = []
        for class_id, confidence, x1, y1, x2, y2 in candidates:
            x1 = int(np.clip(round((x1 - pad_x) / scale), 0, frame_width))
            x2 = int(np.clip(round((x2 - pad_x) / scale), 0, frame_width))
            y1 = int(np.clip(round((y1 - pad_y) / scale), 0, frame_height))
            y2 = int(np.clip(round((y2 - pad_y) / scale), 0, frame_height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "confidence": float(confidence),
                    "bbox": [x1, y1, x2, y2],
                    "position": self._position((x1 + x2) / 2, frame_width),
                    "distance_m": None,
                }
            )
        return detections

    def _decode_nms_output(
        self, rows: np.ndarray
    ) -> list[tuple[int, float, float, float, float, float]]:
        candidates = []
        for x1, y1, x2, y2, confidence, class_id_value in rows:
            if not np.isfinite([x1, y1, x2, y2, confidence, class_id_value]).all():
                continue
            class_id = int(class_id_value)
            if (
                confidence < self.confidence_threshold
                or class_id < 0
                or class_id >= len(self.classes)
            ):
                continue
            candidates.append(
                (class_id, float(confidence), float(x1), float(y1), float(x2), float(y2))
            )
        return candidates

    def _decode_yolo_output(
        self, rows: np.ndarray
    ) -> list[tuple[int, float, float, float, float, float]]:
        candidates = []
        for row in rows:
            if row.size < 5 or not np.isfinite(row).all():
                continue
            x_center, y_center, width, height = row[:4]
            class_scores = row[4:]
            if class_scores.size == len(self.classes) + 1:
                class_scores = class_scores[1:] * class_scores[0]
            if class_scores.size > len(self.classes):
                class_scores = class_scores[: len(self.classes)]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.confidence_threshold:
                continue
            x1 = float(x_center - width / 2)
            y1 = float(y_center - height / 2)
            x2 = float(x_center + width / 2)
            y2 = float(y_center + height / 2)
            candidates.append((class_id, confidence, x1, y1, x2, y2))
        return candidates

    def _nms(
        self, candidates: list[tuple[int, float, float, float, float, float]]
    ) -> list[tuple[int, float, float, float, float, float]]:
        kept: list[tuple[int, float, float, float, float, float]] = []
        for class_id in sorted({item[0] for item in candidates}):
            class_candidates = sorted(
                (item for item in candidates if item[0] == class_id),
                key=lambda item: item[1],
                reverse=True,
            )
            while class_candidates:
                best = class_candidates.pop(0)
                kept.append(best)
                class_candidates = [
                    item
                    for item in class_candidates
                    if self._iou(best[2:], item[2:]) <= self.iou_threshold
                ]
        return kept

    @staticmethod
    def _iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = area_first + area_second - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _position(center_x: float, frame_width: int) -> str:
        fraction = center_x / frame_width
        if fraction < 1 / 3:
            return "left"
        if fraction > 2 / 3:
            return "right"
        return "center"
