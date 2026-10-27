from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from modules.sensors.tof_vl53l1x import ToFReading

LOGGER = logging.getLogger(__name__)

SEVERITY_RANK = {
    "Clear": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Critical": 4,
}


@dataclass(frozen=True)
class Alert:
    severity: str
    message: str
    alert_key: str
    distance_m: float
    sensor_to_alert_latency_ms: float
    detection: dict[str, Any] | None


@dataclass(frozen=True)
class PipelineResult:
    detections: list[dict[str, Any]]
    selected_detection: dict[str, Any] | None
    alert: Alert | None
    inference_error: str | None


class CenterROI:
    def __init__(self, left: float = 0.40, right: float = 0.60) -> None:
        if not 1 / 3 <= left < right <= 2 / 3:
            raise ValueError("Center ROI must remain within the center image lane")
        self.left = left
        self.right = right

    def contains(self, bbox: list[int], frame_width: int) -> bool:
        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        center_fraction = ((bbox[0] + bbox[2]) / 2) / frame_width
        return self.left <= center_fraction <= self.right

    def associate(
        self,
        detections: list[dict[str, Any]],
        reading: ToFReading,
        frame_width: int,
    ) -> list[dict[str, Any]]:
        associated = []
        for detection in detections:
            copy = dict(detection)
            copy["distance_m"] = None
            associated.append(copy)

        if not reading.valid or reading.distance_m is None:
            return associated

        central = [
            detection
            for detection in associated
            if self.contains(detection["bbox"], frame_width)
            and detection.get("position") == "center"
        ]
        if central:
            closest_to_center = min(
                central,
                key=lambda item: (
                    abs(
                        (item["bbox"][0] + item["bbox"][2]) / 2 / frame_width
                        - 0.5
                    ),
                    -item["confidence"],
                ),
            )
            closest_to_center["distance_m"] = reading.distance_m
        return associated


class BaselinePolicy:
    def __init__(self, cooldown_seconds: float = 3.5) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: dict[str, tuple[float, str]] = {}

    @staticmethod
    def severity_for(distance_m: float) -> str:
        if distance_m < 0.5:
            return "Critical"
        if distance_m < 0.8:
            return "High"
        if distance_m < 1.2:
            return "Moderate"
        if distance_m <= 1.5:
            return "Low"
        return "Clear"

    def choose(
        self, detections: list[dict[str, Any]], reading: ToFReading
    ) -> tuple[dict[str, Any] | None, Alert | None]:
        if not reading.valid or reading.distance_m is None:
            return None, None

        severity = self.severity_for(reading.distance_m)
        if severity == "Clear":
            return None, None

        matched = [
            item
            for item in detections
            if item.get("distance_m") is not None
            and item.get("position") == "center"
        ]
        selected = max(
            matched,
            key=lambda item: (
                SEVERITY_RANK[severity],
                item.get("position") == "center",
                item.get("confidence", 0.0),
            ),
            default=None,
        )
        if selected is None:
            message = "Vật cản phía trước"
            alert_key = "front_obstacle"
        else:
            class_name = selected.get("class_name", "Vật cản")
            message = f"{severity}. {class_name} phía trước"
            alert_key = f"{class_name}:center"

        now = time.monotonic()
        previous = self._last_alert.get(alert_key)
        if previous is not None:
            last_time, last_severity = previous
            escalated = SEVERITY_RANK[severity] > SEVERITY_RANK[last_severity]
            if not escalated and now - last_time < self.cooldown_seconds:
                return selected, None

        self._last_alert[alert_key] = (now, severity)
        latency_ms = max(0.0, now - reading.monotonic_timestamp) * 1000
        return selected, Alert(
            severity=severity,
            message=message,
            alert_key=alert_key,
            distance_m=reading.distance_m,
            sensor_to_alert_latency_ms=latency_ms,
            detection=selected,
        )


class BaselinePipeline:
    def __init__(
        self,
        detector: Any | None,
        roi: CenterROI | None = None,
        policy: BaselinePolicy | None = None,
    ) -> None:
        self.detector = detector
        self.roi = roi or CenterROI()
        self.policy = policy or BaselinePolicy()
        self._next_inference_error_log = 0.0

    def process_frame(
        self,
        frame: Any,
        reading: ToFReading,
        detect: Callable[[Any], list[dict[str, Any]]] | None = None,
    ) -> PipelineResult:
        detections: list[dict[str, Any]] = []
        inference_error = None
        detector_fn = detect
        if detector_fn is None and self.detector is not None:
            detector_fn = self.detector.detect

        if detector_fn is not None:
            try:
                detections = detector_fn(frame)
            except Exception as exc:
                inference_error = f"{type(exc).__name__}: {exc}"
                now = time.monotonic()
                if now >= self._next_inference_error_log:
                    LOGGER.exception(
                        "YOLO inference failed; continuing with ToF fail-safe"
                    )
                    self._next_inference_error_log = now + 5

        associated = self.roi.associate(detections, reading, frame.shape[1])
        selected, alert = self.policy.choose(associated, reading)
        return PipelineResult(associated, selected, alert, inference_error)
