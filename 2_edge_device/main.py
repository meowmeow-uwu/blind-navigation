from __future__ import annotations

import argparse
import json
import logging
import pathlib
import threading
import time

import cv2

from core.baseline_pipeline import BaselinePipeline, BaselinePolicy, CenterROI
from modules.audio.audio_broker import AudioBroker
from modules.sensors.tof_vl53l1x import ToFReading, VL53L1XSensor
from modules.vision_local.onnx_detector import OnnxDetector

LOGGER = logging.getLogger("blind_navigation")


class SystemMetrics:
    def __init__(self) -> None:
        self._previous_cpu: tuple[int, int] | None = None

    def sample(self) -> dict[str, float | None]:
        return {
            "cpu_percent": self._cpu_percent(),
            "ram_mb": self._ram_mb(),
            "temperature_c": self._temperature_c(),
        }

    def _cpu_percent(self) -> float | None:
        try:
            values = pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            total = sum(map(int, values))
            idle = int(values[3]) + int(values[4])
        except (OSError, IndexError, ValueError):
            return None
        previous = self._previous_cpu
        self._previous_cpu = (total, idle)
        if previous is None:
            return None
        total_delta = total - previous[0]
        idle_delta = idle - previous[1]
        return 100 * (total_delta - idle_delta) / total_delta if total_delta else 0.0

    @staticmethod
    def _ram_mb() -> float | None:
        try:
            for line in pathlib.Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
        except (OSError, IndexError, ValueError):
            return None
        return None

    @staticmethod
    def _temperature_c() -> float | None:
        for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                return float(path.read_text().strip()) / 1000
            except (OSError, ValueError):
                continue
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi 4B navigation baseline")
    parser.add_argument(
        "--model",
        default="modules/vision_local/models/yolov8n.onnx",
        help="Path to the COCO YOLOv8n ONNX model",
    )
    parser.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds")
    parser.add_argument("--roi-left", type=float, default=0.40)
    parser.add_argument("--roi-right", type=float, default=0.60)
    parser.add_argument("--cooldown", type=float, default=3.5)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must not be negative")
    if args.input_size <= 0:
        parser.error("--input-size must be positive")
    try:
        CenterROI(args.roi_left, args.roi_right)
        BaselinePolicy(args.cooldown)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _sensor_loop(
    sensor: VL53L1XSensor,
    latest: list[ToFReading],
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    next_sensor_error_log = 0.0
    while not stop.is_set():
        try:
            value = sensor.read()
        except Exception:
            now = time.monotonic()
            if now >= next_sensor_error_log:
                LOGGER.exception("Unexpected ToF read error; marking sample invalid")
                next_sensor_error_log = now + 5
            value = ToFReading(
                None, time.time(), False, "read_error", time.monotonic()
            )
        with lock:
            latest[0] = value
        stop.wait(0.05)


def _try_start_tof(sensor: VL53L1XSensor) -> bool:
    try:
        sensor.start()
    except Exception:
        LOGGER.warning(
            "WARNING: VL53L1X unavailable; continuing in degraded vision-only "
            "mode. Distance-based alerts are disabled; do not use as a safety "
            "navigation aid until ToF is working.",
            exc_info=True,
        )
        return False
    return True


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sensor = VL53L1XSensor()
    sensor_available = _try_start_tof(sensor)

    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not capture.isOpened():
        if sensor_available:
            sensor.close()
        LOGGER.error("Unable to open camera index %d", args.camera)
        return 2
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = None
    try:
        detector = OnnxDetector(args.model, input_size=args.input_size)
    except Exception:
        LOGGER.exception("Unable to load YOLO; continuing with ToF fail-safe only")

    pipeline = BaselinePipeline(
        detector,
        roi=CenterROI(args.roi_left, args.roi_right),
        policy=BaselinePolicy(args.cooldown),
    )

    latest = [
        ToFReading(
            None,
            time.time(),
            False,
            "unavailable" if not sensor_available else "not_started",
            time.monotonic(),
        )
    ]
    sensor_lock = threading.Lock()
    stop = threading.Event()
    sensor_thread = None
    if sensor_available:
        sensor_thread = threading.Thread(
            target=_sensor_loop,
            args=(sensor, latest, sensor_lock, stop),
            name="tof-reader",
            daemon=True,
        )
    audio = AudioBroker()
    if sensor_thread is not None:
        sensor_thread.start()
    audio.start()

    metrics = SystemMetrics()
    start_time = time.monotonic()
    interval_start = start_time
    frame_count = 0
    interval_frames = 0
    last_detector_timings = {
        "preprocessing": None,
        "inference": None,
        "postprocessing": None,
    }
    LOGGER.info(
        "Baseline started model=%s tof=%s roi=[%.2f, %.2f] cooldown=%.1fs",
        args.model,
        "available" if sensor_available else "unavailable (vision-only)",
        args.roi_left,
        args.roi_right,
        args.cooldown,
    )

    try:
        while True:
            if args.duration > 0 and time.monotonic() - start_time >= args.duration:
                break
            ok, frame = capture.read()
            if not ok:
                LOGGER.error("Camera frame read failed")
                time.sleep(0.1)
                continue

            with sensor_lock:
                sensor_reading = latest[0]
            result = pipeline.process_frame(frame, sensor_reading)
            last_detector_timings = {
                "preprocessing": None,
                "inference": None,
                "postprocessing": None,
            }
            if detector is not None and result.inference_error is None:
                last_detector_timings = detector.last_timings_ms
            if result.alert is not None:
                audio.enqueue(
                    result.alert.message,
                    priority={
                        "Low": 1,
                        "Moderate": 2,
                        "High": 3,
                        "Critical": 4,
                    }[result.alert.severity],
                )
                LOGGER.info(
                    "Alert emitted severity=%s message=%s distance_m=%.2f "
                    "sensor_to_alert_ms=%.2f selected=%s",
                    result.alert.severity,
                    result.alert.message,
                    result.alert.distance_m,
                    result.alert.sensor_to_alert_latency_ms,
                    json.dumps(result.selected_detection, ensure_ascii=False)
                    if result.selected_detection
                    else None,
                )

            frame_count += 1
            interval_frames += 1
            now = time.monotonic()
            if now - interval_start >= 1:
                system_metrics = metrics.sample()
                fps = interval_frames / (now - interval_start)
                selected = result.selected_detection
                LOGGER.info(
                    "FPS=%.2f preprocess_ms=%s inference_ms=%s postprocess_ms=%s "
                    "sensor_to_alert_ms=%s CPU_pct=%s RAM_MB=%s temp_C=%s "
                    "distance_m=%s tof_status=%s detections=%s selected=%s "
                    "alert=%s frames=%d",
                    fps,
                    _format_metric(last_detector_timings["preprocessing"]),
                    _format_metric(last_detector_timings["inference"]),
                    _format_metric(last_detector_timings["postprocessing"]),
                    _format_metric(
                        result.alert.sensor_to_alert_latency_ms
                        if result.alert is not None
                        else None
                    ),
                    _format_metric(system_metrics["cpu_percent"]),
                    _format_metric(system_metrics["ram_mb"]),
                    _format_metric(system_metrics["temperature_c"]),
                    _format_metric(sensor_reading.distance_m),
                    sensor_reading.status,
                    json.dumps(result.detections, ensure_ascii=False),
                    json.dumps(selected, ensure_ascii=False) if selected else None,
                    result.alert.message if result.alert is not None else None,
                    frame_count,
                )
                interval_frames = 0
                interval_start = now
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested")
    finally:
        stop.set()
        if sensor_thread is not None:
            sensor_thread.join(timeout=2)
        audio.close()
        capture.release()
        if sensor_available:
            sensor.close()
        LOGGER.info("Baseline stopped after %d frames", frame_count)
    return 0


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())