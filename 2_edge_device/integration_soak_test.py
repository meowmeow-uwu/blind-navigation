#!/usr/bin/env python3
"""Sustained detector integration check over a looping video source."""

import argparse
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "best_model_inference", SCRIPT_DIR / "best_model_inference.py"
)
BEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BEST)


def temperature_c() -> float | None:
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 2)
    except (OSError, ValueError):
        return None


def throttle_state() -> dict:
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return {"available": False, "code": None}
        code = int(result.stdout.strip().split("=")[-1], 16)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"available": False, "code": None}
    return {
        "available": True,
        "code": hex(code),
        "undervoltage_current": bool(code & 0x1),
        "throttled_current": bool(code & 0x2),
        "undervoltage_since_boot": bool(code & 0x10000),
        "throttled_since_boot": bool(code & 0x20000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("yolov12n", "yolov26n"), required=True)
    parser.add_argument("--backend", choices=("onnx", "ncnn"), required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=BEST.DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    detector_args = argparse.Namespace(
        package_root=args.package_root,
        model=args.model,
        backend=args.backend,
        precision=args.precision,
        threads=args.threads,
        conf=args.conf,
        iou=args.iou,
    )
    infer, artifact, adapter = BEST.load_detector(detector_args)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video source: {args.source}")

    expected_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(expected_fps) or expected_fps <= 0:
        expected_fps = None
    frame_latencies = []
    cpu_samples = []
    empty_frames = 0
    current_empty_streak = 0
    maximum_empty_streak = 0
    frame_errors = 0
    decode_errors = 0
    detections_total = 0
    maximum_temperature = temperature_c()
    temperature_start = maximum_temperature
    throttling_start = throttle_state()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    psutil.cpu_percent(interval=None)
    started = time.monotonic()
    last_sample = started
    frames = 0
    stop_reason = None

    try:
        while time.monotonic() - started < args.duration_seconds:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    decode_errors += 1
                    stop_reason = "video_decode_failed_after_rewind"
                    break

            frame_started = time.perf_counter()
            try:
                detections = infer(frame)
                for item in detections:
                    assert set(item) == {"class_id", "class_name", "confidence", "bbox_xyxy"}
                    assert isinstance(item["class_id"], int)
                    assert isinstance(item["class_name"], str)
                    assert math.isfinite(item["confidence"])
                    assert 0.0 <= item["confidence"] <= 1.0
                    x1, y1, x2, y2 = item["bbox_xyxy"]
                    assert all(math.isfinite(value) for value in (x1, y1, x2, y2))
                    assert 0 <= x1 < x2 <= frame.shape[1]
                    assert 0 <= y1 < y2 <= frame.shape[0]
                detections_total += len(detections)
                if detections:
                    current_empty_streak = 0
                else:
                    empty_frames += 1
                    current_empty_streak += 1
                    maximum_empty_streak = max(maximum_empty_streak, current_empty_streak)
            except Exception as error:
                frame_errors += 1
                current_empty_streak = 0
                print(f"Frame {frames}: {type(error).__name__}: {error}", flush=True)
            frame_latencies.append((time.perf_counter() - frame_started) * 1000)
            frames += 1

            now = time.monotonic()
            if now - last_sample >= 1.0:
                cpu_samples.append(psutil.cpu_percent(interval=None))
                peak_rss = max(peak_rss, process.memory_info().rss)
                temp = temperature_c()
                if temp is not None:
                    maximum_temperature = max(maximum_temperature or temp, temp)
                    if temp >= 82:
                        stop_reason = "temperature_limit_82c"
                        break
                last_sample = now
                if frames % 1000 == 0:
                    print(
                        f"elapsed={now-started:.0f}s frames={frames} errors={frame_errors} "
                        f"temp={temp}C",
                        flush=True,
                    )
    finally:
        capture.release()

    elapsed = time.monotonic() - started
    throttling_end = throttle_state()
    temperature_end = temperature_c()
    if temperature_end is not None:
        maximum_temperature = max(maximum_temperature or temperature_end, temperature_end)
    fps = frames / elapsed if elapsed > 0 else 0.0
    current_power_event = any(
        state.get(field, False)
        for state in (throttling_start, throttling_end)
        for field in ("undervoltage_current", "throttled_current")
    )
    history_power_event = any(
        state.get(field, False)
        for state in (throttling_start, throttling_end)
        for field in ("undervoltage_since_boot", "throttled_since_boot")
    )
    completed_duration = elapsed >= args.duration_seconds
    abnormal_empty = frames > 0 and empty_frames == frames
    stable = (
        completed_duration
        and frame_errors == 0
        and decode_errors == 0
        and stop_reason is None
        and not current_power_event
        and not abnormal_empty
    )

    def percentile(value: float) -> float | None:
        return round(float(np.percentile(frame_latencies, value)), 3) if frame_latencies else None

    report = {
        "model": args.model,
        "runtime": f"{args.backend}_{args.precision}",
        "artifact": str(artifact),
        "output_adapter": adapter,
        "source_video": str(args.source),
        "source_fps": expected_fps,
        "thresholds": {"confidence": args.conf, "iou": args.iou},
        "duration_target_seconds": args.duration_seconds,
        "duration_actual_seconds": round(elapsed, 2),
        "frames_processed": frames,
        "processing_fps": round(fps, 3),
        "frame_latency_ms": {
            "p50": percentile(50), "p95": percentile(95), "p99": percentile(99)
        },
        "peak_ram_mb_sampled": round(peak_rss / (1024 * 1024), 2),
        "cpu_mean_percent_sampled": round(float(np.mean(cpu_samples)), 2) if cpu_samples else None,
        "cpu_p95_percent_sampled": round(float(np.percentile(cpu_samples, 95)), 2) if cpu_samples else None,
        "temperature_start_c": temperature_start,
        "temperature_end_c": temperature_end,
        "temperature_max_sampled_c": maximum_temperature,
        "throttling_start": throttling_start,
        "throttling_end": throttling_end,
        "current_throttling_or_undervoltage_observed": current_power_event,
        "throttling_or_undervoltage_history_flagged": history_power_event,
        "crash_count": 0 if stop_reason is None else 1,
        "frame_error_count": frame_errors,
        "decode_error_count": decode_errors,
        "empty_detection_frames": empty_frames,
        "empty_detection_fraction": round(empty_frames / frames, 4) if frames else None,
        "max_consecutive_empty_frames": maximum_empty_streak,
        "abnormal_empty_detection": "all_frames_empty" if abnormal_empty else "not_indicated_by_all_empty_rule",
        "detections_total": detections_total,
        "completed_target_duration": completed_duration,
        "detector_soak_stable": stable,
        "tracking_integration": "not_tested_no_tracker_implementation_in_2_edge_device",
        "stop_reason": stop_reason,
        "empty_detection_note": "Empty frames are counted, but cannot be judged semantically anomalous without video ground truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not completed_duration or stop_reason or frame_errors or current_power_event:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
