import logging
import threading
import unittest
from unittest.mock import patch

import numpy as np

from core.baseline_pipeline import BaselinePipeline, BaselinePolicy, CenterROI
from main import _try_start_tof
from modules.audio.audio_broker import AudioBroker
from modules.sensors.tof_vl53l1x import ToFReading, VL53L1XSensor
from modules.vision_local.onnx_detector import OnnxDetector


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [type("Input", (), {"name": "images"})()]

    def get_outputs(self):
        return [type("Output", (), {"name": "output"})()]

    def run(self, output_names, inputs):
        self.inputs = inputs
        return [self.output]


class FakeToFDevice:
    def __init__(self, readings):
        self.readings = iter(readings)

    def open(self):
        pass

    def start_ranging(self, mode):
        pass

    def get_distance(self):
        return next(self.readings)

    def stop_ranging(self):
        pass


def reading(distance_m, valid=True, monotonic_timestamp=1.0):
    return ToFReading(
        distance_m=distance_m if valid else None,
        timestamp=1.0,
        valid=valid,
        status="ok" if valid else "out_of_range",
        monotonic_timestamp=monotonic_timestamp,
    )


def detection(class_name, bbox, confidence=0.9, position="center"):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "position": position,
        "distance_m": None,
    }


class OnnxDetectorTests(unittest.TestCase):
    def test_decodes_yolov8_output_filters_and_suppresses_overlaps(self):
        output = np.zeros((1, 84, 3), dtype=np.float32)
        output[0, :4, 0] = [32, 32, 20, 20]
        output[0, 4, 0] = 0.9
        output[0, :4, 1] = [33, 33, 20, 20]
        output[0, 4, 1] = 0.8
        output[0, :4, 2] = [8, 32, 12, 12]
        output[0, 4 + 56, 2] = 0.75
        detector = OnnxDetector(
            "unused.onnx",
            input_size=64,
            confidence_threshold=0.35,
            session=FakeSession(output),
        )

        detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual([item["class_name"] for item in detections], ["person", "chair"])
        self.assertEqual([item["position"] for item in detections], ["center", "left"])
        self.assertTrue(all(item["distance_m"] is None for item in detections))
        self.assertTrue(all({"confidence", "bbox", "class_name"} <= item.keys()
                            for item in detections))

    def test_letterbox_is_rgb_nchw_and_normalized(self):
        detector = OnnxDetector(
            "unused.onnx",
            input_size=64,
            session=FakeSession(np.zeros((1, 84, 1), dtype=np.float32)),
        )
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[:, :, 0] = 255

        tensor, scale, pad_x, pad_y = detector._preprocess(frame)

        self.assertEqual(tensor.shape, (1, 3, 64, 64))
        self.assertAlmostEqual(scale, 0.8)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 16)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(float(tensor[0, 2, 20, 20]), 1.0)

    def test_decodes_end_to_end_nms_output(self):
        output = np.array([[[10, 12, 40, 50, 0.8, 0]]], dtype=np.float32)
        detector = OnnxDetector(
            "unused.onnx", input_size=64, session=FakeSession(output)
        )
        detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [10, 12, 40, 50])
        self.assertEqual(detections[0]["class_name"], "person")


class SensorAndPolicyTests(unittest.TestCase):
    def test_tof_start_failure_logs_warning_and_allows_startup(self):
        class MissingSensor:
            def start(self):
                raise RuntimeError("I2C bus not available")

        with self.assertLogs("blind_navigation", level=logging.WARNING) as captured:
            started = _try_start_tof(MissingSensor())

        self.assertFalse(started)
        self.assertIn("vision-only", captured.output[0])
        self.assertIn("Distance-based alerts are disabled", captured.output[0])

    def test_vl53l1x_reports_median_distance_and_invalid_status(self):
        sensor = VL53L1XSensor(
            device=FakeToFDevice([500, 2000, 600, 0]), median_window=3
        )
        sensor.start()

        first, second, third, invalid = (sensor.read() for _ in range(4))

        self.assertAlmostEqual(first.distance_m, 0.5)
        self.assertAlmostEqual(second.distance_m, 1.25)
        self.assertAlmostEqual(third.distance_m, 0.6)
        self.assertTrue(first.valid)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.status, "out_of_range")
        sensor.close()

    def test_distance_thresholds(self):
        thresholds = [
            (0.49, "Critical"),
            (0.5, "High"),
            (0.79, "High"),
            (0.8, "Moderate"),
            (1.19, "Moderate"),
            (1.2, "Low"),
            (1.5, "Low"),
            (1.51, "Clear"),
        ]
        for distance, expected in thresholds:
            with self.subTest(distance=distance):
                self.assertEqual(BaselinePolicy.severity_for(distance), expected)

    def test_only_center_roi_detection_receives_distance(self):
        roi = CenterROI(left=0.4, right=0.6)
        detections = [
            detection("chair", [2, 10, 18, 30], position="left"),
            detection("person", [42, 10, 58, 30], position="center"),
            detection("chair", [80, 10, 98, 30], position="right"),
        ]

        associated = roi.associate(detections, reading(0.4), frame_width=100)

        self.assertIsNone(associated[0]["distance_m"])
        self.assertEqual(associated[1]["distance_m"], 0.4)
        self.assertIsNone(associated[2]["distance_m"])
        with self.assertRaises(ValueError):
            CenterROI(left=0.3, right=0.5)

    def test_center_person_or_chair_is_selected_for_alert(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for class_name in ("person", "chair"):
            with self.subTest(class_name=class_name):
                output = [detection(class_name, [45, 10, 55, 40])]
                result = BaselinePipeline(None).process_frame(
                    frame, reading(0.6), detect=lambda _frame: output
                )
                self.assertEqual(
                    result.selected_detection["class_name"], class_name
                )
                self.assertEqual(result.selected_detection["distance_m"], 0.6)
                self.assertEqual(result.alert.severity, "High")

    def test_tof_alerts_without_yolo_detection_and_after_yolo_error(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = BaselinePipeline(detector=None).process_frame(frame, reading(0.4))
        self.assertEqual(result.alert.message, "Vật cản phía trước")

        side_result = BaselinePipeline(None).process_frame(
            frame,
            reading(0.4),
            detect=lambda _frame: [detection("chair", [5, 10, 25, 40], position="left")],
        )
        self.assertIsNone(side_result.detections[0]["distance_m"])
        self.assertEqual(side_result.alert.message, "Vật cản phía trước")

        class BrokenDetector:
            def detect(self, _frame):
                raise RuntimeError("inference unavailable")

        with self.assertLogs("core.baseline_pipeline", level=logging.ERROR):
            failed_result = BaselinePipeline(BrokenDetector()).process_frame(
                frame, reading(0.4)
            )
        self.assertEqual(failed_result.inference_error, "RuntimeError: inference unavailable")
        self.assertEqual(failed_result.alert.message, "Vật cản phía trước")

    def test_alert_cooldown_and_immediate_severity_escalation(self):
        policy = BaselinePolicy(cooldown_seconds=3.5)
        with patch("core.baseline_pipeline.time.monotonic", side_effect=[100, 101, 102]):
            _, first = policy.choose([], reading(0.7, monotonic_timestamp=100))
            _, repeated = policy.choose([], reading(0.7, monotonic_timestamp=101))
            _, escalated = policy.choose([], reading(0.4, monotonic_timestamp=102))

        self.assertEqual(first.severity, "High")
        self.assertIsNone(repeated)
        self.assertEqual(escalated.severity, "Critical")


class AudioBrokerTests(unittest.TestCase):
    def test_audio_is_spoken_on_a_worker_thread(self):
        completed = threading.Event()
        caller_thread = threading.get_ident()
        worker_threads = []

        def speaker(_message):
            worker_threads.append(threading.get_ident())
            completed.set()

        broker = AudioBroker(speaker=speaker)
        broker.start()
        broker.enqueue("warning")
        self.assertTrue(completed.wait(timeout=1))
        broker.close()

        self.assertNotEqual(worker_threads[0], caller_thread)

    def test_queue_does_not_replace_critical_with_low_priority_alert(self):
        speaking = threading.Event()
        release = threading.Event()
        critical_spoken = threading.Event()
        spoken = []

        def speaker(message):
            spoken.append(message)
            if message == "current":
                speaking.set()
                release.wait(timeout=1)
            if message == "critical":
                critical_spoken.set()

        broker = AudioBroker(speaker=speaker)
        broker.start()
        broker.enqueue("current")
        self.assertTrue(speaking.wait(timeout=1))
        broker.enqueue("critical", priority=4)
        broker.enqueue("low", priority=1)
        release.set()
        self.assertTrue(critical_spoken.wait(timeout=1))
        broker.close()

        self.assertEqual(spoken, ["current", "critical"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
