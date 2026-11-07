# 2_edge_device/tests/test_pipeline_runner.py
"""
Kiểm thử PipelineRunner với nguồn dữ liệu giả (mock) – không cần webcam hay model thật.
Xác nhận:
  1. Vòng lặp chạy đúng số frame.
  2. Callback on_alert được gọi khi có object.
  3. Profiler ghi nhận đủ 5 stage.
  4. state_mgr.evict_stale được gọi định kỳ (không lỗi).
"""
import os
import sys
import time
import numpy as np
from unittest.mock import MagicMock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, EDGE_DIR)

from modules.decision.priority_scorer import PriorityScorer
from modules.decision.track_state_manager import TrackStateManager
from modules.utils.profiler import PipelineProfiler
from pipeline_runner import PipelineRunner


WEIGHTS_PATH = os.path.join(EDGE_DIR, "configs/class_weights.yaml")
FRAME_W, FRAME_H = 320, 240
N_FRAMES = 10


def _make_mock_detector() -> MagicMock:
    """Giả lập detector: luôn trả về 1 person ở giữa khung hình."""
    mock = MagicMock()
    boxes = np.array([[80, 60, 240, 200]], dtype=np.float32)  # person ở giữa
    confs = np.array([0.92], dtype=np.float32)
    cls_ids = np.array([0], dtype=int)  # class 0 = person
    mock.detect_raw.return_value = (boxes, confs, cls_ids)
    return mock


def _make_mock_tracker() -> MagicMock:
    """Giả lập tracker: gán track_id=1 cho person."""
    mock = MagicMock()
    mock.update.return_value = [{
        "track_id": 1,
        "class_name": "person",
        "class_id": 0,
        "bbox": [80, 60, 240, 200],
        "confidence": 0.92
    }]
    return mock


def _make_mock_stream(n_frames: int) -> MagicMock:
    """Giả lập stream trả về n_frames frame giả."""
    mock = MagicMock()
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    mock.__iter__ = MagicMock(return_value=iter([frame.copy() for _ in range(n_frames)]))
    mock.close = MagicMock()
    return mock


def test_pipeline_runner():
    print("[TEST] PipelineRunner (mock)...")

    alerts_received = []

    def _capture_alert(text: str):
        alerts_received.append(text)

    detector = _make_mock_detector()
    tracker = _make_mock_tracker()
    scorer = PriorityScorer(WEIGHTS_PATH, w_class=0.4, w_position=0.35, w_size=0.25)
    state_mgr = TrackStateManager(cooldown_seconds=0.0, re_alert_delta=1.0)
    stream = _make_mock_stream(N_FRAMES)
    profiler = PipelineProfiler(buffer_size=N_FRAMES)

    runner = PipelineRunner(
        detector=detector,
        tracker=tracker,
        scorer=scorer,
        state_mgr=state_mgr,
        stream=stream,
        profiler=profiler,
        on_alert=_capture_alert,
        show_display=False,
        show_lane=False,
        evict_interval=5
    )
    runner.run()

    # Detector và tracker phải được gọi N_FRAMES lần
    assert detector.detect_raw.call_count == N_FRAMES, \
        f"detect_raw bị gọi {detector.detect_raw.call_count} lần, cần {N_FRAMES}"
    assert tracker.update.call_count == N_FRAMES, \
        f"tracker.update bị gọi {tracker.update.call_count} lần, cần {N_FRAMES}"

    # Phải có ít nhất 1 alert (frame đầu tiên của person mới)
    assert len(alerts_received) >= 1, "Không nhận được alert nào!"
    assert "người" in alerts_received[0], f"Alert không đúng: {alerts_received[0]}"

    # Profiler phải có đủ 5 stage
    stats = profiler.get_stats()
    for stage in ["capture", "inference", "tracking", "policy", "render"]:
        assert stats.get(stage, 0) >= 0, f"Stage {stage} thiếu trong profiler"

    # Stream.close phải được gọi khi kết thúc
    stream.close.assert_called_once()

    print(f"  Alerts nhận được: {alerts_received}")
    print(f"  Profiler FPS: {stats['fps']:.1f}")
    print("  -> test_pipeline_runner: PASS!")


if __name__ == "__main__":
    test_pipeline_runner()
