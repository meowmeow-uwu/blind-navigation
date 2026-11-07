# 2_edge_device/tests/test_tracker_and_profiler.py
import os
import sys
import time
import numpy as np
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_DEVICE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if EDGE_DEVICE_DIR not in sys.path:
    sys.path.insert(0, EDGE_DEVICE_DIR)

from modules.vision_local.tracker import ByteTrackerWrapper
from modules.utils.profiler import PipelineProfiler


def test_tracker():
    print("[TEST] Bắt đầu test ByteTrackerWrapper...")
    tracker = ByteTrackerWrapper(track_thresh=0.25, track_buffer=30, match_thresh=0.8)

    classes = ["person", "car"]
    track_ids = []

    for frame_idx in range(8):
        # Giả lập 1 vật thể di chuyển dần theo trục X
        boxes = np.array([[100 + frame_idx * 5, 100, 180 + frame_idx * 5, 220]], dtype=np.float32)
        confs = np.array([0.9], dtype=np.float32)
        cls_ids = np.array([0], dtype=int)

        tracked = tracker.update(boxes, confs, cls_ids, classes=classes)
        if len(tracked) > 0:
            track_ids.append(tracked[0]["track_id"])
            print(f"  Frame {frame_idx}: Track ID = {tracked[0]['track_id']}, "
                  f"BBox = {tracked[0]['bbox']}, Class = {tracked[0]['class_name']}")

    assert len(track_ids) > 0, "Tracker không phát hiện object nào"
    # track_id = -1 là pending track (frame đầu ByteTrack chưa xác nhận), bỏ qua
    confirmed_ids = [tid for tid in track_ids if tid >= 0]
    unique_ids = set(confirmed_ids)
    assert len(unique_ids) == 1, f"Track ID bị nhảy: {confirmed_ids}"
    print(f"  -> test_tracker: PASS! (Duy trì ổn định Track ID: {list(unique_ids)[0]})")


def test_profiler():
    print("[TEST] Bắt đầu test PipelineProfiler...")
    profiler = PipelineProfiler(buffer_size=10)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(5):
        with profiler.measure("capture"):
            time.sleep(0.002)
        with profiler.measure("inference"):
            time.sleep(0.010)
        with profiler.measure("tracking"):
            time.sleep(0.003)
        with profiler.measure("policy"):
            time.sleep(0.001)
        with profiler.measure("render"):
            time.sleep(0.002)
        profiler.tick_frame()

    stats = profiler.get_stats()
    print("  -> Profiler stats:", {k: f"{v:.2f}" for k, v in stats.items()})

    assert stats["inference"] > 5.0, "Sai số đo inference"
    assert stats["fps"] > 0, "FPS không hợp lệ"

    hud_frame = profiler.draw_hud(dummy_frame)
    assert hud_frame.shape == (480, 640, 3), "Sai shape ảnh sau khi vẽ HUD"
    print("  -> test_profiler: PASS!")


if __name__ == "__main__":
    test_tracker()
    test_profiler()
