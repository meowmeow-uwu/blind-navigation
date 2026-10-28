"""
test_detector_output.py - Kiểm thử Hợp nhất chùm siêu âm HC-SR04 & Phân luồng khẩn cấp
=======================================================================================
Kiểm tra:
1. Yêu cầu 1: Luồng cảm biến siêu âm phản ứng khẩn cấp tức thời (< 30ms) qua callback độc lập.
2. Yêu cầu 2: Góc nón chùm siêu âm 30 độ gắn kết chính xác cự ly thực vào vật cản trong chùm.
3. Yêu cầu 4: Thuật toán xếp hạng nguy cơ (Risk Prioritization) ưu tiên vật nguy hiểm nhất.
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.sensors.ultrasonic_hcsr04 import HCSR04Sensor
from modules.tracking.byte_tracker import LightweightTracker, TrackedItem
from core.fusion_engine import SensorFusionEngine


def test_emergency_reaction_under_30ms():
    """Yêu cầu 1: Cảm biến siêu âm phát hiện nguy hiểm và gọi callback khẩn cấp trong < 30ms."""
    sensor = HCSR04Sensor(poll_rate_hz=50, emergency_distance_m=1.0)
    triggered = []

    def on_emergency(dist):
        triggered.append((time.perf_counter(), dist))

    sensor.set_emergency_callback(on_emergency)
    sensor.start()

    try:
        t_start = time.perf_counter()
        # Đột ngột xuất hiện vật cản ở cự ly 0.4m
        sensor.set_simulated_distance(0.4, manual=True)

        timeout = time.time() + 0.08
        while not triggered and time.time() < timeout:
            time.sleep(0.002)

        assert len(triggered) > 0, "Callback khẩn cấp không được gọi!"
        reaction_time_ms = (triggered[0][0] - t_start) * 1000.0
        print(f"  Thời gian phản ứng khẩn cấp thực tế: {reaction_time_ms:.2f} ms")
        assert reaction_time_ms < 30.0, f"Độ trễ {reaction_time_ms:.2f}ms vượt quá ngưỡng 30ms!"
    finally:
        sensor.stop()


def test_ultrasonic_beam_fusion():
    """Yêu cầu 2: Chùm sóng nón siêu âm 30 độ gán đúng cự ly thật cho vật trong chùm."""
    cfg = {
        "camera": {"c_x": 320.0, "c_y": 240.0, "frame_width": 640, "frame_height": 480},
        "ultrasonic": {"beam_angle_deg": 30.0, "emergency_distance_m": 1.0},
        "risk_scoring": {},
        "class_danger_weights": {"person": 0.8, "chair": 0.5}
    }
    engine = SensorFusionEngine(cfg)

    # Vật 1: Người đứng ở trung tâm làn giữa, nằm trong chùm siêu âm 30 độ (delta_x nhỏ)
    trk_person = TrackedItem(bbox=(250, 150, 390, 420), class_name="person", confidence=0.92)
    # Vật 2: Ghế nằm lệch hẳn về mép phải ngoài chùm phát sóng siêu âm (> 15 độ lệch)
    trk_chair = TrackedItem(bbox=(540, 200, 630, 380), class_name="chair", confidence=0.85)

    assert engine.is_in_ultrasonic_beam(trk_person.bbox) is True
    assert engine.is_in_ultrasonic_beam(trk_chair.bbox) is False

    # Hợp nhất với số đo cự ly siêu âm HC-SR04 là 1.50m
    fused_results, _ = engine.fuse([trk_person, trk_chair], raw_us_distance=1.50)

    # Người nằm trong chùm nhận khoảng cách từ siêu âm
    person_res = next(r for r in fused_results if r.track_id == trk_person.track_id)
    assert person_res.is_ultrasonic_target is True
    assert abs(person_res.distance_m - 1.50) < 0.25

    # Ghế nằm ngoài chùm dùng ước lượng camera
    chair_res = next(r for r in fused_results if r.track_id == trk_chair.track_id)
    assert chair_res.is_ultrasonic_target is False
    assert chair_res.source_status == "camera_scaled"


def test_risk_prioritization():
    """Yêu cầu 4: Vật nguy hiểm hơn (cầu thang/người trước mặt gần) phải có Risk Score cao nhất."""
    cfg = {
        "camera": {"c_x": 320.0, "c_y": 240.0, "frame_width": 640, "frame_height": 480},
        "ultrasonic": {"emergency_distance_m": 1.0},
        "class_danger_weights": {"stairs": 0.95, "chair": 0.40, "bottle": 0.20}
    }
    engine = SensorFusionEngine(cfg)

    # Cầu thang ở ngay trước mặt (làn giữa, 1.2m)
    risk_stairs = engine.compute_risk_score(distance=1.2, ttc=2.0, bbox=(220, 150, 420, 450), class_name="stairs")
    # Chai nước ở góc bên phải (làn phải, 2.5m)
    risk_bottle = engine.compute_risk_score(distance=2.5, ttc=999.0, bbox=(500, 250, 560, 380), class_name="bottle")

    print(f"  Risk cầu thang trước mặt (1.2m): {risk_stairs:.2f}")
    print(f"  Risk chai nước bên phải (2.5m): {risk_bottle:.2f}")
    assert risk_stairs > risk_bottle
    assert risk_stairs >= 0.60


if __name__ == "__main__":
    print("Running ultrasonic sensor & risk tests...")
    test_emergency_reaction_under_30ms()
    print("  test_emergency_reaction_under_30ms passed!")
    test_ultrasonic_beam_fusion()
    print("  test_ultrasonic_beam_fusion passed!")
    test_risk_prioritization()
    print("  test_risk_prioritization passed!")
    print("ALL SENSOR & RISK TESTS PASSED!")
