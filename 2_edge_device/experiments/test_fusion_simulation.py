"""
test_fusion_simulation.py - Kiểm thử kịch bản mô phỏng toàn diện cho Sensor Fusion (Mức B)
==========================================================================================
Xác thực 100% các yêu cầu kỹ thuật:
1. Công thức hình học mặt sàn Ground-plane d_cam = f_y * h / (v_bottom - c_y) và Pinhole fallback.
2. Bộ lọc Kalman 2 trạng thái [d, ḋ] và tính toán TTC = d / (-ḋ).
3. Bảng xử lý khi hai nguồn mâu thuẫn (Đủ 4 tình huống).
4. Khả năng tự hiệu chỉnh thang đo trực tuyến (Online Scale Calibration).
5. Quy tắc an toàn khẩn cấp (Fail-Safe Emergency Rule).
"""

import sys
import os
import yaml
from pathlib import Path

# Thêm đường dẫn gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.kalman_fusion import CameraDistanceEstimator, TrackDistanceKalmanFilter, OnlineScaleCalibrator
from modules.tracking.byte_tracker import TrackedItem
from core.fusion_engine import SensorFusionEngine


def load_test_config() -> dict:
    config_path = BASE_DIR / "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_1_camera_ground_plane_and_pinhole():
    print("\n--- TEST 1: KIỂM THỬ TÍNH KHOẢNG CÁCH CAMERA (GROUND-PLANE & PINHOLE) ---")
    estimator = CameraDistanceEstimator(
        f_x=480.0, f_y=480.0, c_x=320.0, c_y=240.0,
        camera_height_m=1.45, tilt_angle_deg=5.0,
        frame_height=480, frame_width=640
    )

    # Trường hợp 1: Ghế đứng trên sàn, đáy bbox ở hàng v_bottom = 380 (dưới tâm c_y=240)
    bbox_chair = (250, 200, 390, 380)
    d_ground, method = estimator.estimate_distance(bbox_chair, "chair")
    print(f"  [Ground-plane] Ghế tiếp sàn (v_bottom=380): cự ly = {d_ground:.2f}m (Phương pháp: {method})")
    assert method == "ground_plane"
    assert 1.0 <= d_ground <= 4.0

    # Trường hợp 2: Biển báo/vật không chạm sàn hoặc bị cắt ở đáy ảnh (v_bottom = 475 sát mép 480)
    bbox_cut = (200, 100, 440, 475)
    d_pinhole, method_pin = estimator.estimate_distance(bbox_cut, "chair")
    print(f"  [Pinhole Fallback] Vật cắt đáy ảnh: cự ly = {d_pinhole:.2f}m (Phương pháp: {method_pin})")
    assert method_pin == "pinhole_heuristic"
    print("   TEST 1 PASSED!")


def test_2_kalman_state_and_ttc():
    print("\n--- TEST 2: KIỂM THỬ BỘ LỌC KALMAN [d, ḋ] VÀ TIME-TO-COLLISION (TTC) ---")
    kf = TrackDistanceKalmanFilter(initial_distance=3.0)

    # Mô phỏng đối tượng đang di chuyển lại gần với vận tốc 1.0 m/s qua 10 bước (mỗi bước 0.1s)
    sim_distances = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1]
    for d_meas in sim_distances:
        kf.predict(dt=0.1)
        kf.update_camera(d_meas)

    print(f"  Khoảng cách lọc: {kf.distance:.2f}m (Kỳ vọng ~2.1m)")
    print(f"  Vận tốc d_dot: {kf.velocity:.2f} m/s (Kỳ vọng âm do đang tiến lại gần)")
    print(f"  Tốc độ tiến lại: {kf.closing_speed:.2f} m/s (Kỳ vọng ~1.0 m/s)")
    ttc = kf.get_ttc()
    print(f"  Time-To-Collision (TTC = d / -ḋ): {ttc:.2f} giây")

    assert kf.velocity < -0.3  # d_dot âm
    assert 1.5 <= ttc <= 3.5  # TTC = 2.1m / 1.0m/s ~ 2.1s
    print("   TEST 2 PASSED!")


def test_3_conflict_resolution_matrix():
    print("\n--- TEST 3: KIỂM THỬ BẢNG XỬ LÝ KHI HAI NGUỒN MÂU THUẪN (ĐỦ 4 TÌNH HUỐNG) ---")
    config = load_test_config()
    engine = SensorFusionEngine(config)

    # -------------------------------------------------------------------------
    # Tình huống 1: Hai nguồn khớp (lệch dưới 25%)
    # -------------------------------------------------------------------------
    # Ghế tiếp sàn ở cự ly khoảng ~4.48m (v_bottom = 350)
    track1 = TrackedItem(bbox=(250, 180, 390, 350), class_name="chair", confidence=0.88)
    track1.age = 5
    # Siêu âm đo 4.40m (lệch ~1.8% so với camera 4.48m -> rất khớp)
    res1, is_emer1 = engine.fuse([track1], raw_us_distance=4.40, dt=0.033)
    print(f"  [Tình huống 1 - Khớp < 25%]:")
    print(f"    Khoảng cách hợp nhất: {res1[0].distance_m}m | Trạng thái: {res1[0].source_status}")
    assert res1[0].source_status == "two_sources_matched"

    # -------------------------------------------------------------------------
    # Tình huống 2: Siêu âm xa hơn camera nhiều (vật mềm tiêu âm, bề mặt nghiêng)
    # -------------------------------------------------------------------------
    engine2 = SensorFusionEngine(config)
    # Ghế ở gần (đáy cắt ở mép ảnh 475, width=120 -> d_cam = 2.0m)
    track2 = TrackedItem(bbox=(260, 100, 380, 475), class_name="chair", confidence=0.85)
    # Siêu âm dội sóng nghiêng/tiêu âm báo xa tận 3.5m (> 2.0m * 1.25)
    res2, is_emer2 = engine2.fuse([track2], raw_us_distance=3.50, dt=0.033)
    print(f"  [Tình huống 2 - Siêu âm xa hơn camera]:")
    print(f"    Khoảng cách bảo thủ: {res2[0].distance_m}m | Trạng thái: {res2[0].source_status}")
    assert res2[0].source_status == "us_penalized"
    assert res2[0].distance_m <= 2.2  # Tin camera, cảnh báo bảo thủ cự ly gần

    # -------------------------------------------------------------------------
    # Tình huống 3: Siêu âm báo gần, camera không thấy vật (vách kính, tường trơn)
    # -------------------------------------------------------------------------
    engine3 = SensorFusionEngine(config)
    # Không có bounding box nào từ camera, nhưng siêu âm báo 0.75m (< emergency)
    res3, is_emer3 = engine3.fuse([], raw_us_distance=0.75, dt=0.033)
    res3, is_emer3 = engine3.fuse([], raw_us_distance=0.75, dt=0.033)  # Lần 2 để confirm
    print(f"  [Tình huống 3 - Siêu âm báo gần, Camera không thấy]:")
    print(f"    Số vật cản cảnh báo: {len(res3)} | Tên: {res3[0].class_name} | Khẩn cấp: {is_emer3}")
    assert len(res3) == 1
    assert res3[0].class_name == "vật cản"
    assert res3[0].source_status == "unlabeled_obstacle"
    assert is_emer3 is True

    # -------------------------------------------------------------------------
    # Tình huống 4: Nhiều vật trong một chùm siêu âm
    # -------------------------------------------------------------------------
    engine4 = SensorFusionEngine(config)
    # Vật A ở gần (cắt đáy ảnh, width=200 -> d_cam ~ 1.2m)
    track_near = TrackedItem(bbox=(220, 100, 420, 475), class_name="chair", confidence=0.88)
    # Vật B ở xa phía sau (v_bottom=350 -> d_cam ~ 4.48m)
    track_far = TrackedItem(bbox=(290, 180, 350, 350), class_name="person", confidence=0.82)

    res4, is_emer4 = engine4.fuse([track_near, track_far], raw_us_distance=1.22, dt=0.033)
    print(f"  [Tình huống 4 - Nhiều vật trong 1 chùm siêu âm]:")
    print(f"    Vật nhận echo siêu âm: ID:{res4[0].track_id} ({res4[0].class_name}) - cự ly {res4[0].distance_m}m")
    print(f"    Vật phía sau: ID:{res4[1].track_id} ({res4[1].class_name}) - cự ly {res4[1].distance_m}m ({res4[1].source_status})")
    assert res4[0].track_id == track_near.track_id
    assert res4[0].distance_m < res4[1].distance_m
    assert res4[1].source_status == "camera_scaled"
    print("   TEST 3 PASSED!")


def test_4_online_scale_calibration():
    print("\n--- TEST 4: KIỂM THỬ TỰ HIỆU CHỈNH THANG ĐO (ONLINE SCALE CALIBRATION) ---")
    calibrator = OnlineScaleCalibrator(learning_rate=0.2, initial_scale=1.0, min_stable_frames=2)

    # Giả sử camera đang ước lượng thiếu: d_cam = 1.0m, nhưng siêu âm đo thực là 1.3m (tỷ lệ 1.3)
    print(f"  Thang đo ban đầu: s = {calibrator.scale:.3f}")
    for i in range(5):
        s_curr = calibrator.update(d_us=1.3, d_cam=1.0, track_age=i + 2)

    print(f"  Thang đo sau khi tự học: s = {calibrator.scale:.3f} (Kỳ vọng tiến tới ~1.3)")
    assert calibrator.scale > 1.15
    print("   TEST 4 PASSED!")


def test_5_fail_safe_emergency_rule():
    print("\n--- TEST 5: KIỂM THỬ QUY TẮC AN TOÀN KHẨN CẤP (FAIL-SAFE EMERGENCY RULE) ---")
    config = load_test_config()
    engine = SensorFusionEngine(config)

    # Ngay cả khi chưa nhận diện được gì, nếu siêu âm đo được 0.5m (< 0.8m emergency)
    _, is_emergency = engine.fuse([], raw_us_distance=0.50, dt=0.033)
    print(f"  Siêu âm báo 0.5m mà không có nhãn -> Cờ khẩn cấp lập tức: {is_emergency}")
    assert is_emergency is True
    print("   TEST 5 PASSED!")


def run_all_tests():
    print("=" * 65)
    print("CHẠY BỘ KIỂM THỬ MÔ PHỎNG TOÀN DIỆN SENSOR FUSION (MỨC B)")
    print("=" * 65)
    try:
        test_1_camera_ground_plane_and_pinhole()
        test_2_kalman_state_and_ttc()
        test_3_conflict_resolution_matrix()
        test_4_online_scale_calibration()
        test_5_fail_safe_emergency_rule()
        print("\n" + "=" * 65)
        print("TẤT CẢ 5 BÀI TEST ĐỀU ĐẠT CHUẨN XUẤT SẮC! (100% PASS)")
        print("=" * 65)
        return 0
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH KIỂM THỬ: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
