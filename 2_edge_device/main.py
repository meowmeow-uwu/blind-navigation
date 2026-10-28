"""
main.py - Hệ thống Hỗ trợ Người Khiếm thị Thông minh (PBL6)
============================================================
Kiến trúc 5 trụ cột cốt lõi:
1. Đa luồng bất đồng bộ: Cảm biến siêu âm HC-SR04 chạy độc lập 40Hz, báo khẩn cấp tức thời (<30ms).
2. Hợp nhất đa phương thức: Chùm sóng nón siêu âm 30 độ gắn cự ly thực tế vào vật cản gần nhất.
3. Bám vết thời gian thực: ByteTrack cấp Track ID duy nhất, chỉ nói 1 lần rồi bám vết ngầm.
4. Đánh giá nguy cơ (Risk Prioritization): Xếp hạng theo mức độ nguy hại, cự ly và hướng di chuyển.
5. Giao tiếp thích ứng đa giác quan:
   - < 1.0m: Tiếng bíp dồn dập (tần số/nhịp tăng dần), ngắt câu nói dài, tạo phản xạ dừng ngay.
   - 1.5m - 3.0m: Giọng nói ngắn gọn 1-2 từ khóa ("Bàn trước mặt", "Cột bên trái"), không đè tiếng.
   - > 3.0m: Im lặng thông thoáng thính giác.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from modules.sensors.ultrasonic_hcsr04 import HCSR04Sensor
from modules.tracking.byte_tracker import LightweightTracker
from core.fusion_engine import SensorFusionEngine
from modules.audio.audio_manager import AudioManager
from modules.vision_local.synthetic_camera import SyntheticCamera

LABEL_DISPLAY = {
    "person": "NGUOI", "chair": "GHE", "dining table": "BAN",
    "stairs": "CAU THANG", "pole": "COT", "door": "CUA",
    "vật cản": "VAT CAN", "bicycle": "XE DAP", "car": "O TO"
}


def load_config() -> dict:
    cfg_path = os.path.join(CURRENT_DIR, "configs/config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def init_detector():
    """Khởi tạo YOLOv8 NCNN hoặc Mock Detector nếu chạy trên máy tính phát triển."""
    param_path = os.path.join(CURRENT_DIR, "modules/vision_local/models/yolov8n.param")
    bin_path = os.path.join(CURRENT_DIR, "modules/vision_local/models/yolov8n.bin")
    try:
        from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
        if os.path.exists(param_path) and os.path.exists(bin_path):
            return YoloNcnnRunner(param_path, bin_path, target_size=320)
    except Exception:
        pass

    class SimpleMockDetector:
        def detect(self, _):
            return [{'class_name': 'chair', 'bbox': [240, 180, 400, 420], 'confidence': 0.88}]
    return SimpleMockDetector()


def draw_hud(frame: np.ndarray, fused_results: list, us_dist: float, is_emergency: bool, fps: float) -> np.ndarray:
    """Vẽ giao diện giám sát HUD trực quan (3 làn đường, góc nón siêu âm 30 độ, BBox, trạng thái)."""
    if not HAS_CV2 or frame is None:
        return frame

    h, w = frame.shape[:2]
    vis = frame.copy()
    third = w // 3

    # 1. Vẽ 3 làn đường (Trái - Giữa - Phải)
    cv2.line(vis, (third, 0), (third, h), (70, 70, 70), 1)
    cv2.line(vis, (2 * third, 0), (2 * third, h), (70, 70, 70), 1)
    cv2.putText(vis, "TRAI", (third // 2 - 20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(vis, "GIUA", (w // 2 - 20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(vis, "PHAI", (2 * third + third // 2 - 20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # 2. Vẽ hình nón chùm siêu âm HC-SR04 (~30 độ)
    beam_w = int(third * 0.9)
    pts = np.array([
        [w // 2, 0],
        [w // 2 - beam_w, h],
        [w // 2 + beam_w, h]
    ], np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], (255, 255, 0))
    cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)

    # 3. Vẽ Bounding Boxes các vật cản sau Sensor Fusion & Xếp hạng nguy cơ (Yêu cầu 4)
    for i, item in enumerate(fused_results):
        x1, y1, x2, y2 = item.bbox
        is_top = (i == 0)
        color = (0, 0, 255) if item.is_emergency else ((0, 255, 255) if is_top else (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2 if not is_top else 3)

        d_name = LABEL_DISPLAY.get(item.class_name, item.class_name.upper())
        tag = f"#{item.track_id} {d_name} | {item.distance_m:.1f}m | R:{item.risk_score:.2f}"
        if getattr(item, "is_ultrasonic_target", False):
            tag += " [*SIEU AM]"
        t_sz = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.rectangle(vis, (x1, max(0, y1 - 18)), (x1 + t_sz[0] + 6, max(0, y1)), color, -1)
        cv2.putText(vis, tag, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    # 4. Thanh Header trạng thái
    hdr_color = (0, 0, 160) if is_emergency else (30, 30, 30)
    cv2.rectangle(vis, (0, 0), (w, 30), hdr_color, -1)
    status_msg = f"PBL6 NAV | FPS: {fps:.1f} | SIEU AM: {us_dist:.2f}m"
    if is_emergency:
        status_msg += " | [!] KHAN CAP: DUNG LAI (BEEP)"
    cv2.putText(vis, status_msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return vis


def main():
    parser = argparse.ArgumentParser(description="PBL6 - Smart Blind Navigation Edge System (Ultrasonic HC-SR04)")
    parser.add_argument("--source", type=str, default="auto", help="Nguồn video: 'auto', 'demo', 0, hoặc đường dẫn mp4")
    parser.add_argument("--port", type=int, default=8080, help="Cổng Web Dashboard & Audio (mặc định: 8080)")
    parser.add_argument("--headless", action="store_true", help="Chạy không cửa sổ GUI màn hình")
    parser.add_argument("--mute-local", action="store_true",
                        help="Tắt loa cục bộ trên Pi, chỉ phát âm thanh qua Web Dashboard trên máy tính Mac")
    args = parser.parse_args()

    print("=" * 68)
    print("   HỆ THỐNG ĐIỀU HƯỚNG KHIẾM THỊ THÔNG MINH (PBL6 EDGE-AI)")
    print("   Cảm biến siêu âm HC-SR04 (30-50Hz) + Camera YOLOv8 NCNN")
    if args.mute_local:
        print("   📢 Chế độ âm thanh: Phát qua Web Dashboard (Loa máy Mac/Điện thoại)")
    else:
        print("   📢 Chế độ âm thanh: Phát đồng thời Loa ngoài thiết bị & Web Dashboard")
    print("=" * 68)

    config = load_config()

    # 1. Khởi tạo Mô hình AI & Tracker
    detector = init_detector()
    is_mock = not hasattr(detector, "load_model")
    tracker = LightweightTracker(max_age=12, min_hits=2, iou_threshold=0.25)
    fusion_engine = SensorFusionEngine(config)

    # 2. Khởi tạo Audio Manager (Chống đè tiếng & Phân tầng phản hồi - Yêu cầu 5)
    audio = AudioManager(port=args.port, guidance_cooldown_s=2.5, enable_local_audio=not args.mute_local)

    # 3. Khởi tạo Cảm biến siêu âm HC-SR04 đa luồng (Yêu cầu 1: Bất đồng bộ 40Hz)
    us_cfg = config.get("ultrasonic", {})
    sensor = HCSR04Sensor(
        trig_pin=us_cfg.get("trig_pin", 23),
        echo_pin=us_cfg.get("echo_pin", 24),
        poll_rate_hz=us_cfg.get("poll_rate_hz", 40),
        emergency_distance_m=us_cfg.get("emergency_distance_m", 1.0)
    )
    # Callback khẩn cấp tức thời (<30ms) trực tiếp từ luồng siêu âm, không chờ AI
    sensor.set_emergency_callback(lambda d: audio.trigger_emergency(d))
    sensor.start()
    audio.set_distance_callback(sensor.set_simulated_distance)

    # 4. Khởi tạo Nguồn Video
    cap = None
    synth_cam = None
    use_synthetic = (args.source.lower() in ["demo", "synthetic"])

    if not use_synthetic and HAS_CV2:
        cam_id = int(args.source) if (args.source.isdigit() and args.source != "auto") else 0
        cap = cv2.VideoCapture(cam_id)
        if not (cap and cap.isOpened()):
            print("[CAMERA] Không mở được webcam vật lý -> Chuyển sang Camera Mô Phỏng 3D.")
            use_synthetic = True
            synth_cam = SyntheticCamera()
    else:
        use_synthetic = True
        synth_cam = SyntheticCamera()

    print(f"\n👉 Mở trình duyệt xem trực tiếp: http://localhost:{args.port}")
    print("👉 Phím test cự ly: [1] 3.0m (An toàn)  [2] 2.0m (Dẫn hướng)  [3] 0.5m (Khẩn cấp)  [Q] Thoát\n")

    fps = 0.0
    debug_view = (not args.headless) and HAS_CV2

    try:
        while True:
            t_start = time.perf_counter()

            # Lấy khung hình
            sim_dets = []
            if use_synthetic and synth_cam:
                _, frame, sim_dets, sim_d = synth_cam.read()
                if sensor.is_simulated and not sensor.is_manually_overridden:
                    sensor.set_simulated_distance(sim_d, manual=False)
            elif cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Đọc khoảng cách siêu âm mới nhất
            raw_d, _, is_sensor_emergency = sensor.get_latest_reading()

            # Nhận diện vật thể & Bám vết (Yêu cầu 3: ByteTrack)
            detections = sim_dets if (use_synthetic and is_mock) else detector.detect(frame)
            tracked_items = tracker.update(detections)

            # Hợp nhất cảm biến siêu âm & Xếp hạng nguy cơ (Yêu cầu 2 & 4)
            fused_results, global_emergency = fusion_engine.fuse(tracked_items, raw_d)

            # Điều phối phản hồi thích ứng đa giác quan (Yêu cầu 5)
            if global_emergency or is_sensor_emergency or (raw_d is not None and raw_d < 1.0):
                audio.trigger_emergency(raw_d or 0.8)
            else:
                audio.clear_emergency()
                # Nếu có vật cản và nằm trong khoảng dẫn hướng (1.5m - 3.0m)
                if fused_results:
                    top = fused_results[0]
                    # Chỉ thông báo 1-2 từ khoá và ĐÚNG 1 LẦN duy nhất theo Track ID (Yêu cầu 3 & 5)
                    if 1.5 <= top.distance_m <= 3.0 and top.track_id != -1:
                        for trk in tracker.tracks:
                            if trk.track_id == top.track_id and not trk.announced:
                                spoken = audio.announce_guidance(top.track_id, top.class_name, top.lane, top.distance_m)
                                if spoken:
                                    trk.mark_announced(top.distance_m)
                                break

            # Tính FPS và vẽ HUD
            t_elapsed = time.perf_counter() - t_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_elapsed))

            vis_frame = draw_hud(
                frame=frame,
                fused_results=fused_results,
                us_dist=raw_d,
                is_emergency=global_emergency or is_sensor_emergency,
                fps=fps
            )
            audio.update_frame(vis_frame)

            # Hiển thị cửa sổ Desktop và nhận phím bấm mô phỏng
            if debug_view:
                cv2.imshow("PBL6 - Smart Blind Navigation", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key == ord('1'):
                    sensor.set_simulated_distance(3.0, manual=True)
                elif key == ord('2'):
                    sensor.set_simulated_distance(2.0, manual=True)
                elif key == ord('3'):
                    sensor.set_simulated_distance(0.5, manual=True)
                elif key == ord('4'):
                    sensor.set_simulated_distance(0.7, manual=True)
            else:
                time.sleep(max(0.001, 0.033 - t_elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Đang đóng hệ thống an toàn...")
        audio.stop()
        sensor.stop()
        if cap and cap.isOpened():
            cap.release()
        if HAS_CV2:
            cv2.destroyAllWindows()
        print("[DONE] Đã giải phóng toàn bộ tài nguyên.")


if __name__ == "__main__":
    main()
