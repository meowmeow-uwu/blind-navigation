"""
main.py - Điểm khởi chạy hệ thống hỗ trợ người khiếm thị (Edge Device)
======================================================================
Tích hợp toàn diện:
- Camera OV5647 / USB Webcam
- Mô hình nhận dạng vật cản YOLOv8 NCNN
- Bộ theo dõi đa đối tượng ByteTracker
- Cảm biến siêu âm HC-SR04 (Đa luồng thời gian thực)
- Động cơ hợp nhất cảm biến SensorFusionEngine (Mức B: Kalman theo từng track + Tự hiệu chỉnh thang đo)
- Bộ điều phối âm thanh AudioFeedbackManager
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path

# Đảm bảo import đúng cấu trúc thư mục
CURR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURR_DIR))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import numpy as np

# Import các micro-modules
from modules.sensors.ultrasonic_hcsr04 import HCSR04Sensor
from modules.vision_local.base_detector import BaseDetector
from modules.tracking.byte_tracker import LightweightTracker
from core.fusion_engine import SensorFusionEngine, FusedObstacleResult
from modules.audio_tts.audio_manager import AudioFeedbackManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MAIN")


def load_detector(config: dict) -> BaseDetector:
    """Tải detector YOLOv8 NCNN hoặc fallback Mock Detector nếu chưa có thư viện NCNN."""
    param_path = CURR_DIR / "modules/vision_local/models/yolov8n.param"
    bin_path = CURR_DIR / "modules/vision_local/models/yolov8n.bin"

    try:
        from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
        if param_path.exists() and bin_path.exists():
            logger.info("[AI] Đang tải mô hình YOLOv8 NCNN...")
            return YoloNcnnRunner(str(param_path), str(bin_path), target_size=320)
    except Exception as e:
        logger.warning(f"[AI] Không thể khởi động NCNN ({e}). Sử dụng Mock Detector.")

    class MockDetector(BaseDetector):
        """Mô phỏng phát hiện vật thể phục vụ kiểm thử."""
        def load_model(self):
            pass
        def detect(self, frame: np.ndarray) -> list:
            # Tạo ngẫu nhiên một vật cản trước mặt
            return [
                {
                    'class_name': 'chair',
                    'bbox': [240, 180, 400, 420],
                    'confidence': 0.88
                }
            ]
    return MockDetector()


def draw_hud(frame: np.ndarray,
             fused_results: list,
             us_distance: float,
             scale_factor: float,
             fps: float,
             is_emergency: bool) -> np.ndarray:
    """Vẽ giao diện giám sát HUD trực quan khi chạy chế độ debug."""
    if not HAS_CV2 or frame is None:
        return frame

    h, w = frame.shape[:2]
    vis = frame.copy()

    # 1. Vẽ 3 làn đường (Trái - Giữa - Phải)
    third = w // 3
    cv2.line(vis, (third, 0), (third, h), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(vis, (2 * third, 0), (2 * third, h), (70, 70, 70), 1, cv2.LINE_AA)

    # 2. Vẽ hình nón chùm siêu âm (Beam Cone ~ 30 độ)
    beam_w = int(third * 0.9)
    pts = np.array([
        [w // 2, 0],
        [w // 2 - beam_w, h],
        [w // 2 + beam_w, h]
    ], np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], (255, 255, 0))
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    # 3. Vẽ Bounding Boxes và thông tin dung hợp
    for item in fused_results:
        x1, y1, x2, y2 = item.bbox

        if item.is_emergency:
            color = (0, 0, 255)  # Đỏ khẩn cấp
            tag = "EMERGENCY"
        elif item.risk_score >= 0.5:
            color = (0, 255, 255)  # Vàng cảnh giác
            tag = "URGENT"
        else:
            color = (0, 255, 0)  # Xanh an toàn
            tag = "SAFE"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label: ID | Class | Distance | TTC | Risk
        ttc_str = f"{item.ttc_seconds:.1f}s" if item.ttc_seconds < 900 else "INF"
        label = f"ID:{item.track_id} {item.class_name} | {item.distance_m:.2f}m | TTC:{ttc_str} | Risk:{item.risk_score:.2f}"

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(vis, (x1, max(0, y1 - 20)), (x1 + t_size[0], max(0, y1)), color, -1)
        cv2.putText(vis, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Thanh Header trạng thái hệ thống
    header_color = (0, 0, 150) if is_emergency else (30, 30, 30)
    cv2.rectangle(vis, (0, 0), (w, 35), header_color, -1)
    status_str = f"PBL6 Nav | FPS:{fps:.1f} | US ToF:{us_distance:.2f}m | Scale s:{scale_factor:.2f}"
    if is_emergency:
        status_str += " | 🚨 DỪNG LẠI!"
    cv2.putText(vis, status_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def main():
    logger.info("=" * 65)
    logger.info("KHỞI ĐỘNG HỆ THỐNG ĐỊNH HƯỚNG NGƯỜI KHIẾM THỊ (PBL6)")
    logger.info("Tích hợp: Siêu âm HC-SR04 + Camera YOLO (Mức B: Kalman + Scale)")
    logger.info("=" * 65)

    # 1. Tải cấu hình YAML
    config_path = CURR_DIR / "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Khởi tạo cảm biến siêu âm HC-SR04
    us_cfg = config.get("ultrasonic", {})
    ultrasonic = HCSR04Sensor(
        trig_pin=us_cfg.get("trig_pin", 23),
        echo_pin=us_cfg.get("echo_pin", 24),
        min_range_m=us_cfg.get("min_range_m", 0.03),
        max_range_m=us_cfg.get("max_range_m", 4.0),
        poll_rate_hz=us_cfg.get("poll_rate_hz", 30),
        emergency_distance_m=us_cfg.get("emergency_distance_m", 0.8)
    )
    ultrasonic.start()

    # 3. Khởi tạo Camera
    cam_cfg = config.get("camera", {})
    cap = None
    if HAS_CV2:
        cap = cv2.VideoCapture(cam_cfg.get("device_id", 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("frame_width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("frame_height", 480))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 4. Khởi tạo Detector, Tracker, Fusion Engine và Audio Manager
    detector = load_detector(config)
    tracker = LightweightTracker(max_age=12, min_hits=2, iou_threshold=0.25)
    fusion_engine = SensorFusionEngine(config)

    audio_cfg = config.get("audio", {})
    audio_mgr = AudioFeedbackManager(
        language=audio_cfg.get("language", "vi"),
        speech_rate=audio_cfg.get("rate", 200),
        emergency_cooldown_s=audio_cfg.get("emergency_cooldown_s", 0.6),
        tts_cooldown_s=audio_cfg.get("tts_cooldown_s", 2.0)
    )

    debug_view = config.get("system", {}).get("debug_view", True) and HAS_CV2
    fps_time = time.time()
    fps = 0.0

    try:
        while True:
            t_start = time.perf_counter()

            # Lấy ảnh từ Camera
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
            else:
                # Ảnh mô phỏng nếu không có camera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (40, 40, 40)

            # Đọc cảm biến siêu âm
            us_dist, us_ts, is_us_emergency = ultrasonic.get_latest_reading()

            # QUY TẮC AN TOÀN KHẨN CẤP:
            # Nếu cảm biến siêu âm báo quá gần (< 0.8m), phát cảnh báo dừng ngay!
            if is_us_emergency:
                audio_mgr.trigger_emergency(us_dist)

            # Chạy YOLO nhận dạng
            detections = detector.detect(frame)

            # Cập nhật Tracker để cấp phát Track ID
            tracked_items = tracker.update(detections)

            # Thực thi Sensor Fusion (Mức B: Kalman theo từng track + Hiệu chỉnh scale)
            fused_results, global_emergency = fusion_engine.fuse(tracked_items, us_dist)

            # Xử lý âm thanh dựa trên kết quả dung hợp
            if global_emergency:
                audio_mgr.trigger_emergency(us_dist)
            elif fused_results:
                top_obstacle = fused_results[0]
                audio_mgr.announce_obstacle(
                    track_id=top_obstacle.track_id,
                    class_name=top_obstacle.class_name,
                    distance_m=top_obstacle.distance_m,
                    lane=top_obstacle.lane,
                    is_emergency=top_obstacle.is_emergency
                )

            # Tính toán FPS
            t_elapsed = time.perf_counter() - t_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_elapsed))

            # Hiển thị giao diện nếu bật debug_view
            if debug_view:
                vis_frame = draw_hud(
                    frame=frame,
                    fused_results=fused_results,
                    us_distance=us_dist,
                    scale_factor=fusion_engine.scale_calibrator.scale,
                    fps=fps,
                    is_emergency=global_emergency or is_us_emergency
                )
                cv2.imshow("PBL6 - Smart Blind Navigation", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("\nNgười dùng yêu cầu dừng hệ thống.")
    finally:
        ultrasonic.stop()
        if cap and cap.isOpened():
            cap.release()
        if HAS_CV2:
            cv2.destroyAllWindows()
        logger.info("Đã giải phóng tài nguyên. Hệ thống an toàn.")


if __name__ == "__main__":
    main()
