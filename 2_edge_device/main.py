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
from modules.audio.dashboard_audio_client import DashboardAudioClient
from modules.vision_local.synthetic_camera import SyntheticCamera
from modules.vision_local.freshest_camera import FreshestCameraReader

LABEL_DISPLAY = {
    "person": "NGUOI", "chair": "GHE", "dining table": "BAN",
    "stairs": "CAU THANG", "pole": "COT", "door": "CUA",
    "vật cản": "VAT CAN", "bicycle": "XE DAP", "car": "O TO",
    "motorcycle": "XE MAY", "bus": "XE BUS", "truck": "XE TAI",
    "bottle": "CHAI NUOC", "backpack": "BALO", "cell phone": "DIEN THOAI",
    "laptop": "MAY TINH", "tv": "MAN HINH"
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
            detector = YoloNcnnRunner(param_path, bin_path, target_size=320, conf_thresh=0.20)
            print("[AI] ✅ Đã nạp thành công mô hình YOLOv8 NCNN thời gian thực (target_size=320, conf_thresh=0.20)!")
            return detector
    except Exception as e:
        print(f"[AI] ⚠️ Không thể nạp YOLOv8 NCNN ({e}). Sử dụng Mock Detector.")

    class SimpleMockDetector:
        def detect(self, _):
            return []
    return SimpleMockDetector()


def read_cpu_temp():
    """Đọc nhiệt độ CPU Raspberry Pi từ thermal_zone0 (độ C)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
            return round(float(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None


def make_error_frame(title: str, subtitle: str = "", us_dist=None) -> np.ndarray:
    """Tạo khung hình báo lỗi rõ ràng khi chưa có camera vật lý (không dùng mô phỏng)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (22, 26, 36)
    if HAS_CV2:
        cv2.rectangle(frame, (18, 18), (622, 462), (0, 0, 200), 2)
        cv2.putText(frame, title, (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 110, 255), 2)
        if subtitle:
            cv2.putText(frame, subtitle, (70, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 205, 225), 1)
        us_str = f"{us_dist:.2f}m" if us_dist is not None else "Chua ket noi"
        cv2.putText(frame, f"Sieu am: {us_str}", (70, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 220, 255), 1)
    return frame


def open_physical_camera(source_arg: str):
    """Mở camera vật lý theo index hoặc quét các cổng 0..6; trả về (cap, fresh_cap) hoặc (None, None)."""
    if not HAS_CV2:
        return None, None
    if source_arg == "0" and not os.path.exists("/dev/video0"):
        cam_indices = [1, 2, 3, 4, 5, 6]
    elif source_arg.isdigit():
        cam_indices = [int(source_arg)]
    else:
        cam_indices = [0, 1, 2, 3, 4, 5, 6]
    backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
    for c_idx in cam_indices:
        if not os.path.exists(f"/dev/video{c_idx}"):
            continue
        try:
            c_test = cv2.VideoCapture(c_idx, backend)
            if c_test.isOpened():
                try:
                    c_test.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
                c_test.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                c_test.set(cv2.CAP_PROP_FPS, 30)
                ret_check, test_f = c_test.read()
                if ret_check and test_f is not None:
                    print(f"[CAMERA] ✅ Mở thành công Camera vật lý tại cổng index: {c_idx} (Độ trễ tối thiểu)")
                    return c_test, FreshestCameraReader(c_test)
                c_test.release()
        except Exception:
            pass
    return None, None


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
    us_str = f"{us_dist:.2f}m" if us_dist is not None else "--"
    status_msg = f"PBL6 NAV | FPS: {fps:.1f} | SIEU AM: {us_str}"
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
    parser.add_argument("--dashboard-url", type=str, default="http://127.0.0.1:8088",
                        help="URL Web Dashboard để nhận TTS/đàm thoại PTT (mặc định: http://127.0.0.1:8088)")
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
    dash_url = config.get("dashboard", {}).get("url", args.dashboard_url)
    audio_bridge = DashboardAudioClient(dash_url, audio)
    audio_bridge.start()

    # 3. Khởi tạo Cảm biến siêu âm HC-SR04 đa luồng (Yêu cầu 1: Bất đồng bộ 40Hz)
    is_demo = (args.source.lower() in ["demo", "synthetic"])
    us_cfg = config.get("ultrasonic", {})
    sensor = HCSR04Sensor(
        trig_pin=us_cfg.get("trig_pin", 23),
        echo_pin=us_cfg.get("echo_pin", 24),
        poll_rate_hz=us_cfg.get("poll_rate_hz", 40),
        emergency_distance_m=us_cfg.get("emergency_distance_m", 1.0),
        allow_simulation=is_demo
    )
    # Callback khẩn cấp tức thời (<30ms) trực tiếp từ luồng siêu âm, không chờ AI
    sensor.set_emergency_callback(lambda d: audio.trigger_emergency(d))
    sensor.start()
    if is_demo:
        audio.set_distance_callback(sensor.set_simulated_distance)

    # 4. Khởi tạo Nguồn Video (Chỉ dùng mô phỏng 3D khi truyền --source demo/synthetic)
    cap = None
    fresh_cap = None
    synth_cam = None
    use_synthetic = is_demo
    camera_connected = False
    camera_error = ""
    last_cam_retry = 0.0
    cam_read_failures = 0

    if not use_synthetic:
        if HAS_CV2:
            cap, fresh_cap = open_physical_camera(args.source)
            if fresh_cap is not None:
                camera_connected = True
            else:
                camera_error = "Camera chưa kết nối"
                print("[CAMERA] ❌ Camera chưa kết nối! Hiển thị trạng thái lỗi (không chuyển sang mô phỏng).")
        else:
            camera_error = "Thiếu thư viện OpenCV (cv2)"
            print("[CAMERA] ❌ Thiếu thư viện OpenCV (cv2).")
    else:
        synth_cam = SyntheticCamera()
        camera_connected = True

    print(f"\n👉 Mở trình duyệt xem trực tiếp: http://localhost:{audio.port}")
    if is_demo:
        print("👉 Phím test cự ly: [1] 3.0m (An toàn)  [2] 2.0m (Dẫn hướng)  [3] 0.5m (Khẩn cấp)  [Q] Thoát\n")
    else:
        print("👉 Chế độ phần cứng thật: tự động báo lỗi rõ ràng nếu thiếu camera / siêu âm / GPS. Nhấn [Q] để thoát.\n")

    fps = 0.0
    has_display = bool(os.environ.get("DISPLAY")) or (sys.platform == "darwin")
    debug_view = (not args.headless) and HAS_CV2 and has_display
    dev_name = config.get("system", {}).get("device_name", "raspi")
    emg_thresh = float(us_cfg.get("emergency_distance_m", 1.0))

    try:
        while True:
            t_start = time.perf_counter()

            # Lấy khung hình tươi mới nhất (Zero Latency)
            sim_dets = []
            frame = None
            if use_synthetic and synth_cam:
                _, frame, sim_dets, sim_d = synth_cam.read()
                camera_connected = True
                camera_error = ""
                if sensor.is_simulated and not sensor.is_manually_overridden:
                    sensor.set_simulated_distance(sim_d, manual=False)
            else:
                if fresh_cap and fresh_cap.isOpened():
                    ret, frame = fresh_cap.read()
                    if ret and frame is not None:
                        camera_connected = True
                        camera_error = ""
                        cam_read_failures = 0
                    else:
                        cam_read_failures += 1
                        if (not fresh_cap.isOpened()) or cam_read_failures >= 2:
                            print("[CAMERA] ❌ Mất tín hiệu / đơ camera vật lý, đang giải phóng để kết nối lại...")
                            fresh_cap.release()
                            fresh_cap = None
                            cap = None
                            camera_connected = False
                            camera_error = "Camera chưa kết nối"
                            last_cam_retry = time.time() - 1.5
                        else:
                            time.sleep(0.01)
                            continue
                else:
                    camera_connected = False
                    camera_error = "Camera chưa kết nối" if HAS_CV2 else "Thiếu thư viện OpenCV (cv2)"
                    now_retry = time.time()
                    if HAS_CV2 and (now_retry - last_cam_retry >= 2.5):
                        last_cam_retry = now_retry
                        cap, fresh_cap = open_physical_camera(args.source)
                        if fresh_cap is not None:
                            camera_connected = True
                            camera_error = ""
                            cam_read_failures = 0
                            ret, frame = fresh_cap.read()
                            if not ret or frame is None:
                                camera_connected = False
                                camera_error = "Camera chưa đọc được khung hình"

            # Đọc khoảng cách siêu âm mới nhất
            raw_d, _, is_sensor_emergency = sensor.get_latest_reading()
            us_connected = (raw_d is not None)
            us_error = "" if us_connected else getattr(sensor, "_error_message", "") or "Chưa kết nối cảm biến siêu âm"

            # Nhận diện vật thể & Bám vết (Yêu cầu 3: ByteTrack)
            if use_synthetic:
                detections = sim_dets
            elif camera_connected and frame is not None:
                detections = detector.detect(frame)
            else:
                detections = []
            tracked_items = tracker.update(detections) if detections or tracker.tracks else []

            # Hợp nhất cảm biến siêu âm & Xếp hạng nguy cơ (Yêu cầu 2 & 4)
            fused_results, global_emergency = fusion_engine.fuse(tracked_items, raw_d)

            # Điều phối phản hồi thích ứng đa giác quan (Yêu cầu 5)
            is_any_emergency = global_emergency or is_sensor_emergency or (raw_d is not None and raw_d < emg_thresh)
            if is_any_emergency:
                audio.trigger_emergency(raw_d if raw_d is not None else 0.8)
            else:
                audio.clear_emergency()
                # Nếu có vật cản và nằm trong khoảng dẫn hướng (1.0m - 3.0m)
                if fused_results:
                    # Duyệt theo mức độ nguy cơ giảm dần, tìm vật cản ưu tiên cao nhất cần xướng âm (Yêu cầu 3, 4 & 5)
                    for obs in fused_results:
                        if 1.0 <= obs.distance_m <= 3.0 and obs.track_id != -1:
                            trk = next((t for t in tracker.tracks if t.track_id == obs.track_id), None)
                            if trk and trk.should_reannounce(obs.distance_m):
                                spoken = audio.announce_guidance(obs.track_id, obs.class_name, obs.lane, obs.distance_m, force=trk.announced)
                                if spoken:
                                    trk.mark_announced(obs.distance_m)
                                break  # Chỉ phát âm thanh cho 1 vật cản ưu tiên nhất

            # Tính FPS và vẽ HUD hoặc màn hình báo lỗi camera
            t_elapsed = time.perf_counter() - t_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_elapsed))

            if camera_connected and frame is not None:
                vis_frame = draw_hud(
                    frame=frame,
                    fused_results=fused_results,
                    us_dist=raw_d,
                    is_emergency=is_any_emergency,
                    fps=fps
                )
            else:
                vis_frame = make_error_frame(
                    "CAMERA CHUA KET NOI",
                    "Kiem tra cam USB Webcam / cap CSI tren Raspberry Pi",
                    us_dist=raw_d
                )
            audio.update_frame(vis_frame)

            # Cập nhật telemetry thực tế cho Web Dashboard (/telemetry)
            us_status = "disconnected" if not us_connected else ("emergency" if raw_d < emg_thresh else ("caution" if raw_d < 2.2 else "safe"))
            obs_payload = [
                {
                    "track_id": o.track_id,
                    "class_name": o.class_name,
                    "confidence": round(float(o.confidence), 2),
                    "bbox": list(o.bbox),
                    "distance_m": round(float(o.distance_m), 2),
                    "lane": o.lane,
                    "risk_score": round(float(o.risk_score), 2),
                    "ttc_seconds": round(float(o.ttc_seconds), 1),
                    "is_ultrasonic_target": bool(getattr(o, "is_ultrasonic_target", False)),
                    "is_emergency": bool(o.is_emergency),
                }
                for o in fused_results[:12]
            ]
            audio.update_telemetry({
                "device_name": dev_name,
                "emergency": bool(is_any_emergency),
                "camera": {
                    "connected": bool(camera_connected),
                    "error": camera_error,
                },
                "ultrasonic": {
                    "connected": bool(us_connected),
                    "distance_m": round(float(raw_d), 2) if raw_d is not None else None,
                    "beam_angle_deg": float(us_cfg.get("beam_angle_deg", 30.0)),
                    "emergency_distance_m": emg_thresh,
                    "status": us_status,
                    "error": us_error,
                },
                "gps": {
                    "connected": False,
                    "lat": None,
                    "lng": None,
                    "heading_deg": None,
                    "speed_mps": None,
                    "satellites": 0,
                    "hdop": None,
                    "home_distance_m": None,
                    "trail": [],
                    "error": "Chưa có GPS",
                },
                "system": {
                    "battery_pct": None,
                    "cpu_temp_c": read_cpu_temp(),
                    "fps": round(fps, 1) if camera_connected else 0.0,
                },
                "obstacles": obs_payload,
            })

            # Hiển thị cửa sổ Desktop và nhận phím bấm mô phỏng (chỉ bật phím giả lập khi --source demo)
            if debug_view:
                cv2.imshow("PBL6 - Smart Blind Navigation", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif is_demo and key == ord('1'):
                    sensor.set_simulated_distance(3.0, manual=True)
                elif is_demo and key == ord('2'):
                    sensor.set_simulated_distance(2.0, manual=True)
                elif is_demo and key == ord('3'):
                    sensor.set_simulated_distance(0.5, manual=True)
                elif is_demo and key == ord('4'):
                    sensor.set_simulated_distance(0.7, manual=True)
            else:
                time.sleep(max(0.001, 0.033 - t_elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Đang đóng hệ thống an toàn...")
        audio_bridge.stop()
        audio.stop()
        sensor.stop()
        if fresh_cap:
            fresh_cap.release()
        elif cap and cap.isOpened():
            cap.release()
        if debug_view and HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[DONE] Đã giải phóng toàn bộ tài nguyên.")


if __name__ == "__main__":
    main()
