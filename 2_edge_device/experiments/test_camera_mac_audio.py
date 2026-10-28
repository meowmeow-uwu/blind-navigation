"""
test_camera_mac_audio.py - Kiểm tra Camera thật & Phát âm thanh qua Loa máy Mac
=================================================================================
Mục đích:
- Kiểm tra Camera vật lý (USB Webcam hoặc CSI Camera) trên Raspberry Pi 4.
- Tự động dò tìm cổng Camera khả dụng (index 0, 1, 2, 3...).
- Truyền hình ảnh HUD thời gian thực và giọng nói cảnh báo lên trình duyệt máy Mac.
- Không cần cắm cảm biến siêu âm (chạy độc lập Camera + Loa máy Mac).
"""

import os
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("❌ Lỗi: Cần cài đặt opencv-python hoặc opencv-python-headless.")
    sys.exit(1)

import numpy as np
from modules.tracking.byte_tracker import LightweightTracker
from modules.audio.audio_manager import AudioManager
from modules.vision_local.freshest_camera import FreshestCameraReader


def scan_camera(source_arg="auto"):
    """Dò tìm và mở camera vật lý khả dụng."""
    if source_arg.isdigit():
        idx = int(source_arg)
        backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, idx
            cap.release()
        return None, -1

    # Tự động quét từ 0 đến 4
    for idx in [0, 1, 2, 3, 4]:
        try:
            backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, idx
                cap.release()
        except Exception:
            pass
    return None, -1


def init_detector():
    param_path = BASE_DIR / "modules/vision_local/models/yolov8n.param"
    bin_path = BASE_DIR / "modules/vision_local/models/yolov8n.bin"
    try:
        from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
        if param_path.exists() and bin_path.exists():
            print("  ✅ Đã tải mô hình YOLOv8 NCNN thành công!")
            return YoloNcnnRunner(str(param_path), str(bin_path), target_size=320)
    except Exception:
        pass

    # Simple detector fallback
    print("  ℹ️ Sử dụng Simple Color Detector (hoặc mô phỏng phát hiện) để kiểm tra luồng camera...")
    class SimpleDetector:
        def detect(self, frame):
            # Nhận diện cơ bản vùng chuyển động hoặc bbox ở trung tâm
            h, w = frame.shape[:2]
            return [{
                'class_name': 'person',
                'bbox': [int(w * 0.3), int(h * 0.2), int(w * 0.7), int(h * 0.85)],
                'confidence': 0.85
            }]
    return SimpleDetector()


def main():
    parser = argparse.ArgumentParser(description="Test Camera & Mac Audio Stream")
    parser.add_argument("--source", type=str, default="auto", help="Chỉ số camera (0, 1, 2 hoặc 'auto')")
    parser.add_argument("--port", type=int, default=8080, help="Cổng Web Stream (mặc định: 8080)")
    args = parser.parse_args()

    print("=" * 66)
    print("   KIỂM TRA CAMERA THẬT & PHÁT ÂM THANH QUA MÁY MAC (PBL6)   ")
    print("   (Chế độ độc lập: Không cần cảm biến siêu âm)            ")
    print("=" * 66)

    # 1. Dò tìm camera
    print("\n[1/3] Đang dò tìm Camera vật lý...")
    cap, cam_idx = scan_camera(args.source)
    if cap is None:
        print("  ❌ Không tìm thấy thiết bị Camera nào hoạt động!")
        print("  💡 Hướng dẫn kiểm tra:")
        print("     1. Nếu dùng USB Webcam: Rút ra cắm lại cổng USB 3.0 màu xanh.")
        print("     2. Chạy lệnh: v4l2-ctl --list-devices để xem danh sách cổng.")
        print("     3. Nếu dùng CSI Camera trên Pi: chạy lệnh `rpicam-hello` xem cáp có lỏng không.")
        sys.exit(1)

    print(f"  ✅ Đã kết nối thành công Camera tại cổng index: {cam_idx}")
    fresh_cap = FreshestCameraReader(cap)

    # 2. Khởi tạo detector & tracker
    print("\n[2/3] Khởi tạo AI Detector & Audio Manager...")
    detector = init_detector()
    tracker = LightweightTracker(max_age=15, min_hits=2)
    # Tắt loa cục bộ trên Pi để không bị rè, chỉ stream lên Web/Mac
    audio = AudioManager(port=args.port, guidance_cooldown_s=2.5, enable_local_audio=False)

    print(f"\n[3/3] Hệ thống đã sẵn sàng!")
    print("=" * 66)
    print(f" 👉 TRÊN MÁY TÍNH MAC: Mở Safari / Chrome truy cập:")
    print(f"    http://<IP_CỦA_RASPBERRY_PI>:{audio.port}")
    print(f" 👉 Nhấn nút: [ 🔊 BẬT ÂM THANH TRÌNH DUYỆT ]")
    print(f" 👉 Đứng trước camera: Hình ảnh truyền về Mac và Loa Mac sẽ đọc!")
    print(" 👉 Nhấn Ctrl + C để dừng.")
    print("=" * 66 + "\n")

    fps = 0.0
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = fresh_cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Nhận diện & theo dõi
            dets = detector.detect(frame)
            tracks = tracker.update(dets)

            # Vẽ HUD lên ảnh
            third = w // 3
            cv2.line(frame, (third, 0), (third, h), (80, 80, 80), 1)
            cv2.line(frame, (2 * third, 0), (2 * third, h), (80, 80, 80), 1)

            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                cx = (x1 + x2) / 2
                lane = "left" if cx < third else ("center" if cx < 2 * third else "right")
                lane_vn = "trước mặt" if lane == "center" else ("bên trái" if lane == "left" else "bên phải")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"#{t.track_id} {t.class_name.upper()} ({lane_vn})"
                cv2.putText(frame, tag, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Phát cảnh báo âm thanh tới Mac đúng 1 lần cho mỗi Track ID
                if not t.announced:
                    spoken = audio.announce_guidance(t.track_id, t.class_name, lane, distance_m=2.0)
                    if spoken:
                        t.mark_announced(2.0)

            # Header trạng thái
            t_el = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_el))
            cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.putText(frame, f"CAM INDEX: {cam_idx} | FPS: {fps:.1f} | VAT THE: {len(tracks)}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Cập nhật frame lên Web Dashboard cho máy Mac
            audio.update_frame(frame)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[INFO] Đang dừng...")
    finally:
        fresh_cap.release()
        audio.stop()
        print("[DONE] Đã giải phóng toàn bộ tài nguyên.")


if __name__ == "__main__":
    main()

