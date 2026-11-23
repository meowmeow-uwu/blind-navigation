#!/usr/bin/env python3
"""
test_webcam.py - Kiểm tra USB Webcam trên Raspberry Pi
======================================================
Hỗ trợ:
- Tự động nhận diện thiết bị camera USB (/dev/video*).
- Chạy mượt mà ở cả chế độ SSH / Headless (lưu ảnh mẫu) lẫn Desktop GUI (hiển thị cửa sổ).
- Đo FPS thực tế, độ trễ và tỷ lệ nhận frame.
- Lưu ảnh chụp thử nghiệm kèm thông số kỹ thuật (độ phân giải, timestamp, FPS).

Cách dùng cơ bản:
  python3 test_webcam.py
  python3 test_webcam.py --index 0 --width 1280 --height 720
  python3 test_webcam.py --gui       # (Nếu có cắm màn hình hoặc qua VNC/X11)
  python3 test_webcam.py --count 0   # Chạy liên tục đến khi bấm Ctrl+C
"""

import os
import sys
import time
import glob
import argparse
from datetime import datetime

try:
    import cv2
except ImportError:
    print("[ERROR] Chưa cài đặt OpenCV (cv2)!")
    print("Vui lòng kích hoạt venv hoặc cài đặt:")
    print("  source .venv/bin/activate  (nếu dùng virtualenv)")
    print("  hoặc: pip install opencv-python")
    sys.exit(1)


def scan_v4l2_cameras():
    """Liệt kê các camera khả dụng từ hệ thống V4L2 (/sys/class/video4linux)"""
    devices = []
    dev_paths = sorted(glob.glob("/sys/class/video4linux/video*"))
    for dev_path in dev_paths:
        dev_name = os.path.basename(dev_path)
        name_file = os.path.join(dev_path, "name")
        index = int(dev_name.replace("video", ""))
        camera_name = "Unknown"
        if os.path.exists(name_file):
            with open(name_file, "r") as f:
                camera_name = f.read().strip()
        devices.append((index, f"/dev/{dev_name}", camera_name))
    return devices


def parse_arguments():
    parser = argparse.ArgumentParser(description="Kiểm tra USB Webcam trên Raspberry Pi")
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="Chỉ số camera device (mặc định: 0 ứng với /dev/video0)")
    parser.add_argument("-W", "--width", type=int, default=640,
                        help="Độ rộng khung hình (mặc định: 640)")
    parser.add_argument("-H", "--height", type=int, default=480,
                        help="Độ cao khung hình (mặc định: 480)")
    parser.add_argument("-f", "--fps", type=int, default=30,
                        help="FPS yêu cầu thiết lập (mặc định: 30)")
    parser.add_argument("--codec", type=str, default="MJPG", choices=["MJPG", "YUYV"],
                        help="Codec format: MJPG (khuyên dùng trên USB cam để đạt 30 FPS ở HD/FullHD) hoặc YUYV")
    parser.add_argument("-c", "--count", type=int, default=60,
                        help="Số lượng frame cần test (mặc định: 60 frame, đặt 0 để chạy liên tục)")
    parser.add_argument("-s", "--save", type=str, default="webcam_test.jpg",
                        help="Đường dẫn file ảnh chụp mẫu cần lưu (mặc định: webcam_test.jpg)")
    parser.add_argument("--gui", action="store_true",
                        help="Mở cửa sổ hiển thị video OpenCV (cần môi trường Desktop/X11)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 65)
    print("      KIỂM TRA USB WEBCAM TRÊN RASPBERRY PI")
    print("=" * 65)

    # 1. Quét danh sách thiết bị
    print("[1] Danh sách thiết bị video phát hiện được:")
    cams = scan_v4l2_cameras()
    if not cams:
        print("  - Không tìm thấy thiết bị /dev/video* nào!")
    else:
        for idx, dev_node, name in cams:
            marker = " -> [ĐANG CHỌN]" if idx == args.index else ""
            print(f"  * Index {idx:2d} | {dev_node} | {name}{marker}")
    print()

    # 2. Kiểm tra môi trường hiển thị GUI
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_gui = args.gui and has_display
    if args.gui and not has_display:
        print("[LƯU Ý] Không phát hiện biến môi trường DISPLAY/X11 (bạn đang dùng SSH/Terminal).")
        print("        Chuyển tự động sang chế độ Headless (CLI + lưu ảnh).")

    # 3. Khởi tạo Camera bằng V4L2 Backend
    print(f"[2] Đang mở Camera Index {args.index} (/dev/video{args.index}) với V4L2...")
    cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[ERROR] Không thể kết nối tới camera tại index {args.index}!")
        print("Gợi ý khắc phục:")
        print("  1. Kiểm tra lại cáp cắm USB camera.")
        print("  2. Kiểm tra quyền truy cập: sudo usermod -aG video $USER")
        print("  3. Thử đổi index khác (vd: --index 1, --index 2)")
        sys.exit(1)

    # Cấu hình Codec, Resolution, FPS
    if args.codec == "MJPG":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif args.codec == "YUYV":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Đọc lại thông số thực tế nhận được từ driver phần cứng
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

    print(f"  -> Trạng thái: KẾT NỐI THÀNH CÔNG")
    print(f"  -> Độ phân giải thực tế: {actual_w}x{actual_h}")
    print(f"  -> Tốc độ camera khai báo: {actual_fps:.1f} FPS")
    print(f"  -> Định dạng Codec: {fourcc_str}")
    print()

    # 4. Bắt đầu đọc stream và đo kiểm
    mode_str = "GUI Window (nhấn 'q' để dừng)" if use_gui else "Headless (nhấn Ctrl+C để dừng)"
    target_count_str = f"{args.count} frame" if args.count > 0 else "Liên tục"
    print(f"[3] Bắt đầu kiểm tra luồng video:")
    print(f"  - Chế độ: {mode_str}")
    print(f"  - Số frame test: {target_count_str}")
    print("-" * 65)

    saved_sample = False
    success_frames = 0
    failed_frames = 0
    start_time = time.time()
    last_stat_time = start_time
    frames_in_second = 0

    try:
        while True:
            ret, frame = cap.read()
            now = time.time()

            if not ret or frame is None:
                failed_frames += 1
                print(f"[CẢNH BÁO] Lỗi đọc frame ({failed_frames} frame hỏng)", end="\r")
                time.sleep(0.01)
                continue

            success_frames += 1
            frames_in_second += 1

            # Lưu ảnh mẫu đầu tiên để người dùng kiểm tra góc nhìn & ánh sáng
            if not saved_sample and args.save:
                sample_img = frame.copy()
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                overlay_text = f"{timestamp_str} | {actual_w}x{actual_h} | cam:{args.index}"
                cv2.putText(sample_img, overlay_text, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(args.save, sample_img)
                saved_sample = True

            # Cập nhật thống kê FPS mỗi giây
            if now - last_stat_time >= 1.0:
                current_fps = frames_in_second / (now - last_stat_time)
                elapsed = now - start_time
                status_line = (f"  [ĐANG CHẠY] Đã nhận: {success_frames:4d} frames | "
                               f"FPS hiện tại: {current_fps:5.1f} | "
                               f"Thời gian: {elapsed:4.1f}s")
                print(status_line, end="\r", flush=True)
                frames_in_second = 0
                last_stat_time = now

            # Hiển thị nếu bật GUI
            if use_gui:
                cv2.imshow("Raspberry Pi - USB Webcam Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n-> Người dùng nhấn 'q' để dừng.")
                    break

            # Kiểm tra nếu đủ số frame yêu cầu
            if args.count > 0 and success_frames >= args.count:
                print()
                break

    except KeyboardInterrupt:
        print("\n-> Nhận tín hiệu dừng từ bàn phím (Ctrl+C).")

    finally:
        total_time = time.time() - start_time
        cap.release()
        if use_gui:
            cv2.destroyAllWindows()

    # 5. Báo cáo kết quả
    avg_fps = (success_frames / total_time) if total_time > 0 else 0
    print()
    print("=" * 65)
    print("                 KẾT QUẢ ĐO KIỂM")
    print("=" * 65)
    print(f"  * Tổng frame thành công : {success_frames}")
    print(f"  * Tổng frame lỗi        : {failed_frames}")
    print(f"  * Tổng thời gian        : {total_time:.2f} giây")
    print(f"  * FPS thực tế trung bình: {avg_fps:.2f} FPS")

    if saved_sample and os.path.exists(args.save):
        abs_path = os.path.abspath(args.save)
        file_size_kb = os.path.getsize(abs_path) / 1024
        print(f"  * Ảnh chụp thử nghiệm   : {abs_path} ({file_size_kb:.1f} KB)")
        print(f"    (Bạn có thể tải hoặc mở file ảnh này để kiểm tra góc quay/độ nét)")

    if success_frames > 0 and failed_frames == 0:
        print("\n=> [ĐÁNH GIÁ]: WEBCAM HOẠT ĐỘNG HOÀN HẢO! SẴN SÀNG CHO HỆ THỐNG.")
    elif success_frames > 0:
        print("\n=> [ĐÁNH GIÁ]: WEBCAM HOẠT ĐỘNG (có một số frame bị trễ hoặc rớt).")
    else:
        print("\n=> [ĐÁNH GIÁ]: THẤT BẠI - Không nhận được khung hình nào từ webcam!")
    print("=" * 65)


if __name__ == "__main__":
    main()
