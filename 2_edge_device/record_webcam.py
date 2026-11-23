#!/usr/bin/env python3
"""
record_webcam.py - Quay video từ USB Webcam trên Raspberry Pi
============================================================
Cách dùng:
  # Quay video 10 giây rồi tự dừng và lưu file output.mp4:
  python3 record_webcam.py --duration 10 --output test_video.mp4

  # Quay video liên tục cho đến khi nhấn Ctrl+C hoặc phím 'q':
  python3 record_webcam.py --duration 0 --output output.mp4

  # Quay video độ phân giải HD 720p:
  python3 record_webcam.py --width 1280 --height 720 --output hd_video.mp4
"""

import os
import sys
import time
import argparse
from datetime import datetime

try:
    import cv2
except ImportError:
    print("[ERROR] Chưa cài đặt OpenCV (cv2)!")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quay video từ USB Webcam")
    parser.add_argument("-i", "--index", type=int, default=0, help="Camera index (/dev/videoX)")
    parser.add_argument("-W", "--width", type=int, default=640, help="Độ rộng khung hình (mặc định: 640)")
    parser.add_argument("-H", "--height", type=int, default=480, help="Độ cao khung hình (mặc định: 480)")
    parser.add_argument("-f", "--fps", type=int, default=20, help="FPS của video ghi (mặc định: 20)")
    parser.add_argument("-d", "--duration", type=int, default=10,
                        help="Thời lượng quay tính bằng giây (đặt 0 để quay liên tục tới khi bấm Ctrl+C)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Tên file video lưu (mặc định tự đặt tên theo ngày giờ)")
    parser.add_argument("--codec", type=str, default="mp4v", choices=["mp4v", "XVID", "MJPG"],
                        help="Codec video: mp4v (.mp4), XVID (.avi), MJPG (.avi)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Tự động tạo tên file nếu chưa truyền
    if not args.output:
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".mp4" if args.codec == "mp4v" else ".avi"
        output_file = os.path.join("recordings", f"rec_{timestamp}{ext}")
    else:
        output_file = args.output
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("        QUAY VIDEO USB WEBCAM TRÊN RASPBERRY PI")
    print("=" * 65)
    print(f"[*] Đang mở camera /dev/video{args.index}...")

    cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở camera index {args.index}!")
        print("-> Hãy kiểm tra xem có tiến trình nào đang chiếm camera không (vd: stream_webcam).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(output_file, fourcc, args.fps, (actual_w, actual_h))

    if not writer.isOpened():
        print(f"[ERROR] Không thể khởi tạo VideoWriter cho file: {output_file}")
        cap.release()
        sys.exit(1)

    dur_str = f"{args.duration} giây" if args.duration > 0 else "Liên tục (bấm Ctrl+C để dừng)"
    print(f"[OK] Thiết bị sẵn sàng:")
    print(f"  - Độ phân giải: {actual_w}x{actual_h}")
    print(f"  - Tốc độ ghi   : {args.fps} FPS (Codec: {args.codec})")
    print(f"  - Thời lượng   : {dur_str}")
    print(f"  - File xuất    : {os.path.abspath(output_file)}")
    print("-" * 65)
    print("🔴 ĐANG QUAY VIDEO... (Nhấn Ctrl+C để dừng và lưu)")

    start_time = time.time()
    frames_recorded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            writer.write(frame)
            frames_recorded += 1

            elapsed = time.time() - start_time
            current_fps = frames_recorded / elapsed if elapsed > 0 else 0
            
            # Hiển thị tiến trình trên cùng 1 dòng
            print(f"  [REC] Thời gian: {elapsed:5.1f}s | Đã ghi: {frames_recorded:4d} frames | FPS: {current_fps:4.1f}", end="\r", flush=True)

            if args.duration > 0 and elapsed >= args.duration:
                print()
                break

    except KeyboardInterrupt:
        print("\n-> Nhận tín hiệu dừng từ bàn phím (Ctrl+C).")

    finally:
        total_time = time.time() - start_time
        cap.release()
        writer.release()

    # Báo cáo kết quả
    avg_fps = frames_recorded / total_time if total_time > 0 else 0
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024) if os.path.exists(output_file) else 0

    print()
    print("=" * 65)
    print("                KẾT QUẢ GHI HÌNH")
    print("=" * 65)
    print(f"  * Trạng thái        : THÀNH CÔNG (Đã lưu)")
    print(f"  * File video        : {os.path.abspath(output_file)}")
    print(f"  * Dung lượng        : {file_size_mb:.2f} MB")
    print(f"  * Thời lượng thực tế: {total_time:.1f} giây")
    print(f"  * Tổng số khung hình: {frames_recorded} frames ({avg_fps:.1f} FPS)")
    print("=" * 65)


if __name__ == "__main__":
    main()
