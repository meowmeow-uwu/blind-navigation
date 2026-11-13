"""
record_scenarios.py - Công cụ quay & thu thập dữ liệu video thực nghiệm Object Tracking (PBL6)
=================================================================================================
Quay 1 lần các đoạn video chuẩn (640x480 @ 15 FPS) theo từng kịch bản và lần chạy (run 1, 2, 3...)
để tất cả 4 cấu hình (Raw YOLO, SORT, DeepSORT, Edge ByteTrack) chạy Replay công bằng 100%.

Tập trung hoàn toàn vào bài toán Thị giác máy tính & Bám vết (Object Tracking),
không phụ thuộc và không yêu cầu cảm biến siêu âm.

Cách dùng cơ bản:
  # Quay Kịch bản 1, lần chạy 1, thời lượng 25 giây:
  python record_scenarios.py --scenario scen1_static --run 1 --duration 25 --source auto

  # Quay Kịch bản 1, lần chạy 2:
  python record_scenarios.py --scenario scen1_static --run 2 --duration 25 --source auto

  # Quay tuần tự tất cả 12 kịch bản (mỗi kịch bản 3 lượt lặp):
  python record_scenarios.py --all --runs 3 --duration 25 --source auto
"""

import os
import sys
import csv
import time
import platform
import argparse
from pathlib import Path
from typing import Optional

import cv2

EDGE_ROOT = Path(__file__).resolve().parents[2]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

# Danh mục 12 Kịch bản Thực nghiệm Chuyên sâu cho Object Tracking (Đánh số tuần tự 1 -> 12)
SCENARIOS = {
    # Nhóm I: Cơ bản & Đường đi đơn giản
    "scen1_static": "Kịch bản 1: Đi thẳng qua hành lang có một vật cản đứng yên (ghế tựa ở 3.5m)",
    "scen2_multi": "Kịch bản 2: Đi qua nhiều vật cản cách nhau so le (ghế trái 2.0m, thùng phải 3.5m)",
    "scen3_empty": "Kịch bản 3: Hành lang hoàn toàn trống không có vật cản (kiểm tra cảnh báo nhầm)",

    # Nhóm II: Che khuất & Động lực học đối tượng
    "scen4_occlusion": "Kịch bản 4: Có người đi ngang qua camera / che khuất ngắn hạn (1.0-1.5s)",
    "scen5_cross_paths": "Kịch bản 5: Hai người đi bộ ngược chiều nhau đan chéo cắt mặt trước camera",
    "scen6_reenter": "Kịch bản 6: Đối tượng rời khung hình rồi xuất hiện trở lại sau 1-2 giây",

    # Nhóm III: Chuyển động người đeo & Rung lắc camera (Ego-motion)
    "scen7_edge_to_center": "Kịch bản 7: Vật thể ở gần rìa ảnh (góc 25-30 độ) rồi đi vào vùng giữa",
    "scen8_shake": "Kịch bản 8: Camera rung lắc khi người đeo bước đi nhanh, dằn gót",
    "scen9_fast_approach": "Kịch bản 9: Vật cản tiếp cận nhanh về phía camera (người đi nhanh/xe đẩy)",
    "scen10_turn_corner": "Kịch bản 10: Rẽ góc ngã rẽ hành lang 90 độ, vật cản bất ngờ xuất hiện",

    # Nhóm IV: Điều kiện môi trường & Mật độ phức tạp
    "scen11_lighting_change": "Kịch bản 11: Thay đổi ánh sáng đột ngột (từ hành lang tối ra vùng cửa sổ nắng gắt)",
    "scen12_dense_clutter": "Kịch bản 12: Môi trường đông đúc hỗn hợp nhiều đồ vật (người, ghế, thùng rác, ba lô)",
}


def _open_video_source(source: str, target_fps: float):
    """Mở camera vật lý ('auto', '0', '1'...) hoặc tệp video có sẵn để chuẩn hóa 640x480."""
    src_str = str(source).strip()
    if os.path.isfile(src_str):
        cap = cv2.VideoCapture(src_str)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở tệp video đầu vào: {src_str}")
        return cap, True

    cam_indices = [0, 1, 2, 3, 4] if src_str.lower() == "auto" else [int(src_str)]
    is_linux = (platform.system() == "Linux")
    backend = cv2.CAP_V4L2 if (is_linux and hasattr(cv2, "CAP_V4L2")) else cv2.CAP_ANY

    for idx in cam_indices:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"📷 Đã kết nối Camera vật lý tại index={idx} (Backend: {'V4L2' if is_linux else platform.system()})")
                return cap, False
            cap.release()

    raise RuntimeError(
        f"Không thể mở camera tại nguồn source='{source}'. "
        "Trên Raspberry Pi 4 hãy kiểm tra `v4l2-ctl --list-devices`; "
        "trên macOS hãy cấp quyền Camera cho Terminal/IDE trong System Settings."
    )


def record_single_scenario(
    scenario_id: str,
    out_dir: Path,
    source: str,
    duration_s: float,
    target_fps: float,
    headless: bool,
    run_idx: Optional[int] = None,
    countdown_s: int = 5,
    with_ultrasonic: bool = False,
):
    """Quay một video kịch bản chuẩn cho Object Tracking Benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"{scenario_id}_run{run_idx}" if run_idx is not None else scenario_id
    video_path = out_dir / f"{file_stem}.mp4"
    us_csv_path = out_dir / f"{file_stem}_ultrasonic.csv"

    print("\n" + "=" * 80)
    print(f"🎬 CHUẨN BỊ QUAY: {SCENARIOS.get(scenario_id, scenario_id)}")
    if run_idx is not None:
        print(f"   Lần chạy (Run): #{run_idx}")
    print(f"   Lưu video: {video_path.resolve()}")
    print(f"   Thời lượng mục tiêu: {duration_s:.0f} giây @ {target_fps:.0f} FPS (~{int(duration_s * target_fps)} frames)")
    print("=" * 80)

    cap, is_file_input = _open_video_source(source, target_fps)

    sensor = None
    if with_ultrasonic:
        try:
            from modules.sensors.ultrasonic_hcsr04 import HCSR04Sensor
            sensor = HCSR04Sensor(trig_pin=23, echo_pin=24, poll_rate_hz=40, emergency_distance_m=1.0)
            sensor.start()
            if sensor.is_simulated:
                sensor.set_simulated_distance(2.2, manual=False)
        except Exception as e:
            print(f"⚠️  Không thể khởi tạo cảm biến siêu âm ({e}), tiếp tục chỉ quay video.")
            sensor = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, target_fps, (640, 480))
    if not writer.isOpened():
        cap.release()
        if sensor:
            sensor.stop()
        raise RuntimeError(f"Không thể tạo tệp VideoWriter tại {video_path}")

    if not is_file_input and countdown_s > 0:
        print(f"⏳ Bắt đầu ghi sau {countdown_s} giây... Người thử nghiệm vào vị trí xuất phát!")
        for sec in range(countdown_s, 0, -1):
            print(f"   {sec}...")
            time.sleep(1.0)
    print("🔴 ĐANG GHI HÌNH! (Nhấn 'Q' để dừng sớm hoặc đợi tự ngắt)")

    start_t = time.perf_counter()
    elapsed = 0.0
    frame_id = 0
    src_idx = 0
    us_rows = []
    frame_interval = 1.0 / target_fps
    src_fps = cap.get(cv2.CAP_PROP_FPS) if is_file_input else target_fps
    if not src_fps or src_fps <= 1e-3 or src_fps > 240:
        src_fps = target_fps
    next_emit_t = 0.0

    try:
        while True:
            loop_t = time.perf_counter()
            if not is_file_input and (loop_t - start_t) >= duration_s:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_file_input:
                    break
                continue

            if is_file_input:
                src_t = src_idx / src_fps
                src_idx += 1
                if src_t + 1e-6 < next_emit_t:
                    continue
                next_emit_t += frame_interval
                elapsed = frame_id * frame_interval
            else:
                elapsed = loop_t - start_t

            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            writer.write(frame)

            raw_d = None
            if sensor:
                raw_d, _, is_emer = sensor.get_latest_reading()
                us_rows.append({
                    "frame_id": frame_id,
                    "timestamp": round(loop_t, 6),
                    "elapsed_s": round(elapsed, 4),
                    "us_distance_m": round(raw_d, 4) if raw_d is not None else "",
                    "is_sensor_emergency": int(bool(is_emer)),
                })

            if not headless:
                hud = frame.copy()
                run_tag = f" | Run:{run_idx}" if run_idx is not None else ""
                cv2.putText(
                    hud,
                    f"REC {scenario_id}{run_tag} | F:{frame_id} | {elapsed:.1f}/{duration_s:.0f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
                )
                cv2.imshow("PBL6 Tracking Video Recorder", hud)
                if (cv2.waitKey(1) & 0xFF) in [ord("q"), 27]:
                    break

            frame_id += 1
            if not is_file_input:
                sleep_rem = frame_interval - (time.perf_counter() - loop_t)
                if sleep_rem > 0:
                    time.sleep(sleep_rem)
    finally:
        writer.release()
        cap.release()
        if sensor:
            sensor.stop()
        if not headless:
            cv2.destroyAllWindows()

    if frame_id == 0:
        raise RuntimeError(f"Không ghi được khung hình nào từ source='{source}'!")

    if with_ultrasonic and us_rows:
        with open(us_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame_id", "timestamp", "elapsed_s", "us_distance_m", "is_sensor_emergency"]
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(us_rows)

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"✅ Hoàn thành: Đã lưu {frame_id} frames ({elapsed:.2f}s, {size_mb:.2f} MB) -> {video_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Quay video kịch bản chuẩn phục vụ Benchmark Object Tracking (PBL6)")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()), help="Mã kịch bản cần quay")
    parser.add_argument("--run", type=int, default=None, help="Số thứ tự lần chạy (ví dụ: --run 1, --run 2, --run 3)")
    parser.add_argument("--runs", type=int, default=3, help="Số lần chạy lặp khi quay toàn bộ (--all)")
    parser.add_argument("--all", action="store_true", help="Quay tuần tự tất cả các kịch bản")
    parser.add_argument("--source", type=str, default="auto", help="Nguồn camera ('auto', index '0'/'1', hoặc file video)")
    parser.add_argument("--duration", type=float, default=25.0, help="Thời lượng mỗi video (giây, khuyến nghị: 25.0s)")
    parser.add_argument("--fps", type=float, default=15.0, help="Tốc độ khung hình ghi (FPS, chuẩn thiết bị: 15.0)")
    parser.add_argument("--countdown", type=int, default=5, help="Số giây đếm ngược chuẩn bị trước khi quay")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "data/videos"))
    parser.add_argument("--headless", action="store_true", help="Chạy không hiển thị cửa sổ xem trước (phù hợp khi SSH không có GUI)")
    parser.add_argument("--with-ultrasonic", action="store_true", help="Kèm theo cảm biến siêu âm (tùy chọn, mặc định tắt cho Object Tracking)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.all:
        print(f"📋 BẮT ĐẦU CHẾ ĐỘ QUAY TOÀN BỘ ({len(SCENARIOS)} kịch bản x {args.runs} lần chạy lặp):")
        for sc in SCENARIOS:
            for r in range(1, args.runs + 1):
                input(f"\n👉 SẴN SÀNG: [{sc}] (Lần {r}/{args.runs}) - '{SCENARIOS[sc]}'\n   Nhấn [Enter] để đếm ngược {args.countdown}s và quay...")
                record_single_scenario(
                    scenario_id=sc,
                    out_dir=out_dir,
                    source=args.source,
                    duration_s=args.duration,
                    target_fps=args.fps,
                    headless=args.headless,
                    run_idx=r,
                    countdown_s=args.countdown,
                    with_ultrasonic=args.with_ultrasonic,
                )
    elif args.scenario:
        record_single_scenario(
            scenario_id=args.scenario,
            out_dir=out_dir,
            source=args.source,
            duration_s=args.duration,
            target_fps=args.fps,
            headless=args.headless,
            run_idx=args.run,
            countdown_s=args.countdown,
            with_ultrasonic=args.with_ultrasonic,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
