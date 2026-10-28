"""
test_ultrasonic.py - Chương trình độc lập kiểm tra Cảm biến siêu âm HC-SR04
=============================================================================
Mục đích:
- Kiểm tra kết nối phần cứng của cảm biến siêu âm HC-SR04 trên Raspberry Pi 4.
- Hiển thị khoảng cách thời gian thực (cm / mét) và thước đo trực quan (Progress Bar).
- Phân loại 3 vùng cự ly: KHẨN CẤP (<1m), DẪN HƯỚNG (1.5m - 3m), AN TOÀN (>3m).
- Chẩn đoán chi tiết nếu gặp sự cố (chân chưa cắm, dây lỏng, cầu phân áp trở sai).
- Tùy chọn còi bíp: Phát tiếng bíp với tần số tăng dần khi đưa tay lại gần!
"""

import sys
import time
import math
import wave
import struct
import shutil
import argparse
import subprocess
import platform

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except (ImportError, RuntimeError):
    HAS_RPI_GPIO = False


# Sơ đồ chân mặc định chuẩn PBL6 trên Raspberry Pi 4
DEFAULT_TRIG_PIN = 23   # Pin vật lý 16
DEFAULT_ECHO_PIN = 24   # Pin vật lý 18 (Qua cầu phân áp trở 1k / 2k xuống 3.3V)


WIRING_DIAGRAM = """
======================================================================
           SƠ ĐỒ ĐẤU NỐI CHÂN HC-SR04 VÀO RASPBERRY PI 4
======================================================================
  Cảm biến HC-SR04                         Raspberry Pi 4
  +---------------+                        +----------------------+
  |  VCC  (5V)    | ---------------------> | Pin 2  hoặc Pin 4 (5V)
  |  TRIG (Kích)  | ---------------------> | Pin 16 (GPIO 23)     |
  |  ECHO (Hồi)   | ---[ Trở 1kΩ ]---+---> | Pin 18 (GPIO 24)     |
  |               |                  |     |                      |
  |  GND  (Mass)  | --+--[ Trở 2kΩ ]-+---> | Pin 6  (GND)         |
  +---------------+   |                    +----------------------+
                      +------------------> (Nối chung vào Pin 6 GND)

  ⚠️ LƯU Ý BẮT BUỘC VỀ ĐIỆN ÁP:
  - Chân ECHO phát ra 5V, chân GPIO của Pi chỉ chịu được 3.3V.
  - Bắt buộc dùng 2 điện trở (1kΩ và 2kΩ) làm cầu phân áp như sơ đồ trên.
======================================================================
"""


def create_beep_wav(filename: str = "/tmp/test_us_beep.wav", freq: float = 1400.0, duration: float = 0.05):
    """Tạo file wav tiếng bíp 50ms cho loa máy."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            val = int(22000.0 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            f.writeframes(struct.pack("<h", val))


def play_beep():
    """Phát 1 tiếng bíp ngắn."""
    beep_file = "/tmp/test_us_beep.wav"
    if IS_MAC:
        subprocess.run(["afplay", "-q", "1", beep_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    elif IS_LINUX:
        if shutil.which("aplay"):
            subprocess.run(["aplay", "-q", beep_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            sys.stdout.write('\a')
            sys.stdout.flush()


def measure_single_distance(trig_pin: int, echo_pin: int, timeout_s: float = 0.035):
    """
    Đo trực tiếp 1 lần xung siêu âm từ phần cứng.
    Returns:
        (distance_m: Optional[float], error_msg: Optional[str])
    """
    if not HAS_RPI_GPIO:
        return None, "NO_GPIO"

    try:
        # 1. Đảm bảo chân Trig LOW trước khi phát
        GPIO.output(trig_pin, GPIO.LOW)
        time.sleep(0.000002)

        # 2. Bơm xung 10 micro-giây
        GPIO.output(trig_pin, GPIO.HIGH)
        time.sleep(0.000010)
        GPIO.output(trig_pin, GPIO.LOW)

        t_start = time.perf_counter()
        timeout_wait_start = t_start + timeout_s

        # 3. Chờ chân Echo lên mức HIGH (bắt đầu đo)
        pulse_start = time.perf_counter()
        while GPIO.input(echo_pin) == GPIO.LOW:
            pulse_start = time.perf_counter()
            if pulse_start > timeout_wait_start:
                return None, "LỖI: Chân ECHO không lên mức HIGH (Kiểm tra chân VCC, TRIG và cáp cắm)"

        # 4. Chờ chân Echo xuống mức LOW (sóng phản xạ dội về)
        pulse_end = pulse_start
        timeout_wait_end = pulse_start + timeout_s
        while GPIO.input(echo_pin) == GPIO.HIGH:
            pulse_end = time.perf_counter()
            if pulse_end > timeout_wait_end:
                return None, "LỖI: Chân ECHO bị kẹt mức HIGH (Kiểm tra chân GND và cầu phân áp trở)"

        # 5. Tính toán cự ly: d = (t * 343 m/s) / 2
        pulse_duration = pulse_end - pulse_start
        distance_m = (pulse_duration * 343.0) / 2.0

        if distance_m < 0.02 or distance_m > 4.5:
            return None, f"NGOÀI PHẠM VI ({distance_m*100:.1f} cm vượt quá giới hạn đo 2cm - 400cm)"

        return distance_m, None
    except Exception as e:
        return None, f"LỖI GPIO: {e}"


def get_visual_bar(dist_m: float, max_m: float = 3.5, width: int = 24) -> str:
    """Tạo thanh hiển thị trực quan cự ly."""
    fill_ratio = min(1.0, max(0.0, dist_m / max_m))
    fill_len = int(fill_ratio * width)
    empty_len = width - fill_len
    return "[" + ("█" * fill_len) + ("░" * empty_len) + "]"


def run_live_test(trig_pin: int, echo_pin: int, poll_rate_hz: int = 15, enable_beep: bool = False):
    """Vòng lặp đọc và hiển thị thước đo khoảng cách thời gian thực."""
    print("=" * 70)
    print("   KIỂM THỬ THỜI GIAN THỰC CẢM BIẾN SIÊU ÂM HC-SR04 (PBL6)   ")
    print(f"   Chân GPIO sử dụng: TRIG = {trig_pin} (Pin 16) | ECHO = {echo_pin} (Pin 18)")
    print(f"   Tần số đọc: {poll_rate_hz} Hz | Chế độ còi bíp: {'BẬT (Khẩn cấp < 1m)' if enable_beep else 'TẮT'}")
    print("=" * 70)
    print(" 👉 Đưa tay hoặc vật cản lại gần/ra xa cảm biến để quan sát cự ly.")
    print(" 👉 Nhấn phím Ctrl + C để dừng kiểm tra.\n")

    if enable_beep:
        create_beep_wav()

    # Thống kê đo đạc
    stats = {
        "total": 0,
        "success": 0,
        "min_dist": 999.0,
        "max_dist": 0.0,
        "errors": {}
    }

    last_beep_time = 0.0
    poll_interval = 1.0 / max(1, poll_rate_hz)

    try:
        while True:
            t0 = time.perf_counter()
            stats["total"] += 1

            dist_m, err = measure_single_distance(trig_pin, echo_pin)

            if dist_m is not None:
                stats["success"] += 1
                stats["min_dist"] = min(stats["min_dist"], dist_m)
                stats["max_dist"] = max(stats["max_dist"], dist_m)
                dist_cm = dist_m * 100.0

                bar = get_visual_bar(dist_m)

                # Phân loại trạng thái theo 3 vùng phản hồi của PBL6
                if dist_m < 1.0:
                    status = "🚨 [KHẨN CẤP: DỪNG LẠI!]"
                    # Phát còi bíp nếu bật cờ --beep (nhịp bíp dồn dập khi gần)
                    if enable_beep:
                        now = time.time()
                        interval = max(0.12, min(0.50, 0.12 + dist_m * 0.38))
                        if now - last_beep_time >= interval:
                            last_beep_time = now
                            play_beep()
                elif 1.5 <= dist_m <= 3.0:
                    status = "⚠️  [DẪN HƯỚNG]"
                else:
                    status = "  [AN TOÀN]"

                sys.stdout.write(f"\r📊 {dist_cm:6.1f} cm | {dist_m:4.2f} m  {bar}  {status}   ")
                sys.stdout.flush()
            else:
                stats["errors"][err] = stats["errors"].get(err, 0) + 1
                sys.stdout.write(f"\r❌ KHÔNG ĐỌC ĐƯỢC: {err[:55]}...   ")
                sys.stdout.flush()

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.001, poll_interval - elapsed))

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("BÁO CÁO THỐNG KÊ KẾT QUẢ KIỂM TRA:")
        print("=" * 70)
        total = stats["total"]
        succ = stats["success"]
        succ_rate = (succ / total * 100.0) if total > 0 else 0.0
        print(f" • Tổng số lần phát xung: {total}")
        print(f" • Số lần đo thành công:  {succ} ({succ_rate:.1f}%)")
        if succ > 0:
            print(f" • Khoảng cách gần nhất:  {stats['min_dist']*100:.1f} cm ({stats['min_dist']:.2f} m)")
            print(f" • Khoảng cách xa nhất:   {stats['max_dist']*100:.1f} cm ({stats['max_dist']:.2f} m)")
            print(" ✅ Cảm biến HC-SR04 hoạt động rất tốt và ổn định!")
        else:
            print(" ❌ Toàn bộ các lần đo đều thất bại!")
            print("    Các lỗi gặp phải:")
            for msg, cnt in stats["errors"].items():
                print(f"    - [{cnt} lần]: {msg}")
            print("\n 💡 HƯỚNG DẪN XỬ LÝ LỖI:")
            print(" 1. Kiểm tra chân VCC: Đã nối đúng Pin 2 hoặc Pin 4 (5V) chưa?")
            print(" 2. Kiểm tra chân GND: Đã nối vào Pin 6 và nối chung mass với trở 2kΩ chưa?")
            print(" 3. Kiểm tra chân TRIG: Cắm vào Pin 16 (GPIO 23).")
            print(" 4. Kiểm tra chân ECHO: Đã qua trở 1kΩ vào Pin 18 (GPIO 24) chưa?")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PBL6 - HC-SR04 Ultrasonic Sensor Standalone Tester")
    parser.add_argument("--trig", type=int, default=DEFAULT_TRIG_PIN, help=f"Chân GPIO TRIG (mặc định: {DEFAULT_TRIG_PIN})")
    parser.add_argument("--echo", type=int, default=DEFAULT_ECHO_PIN, help=f"Chân GPIO ECHO (mặc định: {DEFAULT_ECHO_PIN})")
    parser.add_argument("--rate", type=int, default=15, help="Tần số đo (Hz, mặc định: 15)")
    parser.add_argument("--beep", action="store_true", help="Bật còi bíp khẩn cấp khi khoảng cách < 1.0m")
    parser.add_argument("--wiring", "-w", action="store_true", help="Hiển thị sơ đồ đấu nối chân")
    args = parser.parse_args()

    if args.wiring:
        print(WIRING_DIAGRAM)
        return

    if not HAS_RPI_GPIO:
        print("=" * 70)
        print("⚠️  CẢNH BÁO: Thư viện 'RPi.GPIO' không khả dụng trên máy này.")
        print("    Script này được thiết kế để chạy trực tiếp trên RASPBERRY PI 4.")
        print("    Nếu bạn đang kiểm tra sơ đồ chân, xem bằng lệnh:")
        print("    python experiments/test_ultrasonic.py --wiring")
        print("=" * 70)
        return

    # Khởi tạo phần cứng GPIO trên Raspberry Pi
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(args.trig, GPIO.OUT)
    GPIO.setup(args.echo, GPIO.IN)
    GPIO.output(args.trig, GPIO.LOW)
    time.sleep(0.1)

    try:
        run_live_test(
            trig_pin=args.trig,
            echo_pin=args.echo,
            poll_rate_hz=args.rate,
            enable_beep=args.beep
        )
    finally:
        GPIO.cleanup([args.trig, args.echo])
        print("[GPIO] Đã giải phóng chân GPIO an toàn.")


if __name__ == "__main__":
    main()

