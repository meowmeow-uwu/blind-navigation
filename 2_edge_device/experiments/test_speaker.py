"""
test_speaker.py - Chương trình độc lập kiểm tra và chẩn đoán Loa trên Raspberry Pi 4
=====================================================================================
Mục đích:
- Kiểm tra loa / tai nghe (cổng 3.5mm, USB hoặc HDMI) có phát ra âm thanh được không.
- Tự động bật âm lượng (Unmute) và tăng Volume lên 90% nếu bị tắt tiếng.
- Phát tiếng bíp chuẩn (Tone test) và giọng nói tiếng Việt (espeak-ng).
- Quét và phát thử âm thanh trên TẤT CẢ các cổng ra (Audio Devices) của Raspberry Pi.
"""

import os
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


def create_sine_wav(filename: str, freq: float = 1000.0, duration: float = 1.0, sample_rate: int = 44100):
    """Tạo file WAV sóng sin chuẩn không phụ thuộc thư viện ngoài."""
    n_samples = int(sample_rate * duration)
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            # Áp dụng envelope fade in/out ở 5% đầu và cuối để không bị nổ tiếng
            envelope = 1.0
            ramp = int(0.05 * n_samples)
            if i < ramp:
                envelope = i / ramp
            elif i > n_samples - ramp:
                envelope = (n_samples - i) / ramp

            val = int(envelope * 24000.0 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            f.writeframes(struct.pack("<h", val))


def get_alsa_devices():
    """Lấy danh sách các card âm thanh ALSA trên Linux/Raspberry Pi."""
    devices = []
    if not IS_LINUX or not shutil.which("aplay"):
        return devices

    try:
        res = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=False)
        for line in res.stdout.splitlines():
            line_s = line.strip()
            if line_s.startswith("card "):
                devices.append(line_s)
    except Exception:
        pass
    return devices


def unmute_and_boost_volume():
    """Tự động mở âm lượng và Unmute trên Raspberry Pi."""
    if not IS_LINUX or not shutil.which("amixer"):
        return

    print("  -> Đang kiểm tra & kích hoạt âm lượng ALSA...")
    # Thử các kênh phổ biến trên Pi: Headphone, Master, PCM
    controls = ["Headphone", "Master", "PCM"]
    for ctrl in controls:
        try:
            subprocess.run(["amixer", "sset", ctrl, "90%", "unmute"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

    # Lệnh ép ngõ ra jack 3.5mm trên một số bản Pi OS cũ (numid=3: 1=3.5mm jack, 2=HDMI, 0=auto)
    try:
        subprocess.run(["amixer", "cset", "numid=3", "1"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass


def play_wav(wav_path: str, device: str = None):
    """Phát file WAV bằng trình phát hệ thống tương ứng."""
    if IS_MAC:
        subprocess.run(["afplay", wav_path], check=False)
    elif IS_LINUX:
        cmd = ["aplay", "-q"]
        if device:
            cmd.extend(["-D", device])
        cmd.append(wav_path)
        subprocess.run(cmd, check=False)


def speak_vietnamese(text: str, device: str = None):
    """Phát giọng nói tiếng Việt."""
    if IS_MAC:
        subprocess.run(["say", "-v", "Linh", text], check=False)
    elif shutil.which("espeak-ng"):
        # espeak-ng tiếng Việt
        if device and shutil.which("aplay"):
            p1 = subprocess.Popen(["espeak-ng", "-v", "vi", "-s", "155", "--stdout", text], stdout=subprocess.PIPE)
            subprocess.run(["aplay", "-q", "-D", device], stdin=p1.stdout, check=False)
            p1.stdout.close()
            p1.wait()
        else:
            subprocess.run(["espeak-ng", "-v", "vi", "-s", "155", text], check=False)
    elif shutil.which("espeak"):
        subprocess.run(["espeak", "-v", "vi", "-s", "155", text], check=False)
    else:
        print("  ⚠️ Không tìm thấy espeak-ng hoặc espeak.")


def test_suite_auto():
    """Quy trình kiểm tra âm thanh toàn diện tự động."""
    print("=" * 64)
    print("   CHƯƠNG TRÌNH KIỂM TRA LOA ĐỘC LẬP (PBL6 SPEAKER TEST)   ")
    print("=" * 64)

    # 1. Liệt kê thiết bị
    alsa_devs = get_alsa_devices()
    if alsa_devs:
        print("\n[THIẾT BỊ ÂM THANH ALSA PHÁT HIỆN ĐƯỢC]:")
        for dev in alsa_devs:
            print(f"  • {dev}")
    else:
        print(f"\n[HỆ ĐIỀU HÀNH]: {platform.system()} ({platform.machine()})")

    # 2. Mở âm lượng
    unmute_and_boost_volume()

    # Tạo các file WAV mẫu vào /tmp
    tone_1k = "/tmp/test_tone_1000hz.wav"
    tone_beep = "/tmp/test_beep_short.wav"
    create_sine_wav(tone_1k, freq=1000.0, duration=1.2)
    create_sine_wav(tone_beep, freq=1400.0, duration=0.08)

    # 3. TEST 1: Tiếng bíp chuẩn 1000Hz (Kiểm tra màng loa vật lý)
    print("\n----------------------------------------------------------------")
    print("[TEST 1/3] Phát tiếng Bíp chuẩn 1000Hz (Thời lượng 1.2 giây)...")
    print("  -> Lắng nghe loa... Bạn có nghe thấy tiếng 'TÍTTTT' không?")
    play_wav(tone_1k)
    time.sleep(0.5)

    # 4. TEST 2: Phát câu nói tiếng Việt qua espeak-ng
    print("\n----------------------------------------------------------------")
    print("[TEST 2/3] Phát giọng nói Tiếng Việt qua espeak-ng...")
    sample_text = "Xin chào. Hệ thống loa của bạn đang hoạt động bình thường. Sẵn sàng điều hướng."
    print(f"  -> Nội dung: \"{sample_text}\"")
    speak_vietnamese(sample_text)
    time.sleep(0.5)

    # 5. TEST 3: Mô phỏng nhịp bíp cảnh báo cự ly khẩn cấp (Parking Sensor)
    print("\n----------------------------------------------------------------")
    print("[TEST 3/3] Mô phỏng nhịp Bíp cảnh báo khẩn cấp (< 1.0m)...")
    print("  -> Phát chuỗi 6 tiếng bíp nhịp tăng dần...")
    intervals = [0.4, 0.3, 0.2, 0.15, 0.12, 0.1]
    for idx, dt in enumerate(intervals, 1):
        play_wav(tone_beep)
        time.sleep(dt)

    print("\n" + "=" * 64)
    print("HOÀN TẤT KIỂM TRA!")
    print("=" * 64)


def test_all_alsa_devices():
    """Phát âm thanh lần lượt qua từng cổng ra (3.5mm, HDMI, USB...) để tìm đúng cổng."""
    if not IS_LINUX:
        print("Chức năng quét thiết bị ALSA chỉ dành cho Linux/Raspberry Pi.")
        return

    print("\n[QUÉT TẤT CẢ CÁC CỔNG RA ÂM THANH ALSA]...")
    tone_wav = "/tmp/test_sweep_tone.wav"
    create_sine_wav(tone_wav, freq=1200.0, duration=0.8)

    # Quét qua card 0, card 1, card 2
    for card_idx in range(4):
        dev = f"plughw:{card_idx},0"
        print(f"\n  👉 Đang thử phát qua cổng: '{dev}'...")
        try:
            res = subprocess.run(["aplay", "-q", "-D", dev, tone_wav],
                                 capture_output=True, text=True, check=False)
            if res.returncode == 0:
                print(f"     ✅ Cổng '{dev}' phát thành công! Nếu bạn nghe thấy tiếng, loa đang cắm ở cổng này!")
                speak_vietnamese(f"Cổng số {card_idx}", device=dev)
                time.sleep(0.5)
            else:
                print(f"     ❌ Cổng '{dev}' không mở được (không có thiết bị).")
        except Exception as e:
            print(f"     ❌ Lỗi: {e}")


def interactive_menu():
    """Menu chọn kiểm thử thủ công."""
    while True:
        print("\n" + "=" * 50)
        print("     MENU KIỂM THỬ ÂM THANH RASPBERRY PI 4")
        print("=" * 50)
        print(" [1] Chạy toàn bộ quy trình test tự động (Bíp + Giọng nói)")
        print(" [2] Phát tiếng Bíp chuẩn 1000Hz (Tone Test)")
        print(" [3] Phát giọng nói tiếng Việt (\"Ghế trước mặt, Dừng lại\")")
        print(" [4] Phát chuỗi bíp khẩn cấp dồn dập (Emergency Beep)")
        print(" [5] Quét thử tất cả các ngõ ra âm thanh (Tìm cổng 3.5mm/USB)")
        print(" [6] Tăng tối đa Volume (amixer 100%)")
        print(" [q] Thoát")
        print("=" * 50)

        choice = input("👉 Chọn chức năng [1-6, q]: ").strip().lower()
        if choice in ['q', 'exit', 'quit']:
            break
        elif choice == '1':
            test_suite_auto()
        elif choice == '2':
            tone = "/tmp/test_tone_1000hz.wav"
            create_sine_wav(tone, 1000.0, 1.2)
            play_wav(tone)
        elif choice == '3':
            speak_vietnamese("Cảnh báo! Có người trước mặt hai mét. Dừng lại!")
        elif choice == '4':
            tone_b = "/tmp/test_beep_short.wav"
            create_sine_wav(tone_b, 1400.0, 0.08)
            for dt in [0.35, 0.25, 0.18, 0.12, 0.1]:
                play_wav(tone_b)
                time.sleep(dt)
        elif choice == '5':
            test_all_alsa_devices()
        elif choice == '6':
            unmute_and_boost_volume()
            print("  ✅ Đã gửi lệnh bật âm lượng 100%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBL6 - Speaker Test Tool")
    parser.add_argument("--interactive", "-i", action="store_true", help="Chạy menu tương tác")
    parser.add_argument("--sweep", "-s", action="store_true", help="Quét phát trên mọi cổng ALSA")
    parser.add_argument("--say", type=str, default="", help="Phát thử một câu nói tiếng Việt tuỳ chọn")
    args = parser.parse_args()

    if args.say:
        unmute_and_boost_volume()
        speak_vietnamese(args.say)
    elif args.sweep:
        test_all_alsa_devices()
    elif args.interactive:
        interactive_menu()
    else:
        test_suite_auto()

