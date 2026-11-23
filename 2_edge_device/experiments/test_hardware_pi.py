"""
test_hardware_pi.py - Script chẩn đoán nhanh phần cứng trên Raspberry Pi 4
==========================================================================
Chạy lệnh này trước khi khởi động main.py để đảm bảo:
1. Loa / Tai nghe phát được tiếng Việt và tiếng bíp.
2. Camera (CSI hoặc USB Webcam) đọc được khung hình.
3. Cảm biến siêu âm HC-SR04 đo được khoảng cách chuẩn.
"""

import sys
import time
import subprocess
import shutil

print("=" * 60)
print("   KIỂM TRA CHẨN ĐOÁN PHẦN CỨNG RASPBERRY PI 4 (PBL6)   ")
print("=" * 60)

# 1. KIỂM TRA ÂM THANH
print("\n[1/3] Kiểm tra Âm thanh (Loa / Tai nghe)...")
# Tự động kích hoạt âm lượng ALSA (Unmute và tăng 90%)
for ctrl in ["Headphone", "Master", "PCM"]:
    try:
        subprocess.run(["amixer", "sset", ctrl, "90%", "unmute"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

if shutil.which("espeak-ng"):
    print("  -> Đang phát thử giọng nói tiếng Việt qua espeak-ng...")
    subprocess.run(["espeak-ng", "-v", "vi", "-s", "160", "Kiểm tra loa thành công. Hệ thống sẵn sàng."], check=False)
    print("  ✅ Đã gửi lệnh phát âm thanh tới loa.")
    print("  💡 Nếu không nghe thấy tiếng, chạy lệnh riêng: python experiments/test_speaker.py --sweep")
else:
    print("  ⚠️ Chưa cài đặt espeak-ng. Chạy: sudo apt install -y espeak-ng")

# 2. KIỂM TRA CAMERA
print("\n[2/3] Kiểm tra Camera (CSI hoặc USB)...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  ✅ Mở camera thành công! Độ phân giải: {w}x{h}")
        else:
            print("  ⚠️ Camera mở được nhưng không đọc được frame.")
        cap.release()
    else:
        print("  ❌ Không thể mở Camera cổng 0. Kiểm tra cáp CSI hoặc cắm lại USB.")
except Exception as e:
    print(f"  ❌ Lỗi OpenCV Camera: {e}")

# 3. KIỂM TRA CẢM BIẾN SIÊU ÂM HC-SR04
print("\n[3/3] Kiểm tra Cảm biến siêu âm HC-SR04 (GPIO 23 TRIG, GPIO 24 ECHO)...")
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    TRIG = 23
    ECHO = 24
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.1)

    print("  Đang đo thử 3 lần:")
    success_count = 0
    for i in range(1, 4):
        GPIO.output(TRIG, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.000010)
        GPIO.output(TRIG, GPIO.LOW)

        t_now = time.perf_counter()
        timeout = t_now + 0.035

        pulse_start = time.perf_counter()
        while GPIO.input(ECHO) == GPIO.LOW:
            pulse_start = time.perf_counter()
            if pulse_start > timeout:
                break

        pulse_end = pulse_start
        timeout_end = pulse_start + 0.035
        while GPIO.input(ECHO) == GPIO.HIGH:
            pulse_end = time.perf_counter()
            if pulse_end > timeout_end:
                break

        if pulse_start <= timeout and pulse_end > pulse_start:
            duration = pulse_end - pulse_start
            distance_cm = (duration * 34300) / 2
            if 2.0 <= distance_cm <= 400.0:
                print(f"    Lần {i}: {distance_cm:6.1f} cm ({distance_cm/100:.2f} m)")
                success_count += 1
            else:
                print(f"    Lần {i}: Ngoài phạm vi ({distance_cm:.1f} cm)")
        else:
            print(f"    Lần {i}: ⚠️ Không nhận được xung Echo (Cảm biến chưa cắm hoặc lỏng dây)")

        time.sleep(0.25)

    GPIO.cleanup([TRIG, ECHO])
    if success_count > 0:
        print("  ✅ Cảm biến HC-SR04 đọc xung thành công!")
    else:
        print("  ⚠️ Chưa nhận được tín hiệu HC-SR04. Vui lòng cắm cảm biến và kiểm tra dây nối.")
except ImportError:
    print("  ⚠️ RPi.GPIO chưa được cài đặt. Chạy: pip install RPi.GPIO")
except Exception as e:
    print(f"  ❌ Lỗi đo siêu âm: {e}. Kiểm tra lại mạch phân áp trở 1k/2k và dây cắm.")

print("\n" + "=" * 60)
print("Hoàn tất chẩn đoán. Nếu cả 3 mục đều xanh (✅), bạn có thể chạy main.py!")
print("=" * 60)

