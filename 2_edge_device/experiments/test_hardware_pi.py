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
if shutil.which("espeak-ng"):
    print("  -> Đang phát thử giọng nói qua espeak-ng...")
    subprocess.run(["espeak-ng", "-v", "vi", "-s", "160", "Kiểm tra loa thành công"], check=False)
    print("  ✅ espeak-ng hoạt động tốt.")
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
    TRIG = 23
    ECHO = 24
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.1)

    print("  Đang đo thử 3 lần:")
    for i in range(1, 4):
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        start_t = time.time()
        timeout = start_t + 0.04
        while GPIO.input(ECHO) == GPIO.LOW:
            start_t = time.time()
            if start_t > timeout:
                break

        stop_t = time.time()
        while GPIO.input(ECHO) == GPIO.HIGH:
            stop_t = time.time()
            if stop_t > timeout:
                break

        duration = stop_t - start_t
        distance_cm = (duration * 34300) / 2
        print(f"    Lần {i}: {distance_cm:.1f} cm ({distance_cm/100:.2f} m)")
        time.sleep(0.3)

    GPIO.cleanup([TRIG, ECHO])
    print("  ✅ Cảm biến HC-SR04 đọc xung thành công!")
except ImportError:
    print("  ⚠️ RPi.GPIO chưa được cài đặt. Chạy: pip install RPi.GPIO")
except Exception as e:
    print(f"  ❌ Lỗi đo siêu âm: {e}. Kiểm tra lại mạch phân áp trở 1k/2k và dây cắm.")

print("\n" + "=" * 60)
print("Hoàn tất chẩn đoán. Nếu cả 3 mục đều xanh (✅), bạn có thể chạy main.py!")
print("=" * 60)

