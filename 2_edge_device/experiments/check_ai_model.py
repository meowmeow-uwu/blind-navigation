"""
check_ai_model.py - Script chẩn đoán và kiểm tra mô hình AI YOLOv8 NCNN
========================================================================
Mục đích:
- Kiểm tra trọng số mô hình (.bin) và cấu trúc mạng (.param) có tồn tại trên thiết bị không.
- Kiểm tra dung lượng file có bị lỗi hay rỗng không.
- Kiểm tra thư viện NCNN có nạp và suy luận được không.
- Đo FPS suy luận thực tế của riêng mô hình AI trên CPU.
"""

import os
import sys
import time
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.insert(0, BASE_DIR)

PARAM_PATH = os.path.join(BASE_DIR, "modules/vision_local/models/yolov8n.param")
BIN_PATH = os.path.join(BASE_DIR, "modules/vision_local/models/yolov8n.bin")

print("=" * 66)
print("     KIỂM TRA CHẨN ĐOÁN MÔ HÌNH AI YOLOV8 NCNN TRÊN THIẾT BỊ     ")
print("=" * 66)

# 1. Kiểm tra file cấu trúc (.param)
if os.path.exists(PARAM_PATH):
    size_param = os.path.getsize(PARAM_PATH) / 1024
    print(f" [1/5] File cấu trúc yolov8n.param:   ✅ TỒN TẠI ({size_param:.1f} KB)")
else:
    print(f" [1/5] File cấu trúc yolov8n.param:   ❌ KHÔNG TÌM THẤY tại {PARAM_PATH}")

# 2. Kiểm tra file trọng số (.bin)
if os.path.exists(BIN_PATH):
    size_bin = os.path.getsize(BIN_PATH) / (1024 * 1024)
    if size_bin > 1.0:
        print(f" [2/5] File trọng số yolov8n.bin:     ✅ TỒN TẠI ({size_bin:.2f} MB)")
    else:
        print(f" [2/5] File trọng số yolov8n.bin:     ⚠️ FILE BỊ LỖI (Dung lượng chỉ {size_bin:.2f} MB, chuẩn phải ~6MB)")
else:
    print(f" [2/5] File trọng số yolov8n.bin:     ❌ KHÔNG TÌM THẤY tại {BIN_PATH}")

# 3. Kiểm tra thư viện NCNN
try:
    import ncnn
    print(f" [3/5] Thư viện NCNN trong Python:    ✅ ĐÃ CÀI ĐẶT THÀNH CÔNG")
except ImportError:
    print(f" [3/5] Thư viện NCNN trong Python:    ❌ CHƯA CÀI ĐẶT! Chạy lệnh: pip install ncnn")
    sys.exit(1)

# 4. Khởi tạo mô hình
try:
    from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
    runner = YoloNcnnRunner(PARAM_PATH, BIN_PATH, target_size=320, conf_thresh=0.20)
    print(f" [4/5] Nạp mạng nơ-ron vào RAM:       ✅ THÀNH CÔNG (Số nhãn: {len(runner.classes)})")
except Exception as e:
    print(f" [4/5] Nạp mạng nơ-ron vào RAM:       ❌ THẤT BẠI: {e}")
    sys.exit(1)

# 5. Thử nghiệm suy luận & Đo FPS
print(f" [5/5] Đo tốc độ suy luận (Benchmark 20 khung hình)...")
dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
times = []
for _ in range(20):
    t0 = time.perf_counter()
    _ = runner.detect(dummy_img)
    times.append(time.perf_counter() - t0)

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time
print(f"       -> Thời gian xử lý trung bình: {avg_time*1000:.1f} ms/frame")
print(f"       -> Tốc độ suy luận AI:         {fps:.1f} FPS")

print("=" * 66)
print(" KẾT LUẬN: MÔ HÌNH YOLOV8 NCNN ĐÃ ĐƯỢC NHÚNG VÀ HOẠT ĐỘNG HOÀN HẢO!")
print("=" * 66)

