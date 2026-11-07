# 2_edge_device/tests/test_onnx_runner.py
import os
import sys
import time
import numpy as np
import cv2

# Thêm đường dẫn gốc của module vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_DEVICE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if EDGE_DEVICE_DIR not in sys.path:
    sys.path.insert(0, EDGE_DEVICE_DIR)

from modules.vision_local.yolo_onnx_runner import YoloOnnxRunner, letterbox


def test_letterbox():
    """Kiểm tra tính đúng đắn của hàm letterbox."""
    print("[TEST] Bắt đầu test letterbox...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    target_size = 640

    padded_img, ratio, (dw, dh) = letterbox(dummy_img, (target_size, target_size), auto=False)

    assert padded_img.shape == (target_size, target_size, 3), f"Sai shape: {padded_img.shape}"
    assert ratio == 1.0, f"Ratio không khớp: {ratio}"
    assert dw == 0.0, f"dw không khớp: {dw}"
    assert dh == 80.0, f"dh không khớp: {dh} (480 lên 640 cần pad 160 / 2 = 80)"

    print("  -> test_letterbox: PASS!")


def test_yolo_onnx_runner():
    """Kiểm tra khởi tạo và inference của YoloOnnxRunner."""
    print("[TEST] Bắt đầu test YoloOnnxRunner...")
    model_path = os.path.join(EDGE_DEVICE_DIR, "modules/vision_local/models/yolov8n.onnx")

    if not os.path.exists(model_path):
        print(f"  [SKIP] Model chưa tồn tại tại {model_path}")
        return

    runner = YoloOnnxRunner(
        model_path=model_path,
        target_size=640,
        conf_thresh=0.25,
        nms_thresh=0.45
    )

    # Tạo ảnh ngẫu nhiên giả lập kích thước camera 640x480
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 1. Test detect_raw
    boxes, scores, classes = runner.detect_raw(test_frame)
    assert isinstance(boxes, np.ndarray), "boxes phải là np.ndarray"
    assert isinstance(scores, np.ndarray), "scores phải là np.ndarray"
    assert isinstance(classes, np.ndarray), "classes phải là np.ndarray"
    print(f"  -> detect_raw output: {len(boxes)} detections")

    # 2. Test detect chuẩn BaseDetector
    results = runner.detect(test_frame)
    assert isinstance(results, list), "results phải là list"
    print(f"  -> detect output: {len(results)} items")

    # 3. Đo benchmark thời gian inference (warmup 3 lần, chạy 10 lần)
    for _ in range(3):
        runner.detect(test_frame)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        runner.detect(test_frame)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = np.mean(latencies)
    print(f"  -> Benchmark trung bình: {avg_latency:.2f} ms (~{1000/avg_latency:.1f} FPS)")
    print("  -> test_yolo_onnx_runner: PASS!")


if __name__ == "__main__":
    test_letterbox()
    test_yolo_onnx_runner()
