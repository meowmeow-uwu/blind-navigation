# 2_edge_device/experiments/test_yolo_ncnn.py
import cv2
import time
import threading
import sys
import os

# Tự động định vị đường dẫn module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner

# Biến toàn cục chia sẻ giữa các luồng
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
running = True
ai_fps = 0.0

def inference_worker(detector):
    """Luồng phụ chạy ngầm: Chỉ tập trung suy luận Edge-AI không ngừng nghỉ"""
    global latest_frame, latest_detections, running, ai_fps
    
    fps_counter = 0
    fps_timer = time.perf_counter()
    
    while running:
        # Lấy bản sao frame mới nhất để xử lý
        frame_to_process = None
        with frame_lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
                
        if frame_to_process is not None:
            # Chạy mạng nơ-ron
            results = detector.detect(frame_to_process)
            
            with frame_lock:
                latest_detections = results
                
            # Đo đạc FPS thực tế của riêng AI
            fps_counter += 1
            if time.perf_counter() - fps_timer >= 1.0:
                ai_fps = fps_counter / (time.perf_counter() - fps_timer)
                fps_counter = 0
                fps_timer = time.perf_counter()
        else:
            time.sleep(0.005)

def main():
    global latest_frame, latest_detections, running, ai_fps
    
    param_path = os.path.join(current_dir, "../modules/vision_local/models/yolov8n.param")
    bin_path = os.path.join(current_dir, "../modules/vision_local/models/yolov8n.bin")
    
    print("[INFO] Khởi tạo hệ thống YOLOv8 NCNN (Async Mode)...")
    # Đặt target_size=320 để giảm số phép tính FLOPs
    detector = YoloNcnnRunner(param_path, bin_path, target_size=320, conf_thresh=0.5, nms_thresh=0.45)
    
    # Mở Camera OV5647
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Không tìm thấy Camera")
        return

    # Khởi động luồng suy luận AI chạy ngầm
    ai_thread = threading.Thread(target=inference_worker, args=(detector,), daemon=True)
    ai_thread.start()
    
    print("[INFO] Đang mở cửa sổ xem trực tiếp. Bấm 'q' để dừng...")
    
    prev_time = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Đẩy frame mới vào vùng nhớ chung cho luồng AI lấy
            with frame_lock:
                latest_frame = frame
                detections = list(latest_detections)
                
            # Vẽ các Bounding Box nhận diện được
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tránh lỗi tràn chữ ra ngoài mép ảnh
                label_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, label_y - h_text - 2), (x1 + w_text, label_y + 4), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Tính toán FPS hiển thị (Camera display FPS)
            curr_time = time.perf_counter()
            display_fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 30.0
            prev_time = curr_time

            # Hiển thị đồng thời cả 2 chỉ số: FPS của Camera và FPS của lõi AI
            h_frame, _ = frame.shape[:2]
            cv2.putText(frame, f"AI FPS: {ai_fps:.1f}", (15, h_frame - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Display FPS: {display_fps:.1f}", (15, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Edge-AI Preview (Async)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        ai_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Chương trình kết thúc thành công.")

if __name__ == "__main__":
    main()