# 2_edge_device/experiments/test_yolo_ncnn.py
import cv2
import time
import sys
import os

# Đảm bảo import đúng đường dẫn module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner

def main():
    # Đường dẫn tới model tải từ repo Qengineering
    param_path = "../modules/vision_local/models/yolov8n.param"
    bin_path = "../modules/vision_local/models/yolov8n.bin"
    
    print("[INFO] Khởi tạo hệ thống NCNN...")
    detector = YoloNcnnRunner(param_path, bin_path, target_size=320)
    
    # Mở camera (hoặc dùng video test)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Không tìm thấy Camera OV5647")
        return

    print("[INFO] Bắt đầu benchmark FPS...")
    fps_list = []
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # Chạy Edge-AI
            detections = detector.detect(frame)
            
            # Tính FPS
            process_time = time.time() - start_time
            fps = 1.0 / process_time
            fps_list.append(fps)
            
            # Vẽ Bounding Box lên ảnh để test (Tắt bước này khi deploy thực tế để tiết kiệm CPU)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("YOLOv8 NCNN Benchmark", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"\n[RESULT] Benchmark kết thúc. FPS Trung bình: {avg_fps:.2f}")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()