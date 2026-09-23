# 2_edge_device/main.py
import cv2
import time
import os
import sys

from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
from modules.audio_broker import AsyncAudioBroker

def evaluate_lanes(detections, frame_width):
    """
    Chia chiều ngang khung hình thành 3 làn:
    - Làn Trái: 0 -> 33% width
    - Làn Giữa: 33% -> 66% width
    - Làn Phải: 66% -> 100% width
    """
    lane_width = frame_width / 3.0
    lane_center_start = lane_width
    lane_center_end = lane_width * 2.0
    
    obstacles_in_center = []
    obstacles_in_left = []
    obstacles_in_right = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2.0 # Tọa độ tâm vật thể
        
        if cx < lane_center_start:
            obstacles_in_left.append(det['class_name'])
        elif cx <= lane_center_end:
            obstacles_in_center.append(det['class_name'])
        else:
            obstacles_in_right.append(det['class_name'])
            
    return obstacles_in_left, obstacles_in_center, obstacles_in_right

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(current_dir, "modules/vision_local/models/yolov8n.param")
    bin_path = os.path.join(current_dir, "modules/vision_local/models/yolov8n.bin")
    
    print("[INFO] Khởi tạo hệ thống Navigation...")
    detector = YoloNcnnRunner(param_path, bin_path, target_size=320, conf_thresh=0.5, nms_thresh=0.45)
    audio = AsyncAudioBroker(rate=175)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Không tìm thấy Camera OV5647")
        return
        
    audio.speak("Hệ thống điều hướng đã sẵn sàng")
    print("[INFO] Đang chạy điều hướng theo thời gian thực (Bấm Ctrl+C để dừng)...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            img_h, img_w = frame.shape[:2]
            detections = detector.detect(frame)
            
            if detections:
                left_obs, center_obs, right_obs = evaluate_lanes(detections, img_w)
                
                # Logic phân luồng điều hướng
                if center_obs:
                    main_obj = center_obs[0]
                    # Nếu làn giữa vướng nhưng làn phải thoáng -> Hướng dẫn né phải
                    if not right_obs:
                        audio.speak(f"Có {main_obj} phía trước, né sang phải", cooldown=2.5)
                    # Nếu làn phải cũng vướng -> Hướng dẫn né trái
                    elif not left_obs:
                        audio.speak(f"Có {main_obj} phía trước, né sang trái", cooldown=2.5)
                    else:
                        audio.speak(f"Dừng lại, phía trước có {main_obj}", cooldown=1.5)
                elif left_obs:
                    audio.speak(f"{left_obs[0]} bên trái", cooldown=3.5)
                elif right_obs:
                    audio.speak(f"{right_obs[0]} bên phải", cooldown=3.5)
                    
            time.sleep(0.03) # Giữ chu kỳ ổn định
            
    except KeyboardInterrupt:
        pass
    finally:
        audio.speak("Đang tắt hệ thống")
        time.sleep(1.0)
        audio.stop()
        cap.release()
        print("\n[DONE] Đã thoát chương trình an toàn.")

if __name__ == "__main__":
    main()