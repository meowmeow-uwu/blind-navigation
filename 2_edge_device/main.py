# 2_edge_device/main.py
import cv2
import time
import os
import sys
import threading
import RPi.GPIO as GPIO

# Import theo đúng cấu trúc thư mục mới
from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
from modules.audio.audio_broker import AsyncAudioBroker
from modules.vision_cloud.vision_cloud_runner import call_qwen_vl_server

# Bảng dịch nhãn COCO sang tiếng Việt
LABEL_TRANSLATION = {
    "person": "người",
    "chair": "ghế",
    "dining table": "bàn",
    "bottle": "chai nước",
    "backpack": "ba lô",
    "cell phone": "điện thoại",
    "laptop": "máy tính",
    "tv": "màn hình"
}

latest_frame = None
vlm_is_processing = False
lock = threading.Lock()

def evaluate_lanes(detections, frame_width):
    lane_width = frame_width / 3.0
    lane_center_start = lane_width
    lane_center_end = lane_width * 2.0
    
    obstacles_in_left = []
    obstacles_in_center = []
    obstacles_in_right = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2.0
        name = LABEL_TRANSLATION.get(det['class_name'], det['class_name'])
        
        if cx < lane_center_start:
            obstacles_in_left.append(name)
        elif cx <= lane_center_end:
            obstacles_in_center.append(name)
        else:
            obstacles_in_right.append(name)
            
    return obstacles_in_left, obstacles_in_center, obstacles_in_right

def vlm_assistant_worker(audio_broker):
    global latest_frame, vlm_is_processing
    
    with lock:
        if latest_frame is None:
            print("[VLM] Chưa có frame ảnh từ camera để trích xuất.")
            vlm_is_processing = False
            return
        snap_shot = latest_frame.copy()
        
    print("[VLM] Đang gửi ảnh lên Cloud VLM...")
    audio_broker.speak("Đang phân tích không gian xung quanh, vui lòng chờ", cooldown=0.1)
    
    try:
        description = call_qwen_vl_server(snap_shot)
        print(f"[VLM] Phản hồi: {description}")
        audio_broker.speak(description, cooldown=0.1)
    except Exception as e:
        print(f"[VLM ERROR] {e}")
        audio_broker.speak("Không thể kết nối đến máy chủ phân tích", cooldown=0.1)
    finally:
        vlm_is_processing = False

def monitor_sos_button(audio_broker):
    global vlm_is_processing
    BUTTON_PIN = 18
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    print(f"[INFO] Nút bấm SOS (GPIO {BUTTON_PIN}) đã sẵn sàng.")
    
    try:
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.05) # Debounce
                if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    if not vlm_is_processing:
                        vlm_is_processing = True
                        print("\n[SOS] Đã nhấn nút SOS! Khởi chạy VLM Cloud...")
                        vlm_thread = threading.Thread(target=vlm_assistant_worker, args=(audio_broker,), daemon=True)
                        vlm_thread.start()
                    else:
                        print("[INFO] Hệ thống đang phân tích bối cảnh trước đó...")
                        
                    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                        time.sleep(0.1)
            time.sleep(0.05)
    except Exception as e:
        print(f"[ERROR] Luồng GPIO nút bấm: {e}")

def main():
    global latest_frame
    current_dir = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(current_dir, "modules/vision_local/models/yolov8n.param")
    bin_path = os.path.join(current_dir, "modules/vision_local/models/yolov8n.bin")
    
    print("[INFO] Khởi tạo hệ thống Navigation cho Raspberry Pi 4...")
    detector = YoloNcnnRunner(param_path, bin_path, target_size=320, conf_thresh=0.5, nms_thresh=0.45)
    audio = AsyncAudioBroker(port=8080)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Không tìm thấy Camera OV5647")
        return
        
    audio.speak("Hệ thống điều hướng đã sẵn sàng", cooldown=0.1)
    
    btn_thread = threading.Thread(target=monitor_sos_button, args=(audio,), daemon=True)
    btn_thread.start()
    
    print("=" * 60)
    print("[HƯỚNG DẪN] Mở trình duyệt Chrome trên điện thoại Realme:")
    print("            Truy cập: http://<IP_CỦA_PI>:8080")
    print("            Bấm nút 'BẬT NHẬN CẢNH BÁO' để phát loa ngoài")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            with lock:
                latest_frame = frame.copy()
                
            img_h, img_w = frame.shape[:2]
            detections = detector.detect(frame)
            
            # Chỉ phát cảnh báo vật cản nếu VLM Cloud đang không chiếm quyền nói
            if detections and not vlm_is_processing:
                left_obs, center_obs, right_obs = evaluate_lanes(detections, img_w)
                
                if center_obs:
                    main_obj = center_obs[0]
                    if not right_obs:
                        audio.speak(f"Có {main_obj} phía trước, né sang phải", cooldown=2.5)
                    elif not left_obs:
                        audio.speak(f"Có {main_obj} phía trước, né sang trái", cooldown=2.5)
                    else:
                        audio.speak(f"Dừng lại, phía trước có {main_obj}", cooldown=1.5)
                elif left_obs:
                    audio.speak(f"Có {left_obs[0]} bên trái", cooldown=3.5)
                elif right_obs:
                    audio.speak(f"Có {right_obs[0]} bên phải", cooldown=3.5)
                    
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\n[INFO] Đang dừng hệ thống...")
    finally:
        audio.speak("Đang tắt hệ thống", cooldown=0.1)
        time.sleep(1.0)
        audio.stop()
        cap.release()
        GPIO.cleanup()
        print("[DONE] Đã thoát chương trình an toàn.")

if __name__ == "__main__":
    main()