# 2_edge_device/experiments/test_yolo_phone_alert.py
import cv2
import time
import threading
import sys
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner

# Biến trạng thái chia sẻ toàn cục
current_alert = ""
alert_id = 0
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
running = True

# Tên tiếng Việt cho các vật thể quen thuộc
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

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Navigation Voice - PBL6</title>
</head>
<body style="font-family: Arial, sans-serif; text-align: center; padding: 25px; background-color: #f8f9fa;">
    <h2>ĐIỀU HƯỚNG BẰNG GIỌNG NÓI</h2>
    <button id="btn" style="font-size: 20px; font-weight: bold; padding: 15px 30px; border-radius: 10px; background-color: #28a745; color: white; border: none;">
        BẬT NHẬN CẢNH BÁO
    </button>
    <div id="status" style="margin-top: 20px; font-size: 16px; color: #6c757d;">Trạng thái: Sẵn sàng kết nối</div>
    <div id="msg-box" style="margin-top: 25px; padding: 20px; background: white; border-radius: 8px; border-left: 6px solid #007bff; min-height: 40px; font-size: 22px; font-weight: bold; color: #333;">
        (Chưa phát hiện vật cản)
    </div>

    <script>
        let lastId = -1;
        const btn = document.getElementById('btn');
        const status = document.getElementById('status');
        const msgBox = document.getElementById('msg-box');

        function speakText(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel(); // Dừng câu cũ để đọc câu khẩn cấp mới
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'vi-VN';
                utterance.rate = 1.15; // Tốc độ hơi nhanh để phản xạ kịp thời
                window.speechSynthesis.speak(utterance);
            }
        }

        btn.onclick = () => {
            btn.disabled = true;
            btn.style.backgroundColor = '#6c757d';
            btn.innerText = "ĐANG LẮNG NGHE CAMERA...";
            speakText("Hệ thống cảnh báo đã kích hoạt");

            setInterval(async () => {
                try {
                    const res = await fetch('/alert');
                    const data = await res.json();
                    status.innerText = "Trạng thái: Đã kết nối với camera Pi";
                    status.style.color = "#28a745";

                    if (data.id !== lastId && data.text) {
                        lastId = data.id;
                        msgBox.innerText = data.text;
                        speakText(data.text);
                    }
                } catch (e) {
                    status.innerText = "Trạng thái: Mất kết nối!";
                    status.style.color = "#dc3545";
                }
            }, 300);
        };
    </script>
</body>
</html>
"""

class VoiceHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global current_alert, alert_id
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/alert':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"id": alert_id, "text": current_alert}).encode('utf-8'))
        else:
            self.send_error(404)

def evaluate_lanes(detections, frame_width):
    """Chia khung hình làm 3 làn để phân loại phương hướng"""
    lane_width = frame_width / 3.0
    left_obs, center_obs, right_obs = [], [], []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2.0
        name = LABEL_TRANSLATION.get(det['class_name'], det['class_name'])
        
        if cx < lane_width:
            left_obs.append(name)
        elif cx <= lane_width * 2.0:
            center_obs.append(name)
        else:
            right_obs.append(name)

    return left_obs, center_obs, right_obs

def ai_worker(detector):
    """Luồng phụ chạy ngầm: Xử lý YOLO và tạo câu lệnh điều hướng"""
    global latest_frame, latest_detections, running, current_alert, alert_id

    last_spoken_text = ""
    last_spoken_time = 0

    while running:
        frame_copy = None
        with frame_lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()

        if frame_copy is not None:
            results = detector.detect(frame_copy)
            with frame_lock:
                latest_detections = results

            # Tạo thông báo cảnh báo dựa trên phân làn
            if results:
                img_w = frame_copy.shape[1]
                left_obs, center_obs, right_obs = evaluate_lanes(results, img_w)
                
                alert_text = ""
                # Ưu tiên số 1: Xử lý vật cản làn giữa
                if center_obs:
                    obj = center_obs[0]
                    if not right_obs:
                        alert_text = f"Có {obj} phía trước, né sang phải"
                    elif not left_obs:
                        alert_text = f"Có {obj} phía trước, né sang trái"
                    else:
                        alert_text = f"Dừng lại, có {obj} chắn phía trước"
                elif left_obs:
                    alert_text = f"Có {left_obs[0]} bên trái"
                elif right_obs:
                    alert_text = f"Có {right_obs[0]} bên phải"

                curr_time = time.time()
                # Chống lặp từ: Chỉ cập nhật nếu đổi câu hoặc sau 2.5 giây
                if alert_text and (alert_text != last_spoken_text or (curr_time - last_spoken_time) > 2.5):
                    last_spoken_text = alert_text
                    last_spoken_time = curr_time
                    alert_id += 1
                    current_alert = alert_text
                    print(f"[ALERT EMITTED] #{alert_id}: {alert_text}")
        else:
            time.sleep(0.005)

def main():
    global latest_frame, latest_detections, running
    
    param_path = os.path.join(current_dir, "../modules/vision_local/models/yolov8n.param")
    bin_path = os.path.join(current_dir, "../modules/vision_local/models/yolov8n.bin")

    print("[INFO] Khởi tạo mô hình YOLOv8n NCNN...")
    detector = YoloNcnnRunner(param_path, bin_path, target_size=320, conf_thresh=0.5, nms_thresh=0.45)

    # Chạy Server Web ở luồng riêng
    server = HTTPServer(('0.0.0.0', 8080), VoiceHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # Chạy luồng AI ở luồng riêng
    threading.Thread(target=ai_worker, args=(detector,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Không thể mở Camera!")
        return

    print("=" * 60)
    print("[HƯỚNG DẪN] Trên điện thoại Realme, mở Chrome:")
    print("            Truy cập: http://<IP_CỦA_PI>:8080")
    print("            Bấm 'BẬT NHẬN CẢNH BÁO'")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with frame_lock:
                latest_frame = frame
                detections = list(latest_detections)

            # Vẽ trực quan để theo dõi trên màn hình Pi
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10 if y1 > 20 else y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Preview Camera & AI Alerts", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        server.shutdown()
        print("\n[INFO] Đã kết thúc chương trình.")

if __name__ == "__main__":
    main()