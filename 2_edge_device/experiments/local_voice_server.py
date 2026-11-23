# 2_edge_device/experiments/test_phone_audio.py
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import json
import time

current_alert = ""
alert_id = 0

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Receiver - PBL6</title>
</head>
<body style="font-family: Arial, sans-serif; text-align: center; padding: 30px; background-color: #f4f6f9;">
    <h2>LOA CẢNH BÁO ĐIỀU HƯỚNG</h2>
    <p>Nhấn nút bên dưới để cấp quyền phát âm thanh trên điện thoại:</p>
    
    <button id="btn" style="font-size: 20px; font-weight: bold; padding: 18px 30px; border-radius: 12px; background-color: #28a745; color: white; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        BẬT NHẬN ÂM THANH
    </button>
    
    <div id="status" style="margin-top: 25px; font-size: 18px; color: #6c757d;">Trạng thái: Đang chờ bấm nút...</div>
    <div id="msg-box" style="margin-top: 30px; padding: 20px; background: white; border-radius: 10px; border-left: 6px solid #007bff; min-height: 50px; font-size: 22px; font-weight: bold; color: #333;">
        (Chưa có cảnh báo)
    </div>

    <script>
        let lastId = -1;
        const btn = document.getElementById('btn');
        const status = document.getElementById('status');
        const msgBox = document.getElementById('msg-box');

        // Hàm phát âm thanh Text-To-Speech bằng engine Google tiếng Việt của điện thoại
        function speakText(text) {
            if ('speechSynthesis' in window) {
                // Hủy các câu đọc đang dở dang để cập nhật câu khẩn cấp mới
                window.speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'vi-VN';
                utterance.rate = 1.1; // Tốc độ đọc nhanh để cảnh báo kịp thời
                utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            } else {
                alert("Trình duyệt không hỗ trợ Web Speech API!");
            }
        }

        btn.onclick = () => {
            btn.disabled = true;
            btn.style.backgroundColor = '#6c757d';
            btn.innerText = "ĐANG LẮNG NGHE TỪ PI...";
            status.innerText = "Trạng thái: Đang kết nối với Raspberry Pi...";
            status.style.color = "#007bff";

            // Đọc thử 1 câu chào mừng để mở khóa AudioContext trên Android
            speakText("Đã kết nối thành công tới thiết bị điều hướng");

            // Thăm dò lệnh cảnh báo liên tục mỗi 300ms
            setInterval(async () => {
                try {
                    const res = await fetch('/alert');
                    const data = await res.json();
                    
                    status.innerText = "Trạng thái: Đang hoạt động bình thường";
                    status.style.color = "#28a745";

                    if (data.id !== lastId && data.text) {
                        lastId = data.id;
                        msgBox.innerText = data.text;
                        speakText(data.text);
                    }
                } catch (e) {
                    status.innerText = "Trạng thái: Mất kết nối với Raspberry Pi!";
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

def simulate_detections():
    """Giả lập chuỗi cảnh báo điều hướng gửi sang điện thoại theo thời gian thực"""
    global current_alert, alert_id
    alerts = [
        "Có người phía trước, lách sang phải",
        "Cột bên phải, một mét rưỡi",
        "Có bàn phía trước, lách sang trái",
        "Dừng lại, vật cản nguy hiểm ở cự ly gần"
    ]
    time.sleep(5)
    index = 0
    while True:
        alert_id += 1
        current_alert = alerts[index % len(alerts)]
        print(f"[PI LOG] Đang phát cảnh báo #{alert_id}: {current_alert}")
        index += 1
        time.sleep(4)

def main():
    threading.Thread(target=simulate_detections, daemon=True).start()
    server = HTTPServer(('0.0.0.0', 8080), VoiceHandler)
    print("=" * 55)
    print("[INFO] Server âm thanh đang chạy!")
    print("[HƯỚNG DẪN] Trên điện thoại Realme, mở trình duyệt Chrome:")
    print("            Truy cập địa chỉ: http://<IP_CỦA_PI>:8080")
    print("=" * 55)
    server.serve_forever()

if __name__ == "__main__":
    main()