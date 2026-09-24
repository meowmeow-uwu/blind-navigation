# 2_edge_device/modules/audio/audio_broker.py
import threading
import json
import time
import subprocess
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

AUDIO_FILE_PATH = "/tmp/nav_alert.wav"

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Navigation Audio - PBL6</title>
</head>
<body style="font-family: Arial, sans-serif; text-align: center; padding: 25px; background-color: #f8f9fa;">
    <h2>ĐIỀU HƯỚNG BẰNG GIỌNG NÓI</h2>
    <p style="color: #666;">Sau khi bấm nút, bạn có thể <b>khóa màn hình hoặc bỏ túi</b>, âm thanh vẫn sẽ phát bình thường.</p>
    <button id="btn" style="font-size: 20px; font-weight: bold; padding: 18px 32px; border-radius: 12px; background-color: #28a745; color: white; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        KÍCH HOẠT LOA NỀN
    </button>
    <div id="status" style="margin-top: 20px; font-size: 16px; color: #6c757d;">Trạng thái: Chưa kích hoạt</div>
    <div id="msg-box" style="margin-top: 25px; padding: 20px; background: white; border-radius: 8px; border-left: 6px solid #007bff; min-height: 40px; font-size: 20px; font-weight: bold; color: #333;">
        (Chưa phát hiện vật cản)
    </div>

    <!-- Player âm thanh chính chạy liên tục -->
    <audio id="player" preload="auto"></audio>

    <script>
        let lastId = -1;
        const btn = document.getElementById('btn');
        const status = document.getElementById('status');
        const msgBox = document.getElementById('msg-box');
        const player = document.getElementById('player');

        // Bật MediaSession để Android hiểu đây là phiên phát nhạc nền (cho phép chạy khi khóa màn hình)
        if ('mediaSession' in navigator) {
            navigator.mediaSession.metadata = new MediaMetadata({
                title: 'Điều Hướng Trợ Lý',
                artist: 'Raspberry Pi 4',
                album: 'PBL6 Blind Navigation'
            });
        }

        btn.onclick = () => {
            btn.disabled = true;
            btn.style.backgroundColor = '#6c757d';
            btn.innerText = "ĐANG PHÁT ÂM THANH NỀN (CÓ THỂ TẮT MÀN HÌNH)";
            status.innerText = "Đã kết nối! Bạn có thể bấm nút nguồn khóa máy.";
            status.style.color = "#28a745";

            // Kích hoạt quyền Audio context từ thao tác bấm của người dùng
            player.src = "/audio?t=" + Date.now();
            player.play().catch(e => console.log("Init play:", e));

            setInterval(async () => {
                try {
                    const res = await fetch('/alert');
                    const data = await res.json();

                    if (data.id !== lastId && data.text) {
                        lastId = data.id;
                        msgBox.innerText = data.text;

                        // Tải file wav mới nhất vừa được Pi tạo ra
                        player.src = "/audio?t=" + Date.now();
                        player.play().catch(e => console.log("Play error:", e));
                    }
                } catch (e) {
                    status.innerText = "Mất kết nối!";
                    status.style.color = "#dc3545";
                }
            }, 300);
        };
    </script>
</body>
</html>
"""

class VoiceRequestHandler(SimpleHTTPRequestHandler):
    broker_instance = None

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/alert':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            payload = {
                "id": self.broker_instance.alert_id,
                "text": self.broker_instance.current_alert
            }
            self.wfile.write(json.dumps(payload).encode('utf-8'))
        elif self.path.startswith('/audio'):
            if os.path.exists(AUDIO_FILE_PATH):
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                with open(AUDIO_FILE_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

class AsyncAudioBroker:
    def __init__(self, port: int = 8080):
        self.port = port
        self.current_alert = ""
        self.alert_id = 0
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        self.lock = threading.Lock()

        # Tạo file âm thanh rỗng khởi đầu
        subprocess.run(["espeak-ng", "-v", "vi", "-w", AUDIO_FILE_PATH, " "])

        VoiceRequestHandler.broker_instance = self
        self.server = HTTPServer(('0.0.0.0', self.port), VoiceRequestHandler)
        
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[AUDIO BROKER] Background Audio Server đang chạy trên cổng {self.port}...")

    def speak(self, text: str, cooldown: float = 2.0):
        curr_time = time.time()
        with self.lock:
            if text == self.last_spoken_text and (curr_time - self.last_spoken_time) < cooldown:
                return
            
            # Xuất giọng nói ra file WAV cục bộ bằng espeak-ng tiếng Việt (tốc độ ~0.05s)
            subprocess.run(["espeak-ng", "-v", "vi", "-s", "165", "-w", AUDIO_FILE_PATH, text])
            
            self.alert_id += 1
            self.current_alert = text
            self.last_spoken_text = text
            self.last_spoken_time = curr_time
            print(f"[SPEAK] #{self.alert_id}: {text}")

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        print("[AUDIO BROKER] Đã dừng Audio Server.")