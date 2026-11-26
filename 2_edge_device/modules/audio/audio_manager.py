"""
audio_manager.py - Bộ điều phối âm thanh thích ứng đa giác quan (Adaptive Feedback)
==================================================================================
Hiện thực hoá toàn diện:
1. Yêu cầu 5: Giao tiếp thích ứng đa giác quan (Adaptive Audio-Haptic Feedback)
   - Phân tầng phản hồi rõ rệt:
     + Khoảng cách khẩn cấp (< 1.0m): Tiếng bíp dồn dập có tần số và nhịp độ tăng dần
       theo khoảng cách (hoặc lệnh ngắn "Dừng!"), triệt tiêu hoàn toàn câu nói dài dòng.
     + Khoảng cách dẫn hướng (1.5m - 3.0m): Giọng nói ngắn gọn 1-2 từ khóa ("Bàn trước mặt",
       "Cửa bên trái"), giữ sự thông thoáng cho thính giác của người dùng.
     + Khoảng cách an toàn (> 3.0m): Hoàn toàn im lặng.
2. Chống đè tiếng và chống loạn âm thanh 100% (Zero Audio Overlap):
   - Luồng phát âm thanh tuần tự hoá (Single Worker Queue), ngắt lập tức câu nói dẫn hướng
     khi có báo động khẩn cấp. Không bao giờ phát 2 âm thanh đè lên nhau.
   - Giãn cách tối thiểu giữa 2 câu nói dẫn hướng (cooldown >= 2.5s).
   - Tích hợp Web Dashboard với Web Audio API tổng hợp tiếng bíp không gián đoạn.
"""

import os
import sys
import time
import math
import queue
import wave
import struct
import shutil
import platform
import threading
import subprocess
import json
from typing import Optional, Dict, Set
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

IS_MAC = platform.system() == "Darwin"

# Tạo sẵn file âm thanh beep ngắn 50ms cho loa máy
BEEP_WAV_PATH = "/tmp/pbl6_short_beep.wav"


def _create_local_beep_wav():
    """Tạo file wav tiếng bíp 50ms (tần số 1400Hz) chuẩn PCM 16-bit."""
    try:
        sample_rate = 22050
        duration = 0.05
        n_samples = int(sample_rate * duration)
        with wave.open(BEEP_WAV_PATH, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            for i in range(n_samples):
                val = int(28000.0 * math.sin(2.0 * math.pi * 1400.0 * (i / sample_rate)))
                f.writeframes(struct.pack("<h", val))
    except Exception:
        pass


_create_local_beep_wav()

# Bảng dịch 1-2 từ khoá ngắn gọn cho người khiếm thị
LABEL_SHORT_VI = {
    "person": "Người",
    "chair": "Ghế",
    "dining table": "Bàn",
    "table": "Bàn",
    "stairs": "Cầu thang",
    "door": "Cửa",
    "pole": "Cột",
    "bottle": "Chai nước",
    "backpack": "Ba lô",
    "bicycle": "Xe đạp",
    "motorcycle": "Xe máy",
    "car": "Ô tô",
    "couch": "Ghế dài",
    "bed": "Giường",
    "tv": "Màn hình",
    "laptop": "Máy tính",
    "potted plant": "Chậu cây",
    "vật cản": "Vật cản"
}

LANE_SHORT_VI = {
    "center": "trước mặt",
    "left": "bên trái",
    "right": "bên phải"
}

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PBL6 - Điều Hướng Khiếm Thị Thông Minh</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; padding: 12px; background: #121314; color: #e4e6eb; margin: 0; }
        .card { background: #1e1f21; border-radius: 12px; padding: 14px; margin: 12px auto; max-width: 660px; box-shadow: 0 4px 14px rgba(0,0,0,0.5); }
        .stream-box { width: 100%; max-width: 640px; border-radius: 8px; border: 2px solid #333; background: #000; }
        .btn-audio { font-size: 16px; font-weight: bold; padding: 12px 24px; border-radius: 8px; background: #28a745; color: white; border: none; cursor: pointer; transition: 0.2s; }
        .btn-audio.active { background: #dc3545; }
        .btn-test { font-size: 13px; font-weight: 600; padding: 8px 12px; margin: 3px; border-radius: 6px; border: none; cursor: pointer; color: white; }
        .btn-safe { background: #28a745; }
        .btn-warn { background: #ffc107; color: #111; }
        .btn-danger { background: #dc3545; }
        .btn-unlabeled { background: #6f42c1; }
        #status-box { margin-top: 10px; padding: 12px; border-radius: 8px; font-size: 18px; font-weight: bold; min-height: 24px; background: #2a2b2e; }
        .emergency-banner { background: #dc3545 !important; color: #fff !important; animation: pulse 0.5s infinite alternate; }
        @keyframes pulse { from { opacity: 0.85; } to { opacity: 1.0; } }
    </style>
</head>
<body>
    <h3 style="margin: 8px 0;">HỆ THỐNG ĐIỀU HƯỚNG KHIẾM THỊ (PBL6)</h3>
    
    <div class="card">
        <button id="btn-toggle" class="btn-audio" onclick="toggleAudio()">🔊 BẬT ÂM THANH TRÌNH DUYỆT</button>
        <div id="audio-hint" style="font-size: 12px; color: #888; margin-top: 6px;">(Bật khi xem từ điện thoại. Nếu chạy trên máy tính đang mở loa ngoài thì tắt để tránh vọng tiếng)</div>
        <div id="status-box">Hệ thống đang quan sát...</div>
    </div>

    <div class="card">
        <img class="stream-box" src="/video_feed" alt="Camera Stream" />
        <div style="margin-top: 12px;">
            <span style="font-size: 12px; color: #aaa;">MÔ PHỎNG CỰ LY CẢM BIẾN:</span><br/>
            <button class="btn-test btn-safe" onclick="sendDist(3.0)">[1] 3.0m (An toàn - Im lặng)</button>
            <button class="btn-test btn-warn" onclick="sendDist(2.0)">[2] 2.0m (Dẫn hướng 1-2 từ)</button>
            <button class="btn-test btn-danger" onclick="sendDist(0.5)">[3] 0.5m (🚨 KHẨN CẤP: Beep)</button>
            <button class="btn-test btn-unlabeled" onclick="sendDist(0.7)">[4] 0.7m (Vật cản vô hình)</button>
        </div>
    </div>

    <script>
        let audioEnabled = false;
        let audioCtx = null;
        let lastAlertId = -1;
        let nextBeepTime = 0;
        let beepInterval = 0;
        let isEmergency = false;
        let activeUtterance = null;

        // Tải trước danh sách giọng nói tiếng Việt
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                window.speechSynthesis.getVoices();
            };
        }

        function initAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }

        function playWebBeep(freq = 1200, duration = 0.06) {
            if (!audioEnabled || !audioCtx) return;
            try {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + duration);
            } catch(e) {}
        }

        function speakGuidance(text) {
            if (!audioEnabled || !('speechSynthesis' in window)) return;
            try {
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance(text);
                const voices = window.speechSynthesis.getVoices();
                const viVoice = voices.find(v => (v.lang && (v.lang.startsWith('vi') || v.lang.includes('VIE'))));
                if (viVoice) {
                    u.voice = viVoice;
                }
                u.lang = 'vi-VN';
                u.rate = 1.05;
                // Giữ reference toàn cục tránh Garbage Collector giải phóng sớm trên Chrome
                activeUtterance = u;
                u.onend = () => { activeUtterance = null; };
                u.onerror = () => { activeUtterance = null; };
                window.speechSynthesis.speak(u);
            } catch(e) {}
        }

        function toggleAudio() {
            initAudio();
            audioEnabled = !audioEnabled;
            const btn = document.getElementById('btn-toggle');
            if (audioEnabled) {
                btn.innerText = "🔇 TẮT ÂM THANH TRÌNH DUYỆT";
                btn.className = "btn-audio active";
                playWebBeep(1000, 0.1);
                speakGuidance("Âm thanh đã bật");
            } else {
                btn.innerText = "🔊 BẬT ÂM THANH TRÌNH DUYỆT";
                btn.className = "btn-audio";
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            }
        }

        async function sendDist(v) {
            try { await fetch('/set_dist?val=' + v); } catch(e) {}
        }

        // Vòng lặp phát tiếng bíp khẩn cấp chuẩn thời gian (Yêu cầu 5)
        setInterval(() => {
            const now = Date.now();
            if (isEmergency && audioEnabled && now >= nextBeepTime && beepInterval > 0) {
                playWebBeep(1400, 0.05);
                nextBeepTime = now + (beepInterval * 1000);
            }
        }, 30);

        // Polling trạng thái cảnh báo từ Server (Không nghẽn luồng video)
        setInterval(async () => {
            try {
                const res = await fetch('/alert?t=' + Date.now());
                if (!res.ok) return;
                const data = await res.json();
                const box = document.getElementById('status-box');

                if (data.mode === 'emergency') {
                    isEmergency = true;
                    beepInterval = data.interval || 0.25;
                    box.innerText = "🚨 " + data.text + " (" + data.distance + "m)";
                    box.className = "emergency-banner";
                } else {
                    isEmergency = false;
                    box.className = "";
                    if (data.mode === 'guidance') {
                        box.innerText = "🔊 " + data.text;
                        if (data.id !== lastAlertId) {
                            lastAlertId = data.id;
                            speakGuidance(data.text);
                        }
                    } else {
                        box.innerText = "Đường đi thông thoáng";
                    }
                }
            } catch(e) {}
        }, 150);
    </script>
</body>
</html>
"""


class VoiceRequestHandler(SimpleHTTPRequestHandler):
    manager_instance = None

    def log_message(self, format, *args):
        return  # Tắt log HTTP spam

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif parsed.path == '/alert':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            payload = self.manager_instance.get_alert_state()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        elif parsed.path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            payload = self.manager_instance.get_telemetry()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        elif parsed.path == '/set_dist':
            query = parse_qs(parsed.query)
            if 'val' in query and self.manager_instance.dist_callback:
                try:
                    val = float(query['val'][0])
                    self.manager_instance.dist_callback(val)
                except ValueError:
                    pass
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'OK')
        elif parsed.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                # Bật TCP_NODELAY truyền tức thì không gom gói để giảm lag tối đa
                import socket
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            try:
                last_sent_id = -1
                while getattr(self.manager_instance, 'is_running', True):
                    frame_id, jpeg_bytes = self.manager_instance.get_jpeg_frame_with_id()
                    if jpeg_bytes is not None and frame_id != last_sent_id:
                        self.wfile.write(b'--frame\r\n'
                                         b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                        self.wfile.flush()
                        last_sent_id = frame_id
                    time.sleep(0.015)  # Polling nhanh để gửi ngay khi có khung hình mới
            except (BrokenPipeError, ConnectionResetError, Exception):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/tts':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length) if length > 0 else b'{}'
                obj = json.loads(raw.decode('utf-8'))
                txt = str(obj.get('text', '')).strip()
                ok = self.manager_instance.speak_priority(txt) if txt else False
                self.send_response(200 if ok else 400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"ok": ok, "text": txt}, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode('utf-8'))
        else:
            self.send_error(404)


class ReusableThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Máy chủ HTTP đa luồng (Multi-threaded) chống nghẽn luồng video feed."""
    daemon_threads = True
    allow_reuse_address = True


class AudioManager:
    """
    Quản lý phản hồi thích ứng đa giác quan & chống chồng chéo âm thanh (Zero Audio Overlap).
    """

    def __init__(self, port: int = 8080, guidance_cooldown_s: float = 2.5, enable_local_audio: bool = True):
        self.port = port
        self.guidance_cooldown_s = guidance_cooldown_s
        self.enable_local_audio = enable_local_audio

        # Quản lý vòng đời âm thanh và trạng thái khẩn cấp
        self.is_emergency = False
        self.emergency_distance = 0.0
        self.last_beep_time = 0.0
        self.speech_language = "vi"
        self.speech_rate = 185
        self.speaker_volume = 100
        self.beep_enabled = True

        # Quản lý giọng nói dẫn hướng (Yêu cầu 3 & 5: Đọc 1 lần theo Track ID)
        self.announced_track_ids: Set[int] = set()
        self.last_voice_time = 0.0
        self.current_guidance_text = ""
        self.guidance_id = 0

        # Tiến trình âm thanh cục bộ đang chạy
        self._current_proc: Optional[subprocess.Popen] = None
        self._current_is_priority: bool = False
        self._proc_lock = threading.Lock()
        self._ptt_active: bool = False
        self._ptt_proc: Optional[subprocess.Popen] = None
        self._ptt_lock = threading.Lock()
        self._priority_until: float = 0.0

        # Web Streaming Frame & Telemetry
        self._current_jpeg = None
        self._frame_id = 0
        self._frame_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._telemetry = {}
        self.dist_callback = None
        self.is_running = True

        # Hàng đợi âm thanh tuần tự hoá (Serialized Audio Queue)
        self._speech_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

        # Khởi chạy HTTP Server (Tự động chuyển port nếu 8080 đang bận)
        VoiceRequestHandler.manager_instance = self
        self.server = None
        for p in [self.port, self.port + 1, self.port + 2, 8888]:
            try:
                self.server = ReusableThreadingHTTPServer(('0.0.0.0', p), VoiceRequestHandler)
                self.port = p
                self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.server_thread.start()
                print(f"[AUDIO MANAGER] Web Server & Audio Dashboard tại http://localhost:{self.port}")
                break
            except OSError as e:
                if e.errno == 98 or "already in use" in str(e).lower():
                    continue
                else:
                    break

        if self.server is None:
            print(f"[AUDIO MANAGER] ⚠️ Không thể mở Web Server trên cổng {self.port}. Vui lòng giải phóng port bằng: fuser -k {self.port}/tcp")

    def set_distance_callback(self, cb):
        self.dist_callback = cb

    def update_frame(self, frame_bgr):
        """Cập nhật frame cho Web Dashboard với nén tối ưu giảm độ trễ (Low Latency)."""
        if frame_bgr is None:
            return
        try:
            import cv2
            h, w = frame_bgr.shape[:2]
            # Resize nhẹ về bề rộng tối đa 480px để giảm tải mạng Wi-Fi và bộ đệm Chrome
            if w > 480:
                scale = 480.0 / w
                stream_frame = cv2.resize(frame_bgr, (480, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                stream_frame = frame_bgr

            # Nén JPEG chất lượng 52%: Kích thước giảm từ ~45KB xuống ~12KB (giảm 73% băng thông)
            ret, buf = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 52])
            if ret:
                with self._frame_lock:
                    self._current_jpeg = buf.tobytes()
                    self._frame_id += 1
        except Exception:
            pass

    def get_jpeg_frame(self):
        with self._frame_lock:
            return self._current_jpeg

    def get_jpeg_frame_with_id(self):
        with self._frame_lock:
            return self._frame_id, self._current_jpeg

    def update_telemetry(self, payload: dict):
        """Lưu trạng thái telemetry mới nhất để phục vụ HTTP GET /telemetry."""
        with self._telemetry_lock:
            self._telemetry = dict(payload)

    def get_telemetry(self) -> dict:
        with self._telemetry_lock:
            if self._telemetry:
                return dict(self._telemetry)
        return {"alert": self.get_alert_state()}

    # =========================================================================
    # YÊU CẦU 5: PHẢN HỒI PHÂN TẦNG (KHẨN CẤP vs DẪN HƯỚNG)
    # =========================================================================

    def trigger_emergency(self, distance_m: float):
        """
        KÊNH KHẨN CẤP (< 1.0m) (Yêu cầu 1 & 5):
        - Ngắt ngay lập tức câu nói dẫn hướng đang nói (nhưng không ngắt giọng người thân/PTT).
        - Phát tiếng bíp với tần số & nhịp tăng dần theo cự ly gần.
        """
        self.is_emergency = True
        self.emergency_distance = distance_m

        # 1. Ngắt câu dẫn hướng tự động (nhưng giữ lời nhắc/thoại từ người thân)
        if not self._current_is_priority:
            self._preempt_speech()

        # Khi đang phát giọng người thân (PTT hoặc lời nhắc web), tạm nhường loa để nghe rõ tiếng
        if self._ptt_active or self._current_is_priority or (time.time() < self._priority_until):
            return

        # 2. Tính toán chu kỳ bíp tỉ lệ với khoảng cách (gần = bíp dồn dập)
        # 0.1m -> 0.15s; 0.9m -> 0.50s
        interval = max(0.12, min(0.55, 0.12 + distance_m * 0.40))
        now = time.time()

        if now - self.last_beep_time >= interval:
            self.last_beep_time = now
            self._play_local_beep()

    def clear_emergency(self):
        """Xoá trạng thái khẩn cấp khi cự ly đã an toàn."""
        self.is_emergency = False

    def announce_guidance(self, track_id: int, class_name: str, lane: str, distance_m: float, force: bool = False) -> bool:
        """
        KÊNH DẪN HƯỚNG (1.5m - 3.0m) (Yêu cầu 3 & 5):
        - Dùng 1-2 từ khoá ngắn gọn: [Vật] [Hướng] (ví dụ: "Bàn trước mặt", "Cột bên trái").
        - CHỈ NÓI ĐÚNG 1 LẦN cho mỗi Track ID, sau đó bám vết ngầm.
        - Cho phép phát lại nếu nguy cơ tăng cao đáng kể (force=True).
        - Giữ khoảng lặng tối thiểu giữa các lần nói (cooldown >= 2.5s).
        """
        # Nếu đang ở cự ly khẩn cấp thì không nói giọng nói dài
        if self.is_emergency or distance_m < 1.0:
            return False

        # YÊU CẦU 3: Đã đọc Track ID này rồi thì im lặng theo dõi ngầm (100% chống lặp), trừ khi force=True
        if track_id in self.announced_track_ids and not force:
            return False

        now = time.time()
        # Chống nói liến thoắng gây mệt mỏi nhận thức
        if now - self.last_voice_time < self.guidance_cooldown_s:
            return False

        # Đang có tiến trình nói đang chạy thì bỏ qua
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                return False

        # Định dạng câu nói 1-2 từ khoá cực kỳ súc tích
        obj_name = LABEL_SHORT_VI.get(class_name, class_name)
        lane_name = LANE_SHORT_VI.get(lane, "trước mặt")
        speech_text = f"{obj_name} {lane_name}"

        # Đánh dấu đã đọc và dọn dẹp bộ nhớ nếu lưu trữ quá nhiều
        self.announced_track_ids.add(track_id)
        if len(self.announced_track_ids) > 300:
            self.announced_track_ids = {track_id}
        self.last_voice_time = now
        self.guidance_id += 1
        self.current_guidance_text = speech_text

        print(f"🔊 [DẪN HƯỚNG #{self.guidance_id}] (Track ID {track_id}): '{speech_text}' ({distance_m:.1f}m)")

        # Đưa vào hàng đợi phát âm thanh tuần tự
        try:
            self._speech_queue.put_nowait((speech_text, False))
            return True
        except queue.Full:
            return False

    def speak_priority(self, text: str) -> bool:
        """Phát ngay lời nhắc thoại ưu tiên từ người thân (Web Dashboard)."""
        msg = (text or "").strip()
        if not msg:
            return False
        self._priority_until = time.time() + 4.0
        self.last_voice_time = time.time()
        self.guidance_id += 1
        self.current_guidance_text = msg
        print(f"📢 [NGƯỜI THÂN NHẮC]: '{msg}'")

        # Ngắt câu dẫn hướng cũ & làm trống hàng đợi để phát ngay lời nhắc mới
        self._preempt_speech()
        while True:
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._speech_queue.put_nowait((msg, True))
            return True
        except queue.Full:
            return False

    def start_ptt(self):
        """Mở luồng phát âm thanh trực tiếp (PCM16 16kHz) khi người thân nhấn giữ micro."""
        self._ptt_active = True
        self._preempt_speech()
        if not self.enable_local_audio or not shutil.which("aplay"):
            return
        with self._ptt_lock:
            if self._ptt_proc is not None and self._ptt_proc.poll() is None:
                return
            try:
                self._ptt_proc = subprocess.Popen(
                    ["aplay", "-q", "-r", "16000", "-f", "S16_LE", "-c", "1", "-t", "raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                self._ptt_proc = None

    def feed_ptt_pcm(self, raw_pcm: bytes):
        """Đưa các gói PCM16 từ Web Dashboard ra loa Raspberry Pi."""
        if not self.enable_local_audio or not raw_pcm:
            return
        with self._ptt_lock:
            proc = self._ptt_proc
        if proc is None or proc.poll() is not None:
            self.start_ptt()
            with self._ptt_lock:
                proc = self._ptt_proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.write(raw_pcm)
                proc.stdin.flush()
            except Exception:
                self.stop_ptt()

    def stop_ptt(self):
        """Đóng luồng âm thanh Push-to-Talk."""
        self._ptt_active = False
        with self._ptt_lock:
            proc = self._ptt_proc
            self._ptt_proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    # =========================================================================
    # QUẢN LÝ TIẾN TRÌNH ÂM THANH CỤC BỘ (ZERO OVERLAPPING GUARANTEE)
    # =========================================================================

    def _play_local_beep(self):
        """Phát 1 tiếng bíp cục bộ trên máy tính/Raspberry Pi."""
        if not self.enable_local_audio or not self.beep_enabled:
            return

        def _worker():
            try:
                if IS_MAC and os.path.exists(BEEP_WAV_PATH):
                    subprocess.run(["afplay", "-q", "1", BEEP_WAV_PATH],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif shutil.which("aplay") and os.path.exists(BEEP_WAV_PATH):
                    subprocess.run(["aplay", "-q", BEEP_WAV_PATH],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # Terminal beep fallback
                    sys.stdout.write('\a')
                    sys.stdout.flush()
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def set_system_volume(self, volume_pct: int):
        """Áp dụng mức âm lượng loa thật trên Raspberry Pi qua PipeWire (wpctl) hoặc ALSA (amixer)."""
        pct = max(0, min(100, int(volume_pct)))
        self.speaker_volume = pct
        if not self.enable_local_audio:
            return
        try:
            if shutil.which("wpctl"):
                subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct / 100.0:.2f}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            elif shutil.which("amixer"):
                for ctrl in ("PCM", "Headphone", "Master"):
                    subprocess.run(["amixer", "sset", ctrl, f"{pct}%", "unmute"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

    def apply_audio_config(self, audio_cfg: dict):
        """Cập nhật thông số âm thanh từ Web Dashboard hoặc config.yaml."""
        if not isinstance(audio_cfg, dict):
            return
        if "language" in audio_cfg:
            self.speech_language = str(audio_cfg["language"] or "vi")
        if "rate" in audio_cfg:
            try:
                self.speech_rate = max(100, min(260, int(audio_cfg["rate"])))
            except Exception:
                pass
        if "beep_enabled" in audio_cfg:
            self.beep_enabled = bool(audio_cfg["beep_enabled"])
        if "speaker_volume" in audio_cfg:
            try:
                self.set_system_volume(int(audio_cfg["speaker_volume"]))
            except Exception:
                pass

    def _preempt_speech(self):
        """Ngắt ngay tiến trình nói khi có sự kiện khẩn cấp."""
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
                self._current_proc = None
            self._current_is_priority = False

    def _speech_worker(self):
        """Luồng chuyên trách phát âm thanh duy nhất để không bao giờ bị đè tiếng."""
        while self.is_running:
            try:
                item = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if isinstance(item, tuple):
                text, is_priority = item
            else:
                text, is_priority = item, False

            if not self.enable_local_audio or self._ptt_active:
                continue
            if self.is_emergency and not is_priority:
                continue  # Bỏ qua dẫn hướng tự động khi đang khẩn cấp

            proc = None
            with self._proc_lock:
                self._current_is_priority = is_priority
                try:
                    if IS_MAC:
                        self._current_proc = subprocess.Popen(
                            ["say", "-v", "Linh", text],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                    elif shutil.which("espeak-ng"):
                        self._current_proc = subprocess.Popen(
                            ["espeak-ng", "-v", self.speech_language, "-s", str(self.speech_rate), "-a", "180", text],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                    elif shutil.which("espeak"):
                        self._current_proc = subprocess.Popen(
                            ["espeak", "-v", self.speech_language, "-s", str(self.speech_rate), "-a", "180", text],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                except Exception:
                    self._current_proc = None
                proc = self._current_proc

            if proc:
                try:
                    proc.wait(timeout=4.0)
                except Exception:
                    self._preempt_speech()
            with self._proc_lock:
                if self._current_proc is proc:
                    self._current_proc = None
                self._current_is_priority = False
                if is_priority:
                    self._priority_until = 0.0

    def get_alert_state(self) -> dict:
        """Đóng gói trạng thái gửi cho Web Dashboard."""
        if self.is_emergency:
            interval = max(0.12, min(0.55, 0.12 + self.emergency_distance * 0.40))
            return {
                "mode": "emergency",
                "text": "DỪNG LẠI!",
                "distance": round(self.emergency_distance, 2),
                "interval": round(interval, 2)
            }
        elif self.current_guidance_text and (time.time() - self.last_voice_time < 3.5):
            return {
                "mode": "guidance",
                "id": self.guidance_id,
                "text": self.current_guidance_text
            }
        else:
            return {
                "mode": "safe",
                "text": "An toàn"
            }

    def stop(self):
        self.is_running = False
        self.stop_ptt()
        self._preempt_speech()
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass

