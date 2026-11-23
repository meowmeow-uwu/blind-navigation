#!/usr/bin/env python3
"""
stream_webcam.py - Web Stream USB Webcam có tính năng GHI HÌNH TRỰC TIẾP VỀ MÁY TÍNH
==================================================================================
Điểm nổi bật:
1. GHI HÌNH 100% TRÊN MÁY TÍNH (Client-Side MediaRecorder):
   - Raspberry Pi KHÔNG TỐN 1 BYTE DUNG LƯỢNG NÀO trên thẻ nhớ.
   - Trình duyệt (Chrome/Edge) trên laptop của bạn tự nhận luồng video, tự nén bằng
     phần cứng máy tính và lưu thẳng vào thư mục Downloads của bạn.
2. Pi chỉ làm nhiệm vụ duy nhất là gửi luồng video mượt mà, CPU nhẹ mát.
"""

import os
import sys
import time
import json
import socket
import select
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import cv2
except ImportError:
    print("[ERROR] Chưa cài đặt OpenCV (cv2)!")
    sys.exit(1)


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raspberry Pi Webcam Stream & Direct PC Recorder</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #111418;
            color: #eee;
            margin: 0;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            max-width: 920px;
            width: 100%;
            background: #1a1e24;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            overflow: hidden;
            border: 1px solid #2d3540;
        }
        header {
            padding: 14px 20px;
            background: #222831;
            border-bottom: 1px solid #2d3540;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        h1 {
            font-size: 1.15rem;
            margin: 0;
            color: #00e676;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .live-dot {
            width: 10px;
            height: 10px;
            background: #00e676;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px #00e676;
        }
        .stream-box {
            position: relative;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 280px;
            max-height: 52vh;
            overflow: hidden;
        }
        img#mjpg-view {
            max-width: 100%;
            max-height: 52vh;
            width: auto;
            height: auto;
            display: block;
            object-fit: contain;
        }
        canvas#record-canvas {
            display: none; /* Canvas ngầm dùng để capture frame cho MediaRecorder */
        }
        .rec-indicator {
            position: absolute;
            top: 14px;
            right: 14px;
            background: rgba(230, 0, 0, 0.9);
            color: #fff;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: bold;
            display: none;
            align-items: center;
            gap: 8px;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
            animation: pulse-red 1.2s infinite;
        }
        @keyframes pulse-red {
            0% { opacity: 0.85; transform: scale(0.98); }
            50% { opacity: 1; transform: scale(1.02); }
            100% { opacity: 0.85; transform: scale(0.98); }
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            padding: 14px 20px;
            background: #171b21;
        }
        .stat-card {
            background: #222831;
            border: 1px solid #2d3540;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        .stat-label {
            font-size: 0.7rem;
            color: #8b98a5;
            text-transform: uppercase;
            margin-bottom: 3px;
        }
        .stat-val {
            font-size: 1.3rem;
            font-weight: 700;
            color: #f1f1f1;
            font-variant-numeric: tabular-nums;
        }
        .stat-val.highlight { color: #00e676; }
        .controls-top {
            padding: 12px 18px;
            background: #1c222b;
            border-bottom: 1px solid #2d3644;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .btn-toolbar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        button {
            background: #28313d;
            color: #eee;
            border: 1px solid #3d4a5c;
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        button:hover { background: #3d4a5c; color: #fff; }
        button.btn-record {
            background: #e53935;
            color: #fff;
            border-color: #ff5252;
            font-weight: bold;
            box-shadow: 0 0 12px rgba(229, 57, 53, 0.4);
        }
        button.btn-record:hover { background: #c62828; }
        button.btn-record.recording {
            background: #ff1744;
            animation: pulse-border 1.5s infinite;
        }
        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7); }
            70% { box-shadow: 0 0 0 12px rgba(255, 23, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }
        }
        .badge-pc {
            background: #0288d1;
            color: #fff;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            letter-spacing: 0.5px;
        }
        .note-card {
            padding: 8px 14px;
            background: #171b21;
            border-top: 1px solid #222831;
            font-size: 0.8rem;
            color: #aaa;
            line-height: 1.4;
        }
        #save-status-box {
            display: none;
            background: #1b382b;
            border: 1px solid #00e676;
            color: #e8f5e9;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="live-dot"></span> LIVE WEBCAM FEED</h1>
            <div style="font-size: 0.85rem; color: #aaa;">
                Lưu trực tiếp: <strong style="color: #4fc3f7;">Ổ cứng Máy tính / Laptop (0 byte trên Pi)</strong>
            </div>
        </header>

        <!-- THANH ĐIỀU KHIỂN ĐẶT NGAY TRÊN ĐẦU, KHÔNG CẦN CUỘN CHUỘT -->
        <div class="controls-top">
            <div class="btn-toolbar">
                <button id="btn-record" class="btn-record" onclick="toggleClientRecord()">
                    <span>🔴</span> <span id="btn-record-text">Bắt đầu Quay Video</span>
                    <span class="badge-pc">LƯU VÀO MÁY TÍNH</span>
                </button>
                <button onclick="downloadClientSnapshot()">
                    <span>📷</span> Chụp ảnh Lưu về Máy
                </button>
                <button onclick="toggleOsd()" id="btn-osd">
                    <span>🔤</span> Chữ OSD: BẬT
                </button>
                <button onclick="reloadStream()">
                    <span>🔄</span> Làm mới Stream
                </button>
            </div>

            <div id="save-status-box">
                ✅ <strong>Đã ghi hình thành công!</strong> File video đã được lưu trực tiếp vào thư mục <strong>Downloads trên máy tính</strong>. Thẻ nhớ Raspberry Pi không tốn 1 byte nào!
            </div>
        </div>

        <div class="stream-box">
            <img id="mjpg-view" src="/stream.mjpg" crossOrigin="anonymous" alt="Đang kết nối camera..." />
            <canvas id="record-canvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
            <div class="rec-indicator" id="rec-badge">
                <span>🔴</span> ĐANG GHI VÀO MÁY TÍNH: <span id="rec-timer">00:00</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">FPS CAMERA</div>
                <div class="stat-val highlight" id="stat-fps">-- FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">TRẠNG THÁI GHI</div>
                <div class="stat-val" id="stat-rec-status" style="color: #aaa; font-size: 1.1rem;">SẴN SÀNG</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">BỘ NHỚ TRÊN RASPBERRY PI</div>
                <div class="stat-val" style="color: #00e676;">0 MB (KHÔNG TỐN)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">ĐỘ SÁNG</div>
                <div class="stat-val" id="stat-bright">-- / 255</div>
            </div>
        </div>

        <div class="note-card">
            💡 <strong>Cơ chế Direct-to-PC (HTML5 MediaRecorder):</strong> Video được nén bằng card đồ họa của laptop và lưu thẳng vào ổ cứng máy tính. Raspberry Pi chỉ gửi stream qua mạng, 0 byte bộ nhớ, CPU mát mẻ!
        </div>
    </div>
    </div>

    <script>
        const imgView = document.getElementById('mjpg-view');
        const canvas = document.getElementById('record-canvas');
        const ctx = canvas.getContext('2d');

        let mediaRecorder = null;
        let recordedChunks = [];
        let isRecording = false;
        let recordInterval = null;
        let recordSeconds = 0;
        let drawLoopActive = false;

        function reloadStream() {
            imgView.src = '/stream.mjpg?t=' + Date.now();
        }

        // Vòng lặp vẽ ảnh từ <img> sang <canvas> khi đang ghi hình
        function drawFrameToCanvas() {
            if (!isRecording) return;
            try {
                if (imgView.complete && imgView.naturalWidth > 0) {
                    ctx.drawImage(imgView, 0, 0, canvas.width, canvas.height);
                }
            } catch(e) {}
            requestAnimationFrame(drawFrameToCanvas);
        }

        // BẮT ĐẦU / DỪNG GHI HÌNH TRỰC TIẾP TRÊN TRÌNH DUYỆT (LƯU VÀO MÁY TÍNH)
        function toggleClientRecord() {
            const btn = document.getElementById('btn-record');
            const btnText = document.getElementById('btn-record-text');
            const recBadge = document.getElementById('rec-badge');
            const recStatus = document.getElementById('stat-rec-status');
            const alertBox = document.getElementById('save-status-box');

            if (!isRecording) {
                // 1. Khởi tạo Stream từ Canvas của trình duyệt
                recordedChunks = [];
                const stream = canvas.captureStream(25); // 25 FPS stream

                // Chọn MIME type tốt nhất được trình duyệt hỗ trợ
                let mimeType = 'video/webm;codecs=vp9';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'video/webm;codecs=vp8';
                }
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'video/webm';
                }

                try {
                    mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
                } catch(e) {
                    mediaRecorder = new MediaRecorder(stream);
                }

                mediaRecorder.ondataavailable = function(event) {
                    if (event.data && event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = function() {
                    // Tạo file video từ RAM của trình duyệt và lưu thẳng vào ổ cứng máy tính
                    const blob = new Blob(recordedChunks, { type: 'video/webm' });
                    const url = URL.createObjectURL(blob);
                    const now = new Date();
                    const timestamp = now.getFullYear() +
                        String(now.getMonth() + 1).padStart(2, '0') +
                        String(now.getDate()).padStart(2, '0') + '_' +
                        String(now.getHours()).padStart(2, '0') +
                        String(now.getMinutes()).padStart(2, '0') +
                        String(now.getSeconds()).padStart(2, '0');
                    
                    const filename = `webcam_record_${timestamp}.webm`;

                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);

                    alertBox.style.display = 'block';
                };

                mediaRecorder.start(500); // Lưu mỗi chunk 500ms
                isRecording = true;
                drawFrameToCanvas();

                // Cập nhật giao diện
                btn.classList.add('recording');
                btnText.innerText = 'Dừng & Lưu vào Máy Tính';
                recBadge.style.display = 'flex';
                alertBox.style.display = 'none';

                recordSeconds = 0;
                recordInterval = setInterval(() => {
                    recordSeconds++;
                    const min = Math.floor(recordSeconds / 60).toString().padStart(2, '0');
                    const sec = (recordSeconds % 60).toString().padStart(2, '0');
                    document.getElementById('rec-timer').innerText = `${min}:${sec}`;
                    recStatus.innerText = `ĐANG GHI (${min}:${sec})`;
                    recStatus.style.color = '#f44336';
                }, 1000);

            } else {
                // DỪNG GHI HÌNH
                isRecording = false;
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                clearInterval(recordInterval);

                btn.classList.remove('recording');
                btnText.innerText = 'Bắt đầu Quay Video';
                recBadge.style.display = 'none';
                recStatus.innerText = 'SẴN SÀNG';
                recStatus.style.color = '#aaa';
            }
        }

        // Chụp ảnh trực tiếp lưu thẳng về máy tính
        function downloadClientSnapshot() {
            ctx.drawImage(imgView, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(function(blob) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'snapshot_' + Date.now() + '.jpg';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 'image/jpeg', 0.95);
        }

        async function fetchStats() {
            try {
                const res = await fetch('/stats');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('stat-fps').innerText = data.fps.toFixed(1) + ' FPS';
                    document.getElementById('stat-bright').innerText = Math.round(data.mean_brightness) + ' / 255';

                    const btnOsd = document.getElementById('btn-osd');
                    btnOsd.innerText = data.show_osd ? '🔤 Chữ OSD: BẬT' : '🔤 Chữ OSD: TẮT';
                }
            } catch(e) {}
        }

        async function toggleOsd() {
            try {
                await fetch('/toggle_osd');
                fetchStats();
            } catch(e) {}
        }

        setInterval(fetchStats, 500);
        fetchStats();
    </script>
</body>
</html>
"""


class FastCameraReader:
    """Chỉ làm duy nhất nhiệm vụ bắt hình và phát stream (Không tốn CPU ghi video trên Pi)"""
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.running = False
        self.latest_jpeg = None
        self.latest_raw_frame = None

        self.measured_fps = 0.0
        self.frame_count = 0
        self.mean_brightness = 0.0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.frames_since_fps = 0
        self.show_osd = True

        if self.cap.isOpened():
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True, name="CamStreamWorker")
            self.thread.start()

    def _worker(self):
        while self.running:
            ret, frame = self.cap.read()
            now = time.time()

            if ret and frame is not None:
                self.frame_count += 1
                self.frames_since_fps += 1

                # Đo FPS mỗi 0.5s
                dt = now - self.last_fps_time
                if dt >= 0.5:
                    self.measured_fps = self.frames_since_fps / dt
                    self.frames_since_fps = 0
                    self.last_fps_time = now
                    self.mean_brightness = float(cv2.mean(frame)[0])

                display_frame = frame
                if self.show_osd:
                    display_frame = frame.copy()
                    osd_text = f"FPS: {self.measured_fps:4.1f} | {self.actual_w}x{self.actual_h}"
                    cv2.rectangle(display_frame, (8, 8), (250, 36), (0, 0, 0), -1)
                    cv2.putText(display_frame, osd_text, (14, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                ret_enc, jpeg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret_enc:
                    with self.lock:
                        self.latest_jpeg = jpeg.tobytes()
                        self.latest_raw_frame = frame
                    self.new_frame_event.set()
            else:
                time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def toggle_osd(self):
        self.show_osd = not self.show_osd
        return self.show_osd

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


camera = None
SERVER_WIDTH = 640
SERVER_HEIGHT = 480


class StreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
        except Exception:
            pass

    def do_GET(self):
        global camera, SERVER_WIDTH, SERVER_HEIGHT

        if self.path in ("/", "/index.html"):
            html = HTML_PAGE.replace("{WIDTH}", str(SERVER_WIDTH))\
                            .replace("{HEIGHT}", str(SERVER_HEIGHT))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                while True:
                    camera.new_frame_event.wait(timeout=0.1)
                    camera.new_frame_event.clear()

                    _, writable, _ = select.select([], [self.connection], [], 0.0)
                    if not writable:
                        continue

                    frame_bytes = camera.get_frame()
                    if frame_bytes is not None:
                        header = f"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame_bytes)}\r\n\r\n".encode("utf-8")
                        self.wfile.write(header)
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif self.path == "/stats":
            if camera is not None:
                uptime = int(time.time() - camera.start_time)
                data = {
                    "fps": round(camera.measured_fps, 1),
                    "resolution": f"{camera.actual_w}x{camera.actual_h}",
                    "frames": camera.frame_count,
                    "mean_brightness": round(camera.mean_brightness, 1),
                    "show_osd": camera.show_osd
                }
            else:
                data = {"fps": 0.0, "resolution": "0x0", "frames": 0, "mean_brightness": 0}

            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/toggle_osd":
            if camera is not None:
                camera.toggle_osd()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    global camera, SERVER_WIDTH, SERVER_HEIGHT

    parser = argparse.ArgumentParser(description="Webcam Live Stream (Direct PC Recording)")
    parser.add_argument("-i", "--index", type=int, default=0, help="Camera index (/dev/videoX)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port web server (mặc định: 8080)")
    parser.add_argument("-W", "--width", type=int, default=640, help="Độ rộng (mặc định: 640)")
    parser.add_argument("-H", "--height", type=int, default=480, help="Độ cao (mặc định: 480)")
    parser.add_argument("-f", "--fps", type=int, default=30, help="FPS mục tiêu (mặc định: 30)")
    args = parser.parse_args()

    os.system(f"v4l2-ctl -d /dev/video{args.index} --set-ctrl=auto_exposure=3 2>/dev/null")

    print("=" * 68)
    print("   WEBCAM STREAM & GHI HÌNH TRỰC TIẾP VÀO MÁY TÍNH (0 BYTE TRÊN PI)")
    print("=" * 68)
    print(f"[*] Đang mở camera /dev/video{args.index}...")

    camera = FastCameraReader(index=args.index, width=args.width, height=args.height, fps=args.fps)
    if not camera.cap.isOpened():
        print(f"\n[ERROR] Không thể mở camera index {args.index}!")
        print("-> Hãy bấm Ctrl+C ở terminal cũ để tắt tiến trình đang chiếm camera.")
        sys.exit(1)

    SERVER_WIDTH = camera.actual_w
    SERVER_HEIGHT = camera.actual_h

    local_ip = get_local_ip()
    port = args.port

    server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)

    print(f"[OK] Camera đã mở: {SERVER_WIDTH}x{SERVER_HEIGHT}")
    print("-" * 68)
    print("  TRUY CẬP TRÌNH DUYỆT ĐỂ XEM VÀ QUAY VIDEO LƯU VÀO MÁY TÍNH:")
    print()
    print(f"  👉 Truy cập Wi-Fi : http://{local_ip}:{port}")
    print(f"  👉 Hoặc Localhost : http://localhost:{port}")
    print()
    print("  (Nhấn Ctrl+C trong terminal để dừng phát)")
    print("=" * 68)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Đang dừng máy chủ...")
    finally:
        server.server_close()
        camera.release()
        print("[*] Đã đóng camera thành công.")


if __name__ == "__main__":
    main()
