"""
dashboard_audio_client.py - Client WebSocket nhẹ kết nối tới Web Dashboard (/ws/edge_audio)
=========================================================================================
Nhận lệnh đọc TTS và luồng âm thanh Push-to-Talk (PCM16 16kHz) từ người thân trên
3_web_dashboard để phát trực tiếp ra loa Raspberry Pi mà không cần thêm thư viện ngoài.
"""

import os
import json
import time
import base64
import socket
import struct
import threading
from urllib.parse import urlparse
from typing import Optional


class DashboardAudioClient:
    """Client RFC 6455 thuần thư viện chuẩn kết nối /ws/edge_audio."""

    def __init__(self, dashboard_url: str, audio_manager):
        self.dashboard_url = (dashboard_url or "").strip().rstrip("/")
        self.audio_manager = audio_manager
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self.connected = False

    def start(self):
        if not self.dashboard_url or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="DashboardAudioWS", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.connected = False
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _parse_target(self):
        url = self.dashboard_url
        if not url.startswith(("http://", "https://", "ws://", "wss://")):
            url = "http://" + url
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme in ("https", "wss") else 8088)
        base = parsed.path.rstrip("/")
        path = f"{base}/ws/edge_audio"
        return host, port, path

    def _send_frame(self, sock: socket.socket, opcode: int, payload: bytes = b""):
        mask_key = os.urandom(4)
        n = len(payload)
        header = bytearray([0x80 | (opcode & 0x0F)])
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        with self._send_lock:
            sock.sendall(bytes(header) + masked)

    def _recv_exact(self, sock: socket.socket, buf: bytearray, n: int) -> Optional[bytes]:
        while len(buf) < n:
            if not self._running:
                return None
            try:
                chunk = sock.recv(max(4096, n - len(buf)))
            except socket.timeout:
                return b"" if len(buf) == 0 else None
            if not chunk:
                return None
            buf.extend(chunk)
        out = bytes(buf[:n])
        del buf[:n]
        return out

    def _read_frame(self, sock: socket.socket, buf: bytearray):
        """Trả về (fin, opcode, payload), hoặc ('TIMEOUT', 0, b'') nếu hết timeout, hoặc None nếu đóng."""
        while len(buf) < 2:
            if not self._running:
                return None
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                if len(buf) == 0:
                    return ("TIMEOUT", 0, b"")
                continue
            if not chunk:
                return None
            buf.extend(chunk)

        b0, b1 = buf[0], buf[1]
        del buf[:2]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F

        if length == 126:
            ext = self._recv_exact(sock, buf, 2)
            if not ext:
                return None
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = self._recv_exact(sock, buf, 8)
            if not ext:
                return None
            length = struct.unpack("!Q", ext)[0]

        mask_key = b""
        if masked:
            mk = self._recv_exact(sock, buf, 4)
            if not mk:
                return None
            mask_key = mk

        payload = b""
        if length > 0:
            while len(buf) < length:
                if not self._running:
                    return None
                try:
                    chunk = sock.recv(max(4096, length - len(buf)))
                except socket.timeout:
                    continue
                if not chunk:
                    return None
                buf.extend(chunk)
            payload = bytes(buf[:length])
            del buf[:length]
            if masked:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))

        return (fin, opcode, payload)

    def _handle_text(self, text: str):
        try:
            obj = json.loads(text)
        except Exception:
            return
        tp = obj.get("type")
        if tp == "tts":
            msg = str(obj.get("text", "")).strip()
            if msg:
                self.audio_manager.speak_priority(msg)
        elif tp == "ptt_start":
            self.audio_manager.start_ptt()
        elif tp == "ptt_stop":
            self.audio_manager.stop_ptt()
        elif tp == "config":
            data = obj.get("data", {})
            if isinstance(data, dict) and isinstance(data.get("audio"), dict):
                self.audio_manager.apply_audio_config(data["audio"])

    def _run_loop(self):
        host, port, path = self._parse_target()
        announced_once = False
        while self._running:
            sock = None
            try:
                sock = socket.create_connection((host, port), timeout=4.0)
                key = base64.b64encode(os.urandom(16)).decode("ascii")
                req = (
                    f"GET {path} HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    f"Upgrade: websocket\r\n"
                    f"Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {key}\r\n"
                    f"Sec-WebSocket-Version: 13\r\n\r\n"
                ).encode("ascii")
                sock.sendall(req)

                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    chunk = sock.recv(2048)
                    if not chunk:
                        raise ConnectionError("WebSocket handshake closed")
                    buf.extend(chunk)
                    if len(buf) > 16384:
                        raise ConnectionError("Handshake header too large")

                hdr_end = buf.find(b"\r\n\r\n") + 4
                hdr = bytes(buf[:hdr_end]).decode("latin1", errors="ignore")
                del buf[:hdr_end]
                if " 101 " not in hdr.splitlines()[0]:
                    raise ConnectionError(f"Unexpected handshake response: {hdr.splitlines()[0]}")

                self._sock = sock
                sock.settimeout(1.0)
                self.connected = True
                if not announced_once:
                    print(f"[AUDIO BRIDGE] ✅ Đã kết nối WebSocket âm thanh tới Web Dashboard: ws://{host}:{port}{path}")
                    announced_once = True

                last_ping = time.time()
                frag_opcode = None
                frag_buf = bytearray()

                while self._running:
                    frm = self._read_frame(sock, buf)
                    if frm is None:
                        break
                    fin, opcode, payload = frm
                    if fin == "TIMEOUT":
                        if time.time() - last_ping >= 15.0:
                            self._send_frame(sock, 0x9, b"hb")
                            last_ping = time.time()
                        continue

                    if opcode == 0x8:  # CLOSE
                        break
                    elif opcode == 0x9:  # PING
                        self._send_frame(sock, 0xA, payload)
                        continue
                    elif opcode == 0xA:  # PONG
                        continue

                    if opcode == 0x0 and frag_opcode is not None:
                        frag_buf.extend(payload)
                        if not fin:
                            continue
                        opcode = frag_opcode
                        payload = bytes(frag_buf)
                        frag_opcode = None
                        frag_buf.clear()
                    elif not fin and opcode in (0x1, 0x2):
                        frag_opcode = opcode
                        frag_buf = bytearray(payload)
                        continue

                    if opcode == 0x1:
                        self._handle_text(payload.decode("utf-8", errors="ignore"))
                    elif opcode == 0x2:
                        self.audio_manager.feed_ptt_pcm(payload)

            except Exception:
                pass
            finally:
                self.connected = False
                self._sock = None
                try:
                    self.audio_manager.stop_ptt()
                except Exception:
                    pass
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
            if self._running:
                time.sleep(2.0)
