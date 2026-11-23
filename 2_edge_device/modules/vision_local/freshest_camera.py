"""
freshest_camera.py - Bộ thu hình camera độ trễ cực thấp (Zero-Latency Frame Grabber)
=====================================================================================
Giải quyết triệt để vấn đề:
1. Trễ do V4L2 Buffer Queue trên Linux / Raspberry Pi:
   Mặc định OpenCV đệm 4-5 frame trong hàng đợi driver V4L2. Khi AI xử lý mất 100ms,
   camera vẫn quay ở 30 FPS khiến hàng đợi luôn đầy frame cũ -> Video bị trễ 0.5s - 1.5s
   so với thực tế (lia camera từ A sang B nhưng một lúc sau màn hình mới nhảy đến B).
2. Luồng đọc nền (Background Thread) liên tục đọc và giải phóng buffer V4L2 ở 30 FPS,
   chỉ lưu giữ duy nhất khung hình MỚI NHẤT trong RAM.
3. Khi AI hoặc luồng chính cần frame, nó nhận được khung hình được chụp cách đó < 15ms.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FreshestCameraReader:
    """
    Trình bao bọc cv2.VideoCapture đa luồng giúp loại bỏ hoàn toàn độ trễ tích lũy.
    """
    def __init__(self, cap):
        self.cap = cap
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        if self.cap and self.cap.isOpened():
            # Cố gắng đặt buffer size = 1 nếu driver V4L2 hỗ trợ
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.start()

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._worker, daemon=True, name="FreshestCamWorker")
            self._thread.start()

    def _worker(self):
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
            else:
                time.sleep(0.005)

    def read(self, timeout: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
        """Lấy khung hình tươi mới nhất vừa được chụp từ phần cứng."""
        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame
        return False, None

    def isOpened(self) -> bool:
        return self.cap.isOpened() if self.cap else False

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        if self.cap and self.cap.isOpened():
            self.cap.release()
