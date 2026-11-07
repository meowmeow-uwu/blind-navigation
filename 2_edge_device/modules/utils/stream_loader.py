# 2_edge_device/modules/utils/stream_loader.py
from typing import Generator, Optional, Tuple, Union
import cv2
import time
import numpy as np


class StreamLoader:
    """
    Tải và cấp phát frame từ Webcam hoặc file Video offline.

    Hỗ trợ:
      - source=0 (hoặc int) → Webcam
      - source="path/to/video.mp4" → File video offline với đồng bộ FPS thực

    Sử dụng như context manager:
        with StreamLoader(source=0, width=640, height=480) as stream:
            for frame in stream:
                process(frame)
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps_sync: bool = True
    ):
        """
        Args:
            source: 0 cho webcam, hoặc đường dẫn file video (str).
            width: Chiều rộng mong muốn (chỉ áp dụng cho webcam).
            height: Chiều cao mong muốn (chỉ áp dụng cho webcam).
            fps_sync: Nếu True và đọc từ file video, điều phối tốc độ đọc
                      khớp với FPS gốc của video, tránh xử lý nhanh hơn thời gian thực.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps_sync = fps_sync

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_video_file = isinstance(source, str)
        self._frame_interval: float = 0.0
        self._last_read_time: float = 0.0

    def open(self) -> None:
        """Mở nguồn video/webcam."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(
                f"[StreamLoader] Không thể mở nguồn: {self.source!r}. "
                "Kiểm tra webcam/đường dẫn file."
            )

        if not self._is_video_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self._is_video_file and self.fps_sync:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._frame_interval = 1.0 / fps if fps > 0 else 0.033
        else:
            self._frame_interval = 0.0

        self._last_read_time = time.perf_counter()

    def close(self) -> None:
        """Đóng nguồn video và giải phóng tài nguyên."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "StreamLoader":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def meta(self) -> dict:
        """Trả về metadata của nguồn (width, height, fps thực tế)."""
        if self._cap is None:
            return {}
        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "is_video_file": self._is_video_file
        }

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterator trả về từng frame BGR.
        Kết thúc tự động khi hết file hoặc mất tín hiệu webcam.
        """
        if self._cap is None:
            self.open()

        while True:
            # Đồng bộ FPS khi đọc file video offline
            if self._is_video_file and self._frame_interval > 0:
                elapsed = time.perf_counter() - self._last_read_time
                sleep_time = self._frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            ret, frame = self._cap.read()
            self._last_read_time = time.perf_counter()

            if not ret:
                break

            yield frame
