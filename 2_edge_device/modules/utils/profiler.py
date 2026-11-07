# 2_edge_device/modules/utils/profiler.py
from typing import Dict, List, Optional
import time
from collections import deque
from contextlib import contextmanager
import cv2
import numpy as np


class PipelineProfiler:
    """
    Bộ đo và phân tích hiệu năng (Profiler) đa công đoạn:
    capture, inference, tracking, policy, render.
    Tính toán trung bình trượt (rolling window) và hiển thị HUD trực quan.
    """

    STAGES = ["capture", "inference", "tracking", "policy", "render"]

    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.latencies: Dict[str, deque] = {
            stage: deque(maxlen=self.buffer_size) for stage in self.STAGES
        }
        self.frame_times: deque = deque(maxlen=self.buffer_size)
        self._last_frame_timestamp: Optional[float] = None

    @contextmanager
    def measure(self, stage: str):
        """Context manager đo độ trễ cho một công đoạn xác định."""
        t_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if stage in self.latencies:
                self.latencies[stage].append(elapsed_ms)
            else:
                self.latencies[stage] = deque([elapsed_ms], maxlen=self.buffer_size)

    def tick_frame(self) -> None:
        """Ghi nhận mốc thời gian hoàn tất một frame để tính tổng FPS thực tế."""
        now = time.perf_counter()
        if self._last_frame_timestamp is not None:
            delta = now - self._last_frame_timestamp
            if delta > 0:
                self.frame_times.append(delta)
        self._last_frame_timestamp = now

    def get_stats(self) -> Dict[str, float]:
        """
        Lấy thống kê trung bình độ trễ (ms) từng stage và FPS tổng.
        """
        stats: Dict[str, float] = {}
        total_pipeline_ms = 0.0

        for stage, times in self.latencies.items():
            avg_ms = float(np.mean(times)) if len(times) > 0 else 0.0
            stats[stage] = avg_ms
            total_pipeline_ms += avg_ms

        stats["total_pipeline_ms"] = total_pipeline_ms

        # FPS tính theo chu kỳ frame thực tế hoặc theo tổng pipeline time
        if len(self.frame_times) > 0:
            avg_delta = float(np.mean(self.frame_times))
            stats["fps"] = 1.0 / avg_delta if avg_delta > 0 else 0.0
        else:
            stats["fps"] = 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0.0

        return stats

    def draw_hud(self, frame: np.ndarray, x: int = 15, y: int = 15) -> np.ndarray:
        """
        Vẽ bảng thông số phân rã độ trễ (Latency Breakdown) bán trong suốt lên góc ảnh.
        """
        stats = self.get_stats()
        lines = [
            f"FPS:        {stats.get('fps', 0.0):.1f}",
            f"Capture:    {stats.get('capture', 0.0):.1f} ms",
            f"Inference:  {stats.get('inference', 0.0):.1f} ms",
            f"Tracking:   {stats.get('tracking', 0.0):.1f} ms",
            f"Policy:     {stats.get('policy', 0.0):.1f} ms",
            f"Render:     {stats.get('render', 0.0):.1f} ms",
            f"Pipeline:   {stats.get('total_pipeline_ms', 0.0):.1f} ms"
        ]

        # Kích thước box nền
        box_w, box_h = 220, len(lines) * 22 + 15
        img_h, img_w = frame.shape[:2]
        x2 = min(x + box_w, img_w)
        y2 = min(y + box_h, img_h)

        # Lớp phủ bán trong suốt (semi-transparent overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Viền hộp
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 255), 1)

        # Vẽ từng dòng text
        text_y = y + 20
        for i, line in enumerate(lines):
            # Dòng đầu tiên (FPS) và dòng cuối (Pipeline) dùng màu nổi bật
            color = (0, 255, 0) if i == 0 else ((0, 255, 255) if i == len(lines)-1 else (240, 240, 240))
            cv2.putText(
                frame,
                line,
                (x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )
            text_y += 22

        return frame
