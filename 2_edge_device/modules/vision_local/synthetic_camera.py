"""
synthetic_camera.py - Luồng Camera và Vật thể mô phỏng (Synthetic Camera Stream)
================================================================================
Dành cho kiểm thử cục bộ (Local Testing) trên máy tính/laptop:
- Mô phỏng góc nhìn thứ nhất (FPV) của người khiếm thị.
- Chùm phát siêu âm HC-SR04 hướng dọc theo trục chính diện camera (~30 độ).
- Vật thể chuyển động: người đi bộ ở làn giữa di chuyển từ xa lại gần,
  giúp kiểm chứng đầy đủ 3 phân tầng phản hồi (An toàn -> Dẫn hướng 1-2 từ -> Bíp khẩn cấp).
"""

import time
import math
import numpy as np


class SyntheticCamera:
    """Sinh luồng video mô phỏng người dùng di chuyển trong không gian 3D."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.w = width
        self.h = height
        self.fps = fps
        self.start_time = time.time()
        self.f_x = 480.0
        self.f_y = 480.0
        self.c_x = width / 2.0
        self.c_y = height / 2.0
        self.h_cam = 1.45

    def read(self):
        """
        Trả về: (ret: bool, frame: np.ndarray, detections: list, closest_distance: float)
        """
        t = time.time() - self.start_time
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Vẽ nền không gian phối cảnh (Tường trên, Sàn dưới)
        frame[0:240, :] = [160, 150, 140]
        frame[240:, :] = [60, 60, 60]

        vp_x, vp_y = int(self.c_x), int(self.c_y)
        for bottom_x in [0, int(self.w * 0.33), int(self.w * 0.67), self.w]:
            self._draw_line(frame, vp_x, vp_y, bottom_x, self.h, (100, 100, 100), 1)

        offset = int((t * 50) % 35)
        for y in range(vp_y + 10 + offset, self.h, 35):
            self._draw_line(frame, 0, y, self.w, y, (80, 80, 80), 1)

        detections = []

        # 2. Vật thể 1: "person" ở làn giữa tiến lại gần
        # Chu kỳ 10s: từ 4.0m -> 0.4m rồi lặp lại
        cycle = 10.0
        phase = (t % cycle) / cycle
        d_person = 4.0 - phase * 3.6  # 4.0m -> 0.4m

        v_bottom = int(self.c_y + (self.f_y * self.h_cam) / d_person)
        person_h_px = int((self.f_y * 1.7) / d_person)
        person_w_px = int((self.f_x * 0.5) / d_person)
        v_top = v_bottom - person_h_px
        center_x = int(self.c_x + 10 * math.sin(t * 1.2))

        x1 = max(0, center_x - person_w_px // 2)
        x2 = min(self.w - 1, center_x + person_w_px // 2)
        y1 = max(0, v_top)
        y2 = min(self.h - 1, v_bottom)

        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = (170, 90, 60)
            head_h = max(4, person_h_px // 6)
            head_y1 = max(0, y1 - head_h)
            frame[head_y1:y1, max(0, center_x - head_h // 2):min(self.w, center_x + head_h // 2)] = (190, 200, 220)

            detections.append({
                'class_name': 'person',
                'bbox': [x1, y1, x2, y2],
                'confidence': 0.94
            })

        # 3. Vật thể 2: "chair" ở làn bên phải cố định tại 2.3m
        d_chair = 2.3
        v_bottom_c = int(self.c_y + (self.f_y * self.h_cam) / d_chair)
        chair_h_px = int((self.f_y * 0.8) / d_chair)
        chair_w_px = int((self.f_x * 0.55) / d_chair)
        cx_chair = int(self.w * 0.82)
        cx1 = cx_chair - chair_w_px // 2
        cx2 = cx_chair + chair_w_px // 2
        cy1 = max(0, v_bottom_c - chair_h_px)
        cy2 = min(self.h - 1, v_bottom_c)

        if cx2 > cx1 and cy2 > cy1:
            frame[cy1:cy2, cx1:cx2] = (45, 120, 190)
            detections.append({
                'class_name': 'chair',
                'bbox': [cx1, cy1, cx2, cy2],
                'confidence': 0.88
            })

        return True, frame, detections, d_person

    def _draw_line(self, img, x1, y1, x2, y2, color, thickness=1):
        try:
            import cv2
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        except Exception:
            pass
