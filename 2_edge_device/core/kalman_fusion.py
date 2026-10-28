"""
kalman_fusion.py - Bộ lọc Kalman theo từng Track và Tự hiệu chỉnh thang đo (Mức B)
==================================================================================
Giải quyết vấn đề Fusion data giữa Camera và Cảm biến siêu âm HC-SR04:
1. Mô hình hình học Camera kết hợp Ground-plane & Pinhole heuristic fallback.
2. Bộ lọc Kalman 2 trạng thái [d, ḋ] (Khoảng cách, Tốc độ tiến lại) cho từng Track.
3. Cập nhật bất đồng bộ (Asynchronous Update) với Gating loại trừ số đo dị biệt.
4. Ước tính Time-To-Collision (TTC) trực tiếp: TTC = d / (-ḋ).
5. Cơ chế Online Scale Calibration (s = d_us / d_cam) tự học thang đo từ siêu âm.
"""

import time
import math
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class CameraDistanceEstimator:
    """
    Ước lượng khoảng cách từ Bounding Box trên ảnh:
    - Vật đứng trên sàn: d_cam = f_y * h / (v_bottom - c_y)
    - Vật không chạm sàn hoặc bị cắt đáy: d_cam = W_real * s * f_x / bbox_width
    """

    def __init__(self,
                 f_x: float = 480.0,
                 f_y: float = 480.0,
                 c_x: float = 320.0,
                 c_y: float = 240.0,
                 camera_height_m: float = 1.45,
                 tilt_angle_deg: float = 5.0,
                 frame_height: int = 480,
                 frame_width: int = 640,
                 reference_widths: Optional[Dict[str, float]] = None):
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y
        self.camera_height_m = camera_height_m
        self.tilt_angle_rad = math.radians(tilt_angle_deg)
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Chiều rộng vật lý chuẩn (mét) cho các lớp COCO
        self.reference_widths = reference_widths or {
            "person": 0.45,
            "chair": 0.50,
            "dining table": 1.20,
            "couch": 1.60,
            "bed": 1.80,
            "bicycle": 0.60,
            "motorcycle": 0.80,
            "car": 1.80,
            "bus": 2.50,
            "truck": 2.50,
            "pole": 0.25,
            "stairs": 1.00,
            "fire hydrant": 0.35,
            "potted plant": 0.40,
            "door": 0.85,
            "default": 0.50
        }

        # Danh sách các lớp mặc định thường tiếp xúc trực tiếp với mặt sàn
        self.ground_classes = {
            "person", "chair", "dining table", "couch", "bed",
            "bicycle", "motorcycle", "car", "bus", "truck",
            "pole", "stairs", "fire hydrant", "potted plant", "bench"
        }

    def estimate_distance(self,
                          bbox: Tuple[int, int, int, int],
                          class_name: str,
                          scale_factor: float = 1.0) -> Tuple[float, str]:
        """
        Tính toán khoảng cách từ camera đến vật thể.

        Args:
            bbox: (x1, y1, x2, y2)
            class_name: tên lớp đối tượng
            scale_factor: hệ số thang đo học online từ siêu âm s = d_us / d_cam

        Returns:
            (distance_m, estimation_method)
        """
        x1, y1, x2, y2 = bbox
        bbox_width = max(1, x2 - x1)
        v_bottom = y2

        # Kiểm tra điều kiện áp dụng Ground-Plane Geometry:
        # 1. Đáy bbox phải nằm dưới trục quang học (v_bottom > c_y)
        # 2. Không bị chạm sát mép đáy ảnh (v_bottom < frame_height - 12)
        # 3. Thuộc nhóm vật thể đứng trên mặt sàn
        is_touching_bottom_edge = (v_bottom >= self.frame_height - 12)
        is_above_horizon = (v_bottom <= self.c_y + 5)
        is_ground_object = class_name in self.ground_classes

        if is_ground_object and not is_touching_bottom_edge and not is_above_horizon:
            # 1. Áp dụng mô hình hình học mặt sàn (Ground-plane distance)
            # Có bù góc nghiêng camera alpha:
            # theta_y = arctan((v_bottom - c_y) / f_y)
            # d_cam = h / tan(alpha + theta_y)
            delta_v = v_bottom - self.c_y
            theta_y = math.atan2(delta_v, self.f_y)
            total_angle = self.tilt_angle_rad + theta_y

            if total_angle > 0.05:  # Tránh chia cho 0 hoặc góc âm
                d_ground = self.camera_height_m / math.tan(total_angle)
                if 0.2 <= d_ground <= 12.0:
                    return round(d_ground, 3), "ground_plane"

        # 2. Heuristic fallback: Pinhole Model theo kích thước chiều rộng lớp
        # d_pinhole = (W_real * scale_factor * f_x) / bbox_width
        w_real = self.reference_widths.get(class_name, self.reference_widths.get("default", 0.50))
        effective_w = w_real * scale_factor
        d_pinhole = (effective_w * self.f_x) / bbox_width
        d_pinhole = max(0.15, min(d_pinhole, 10.0))

        return round(d_pinhole, 3), "pinhole_heuristic"


class TrackDistanceKalmanFilter:
    """
    Bộ lọc Kalman 1D bậc 2 cho từng Track đối tượng.
    Vector trạng thái x = [d, d_dot]^T
    - d: khoảng cách (mét)
    - d_dot: tốc độ thay đổi khoảng cách (m/s). d_dot < 0 nghĩa là đang tiến lại gần!
    """

    def __init__(self,
                 initial_distance: float,
                 q_accel_noise: float = 0.5,
                 r_ultrasonic: float = 0.0025,
                 r_camera: float = 0.09,
                 gating_threshold: float = 4.0):
        # Trạng thái ban đầu [d, d_dot]
        self.x = np.array([[initial_distance], [0.0]], dtype=np.float64)

        # Ma trận hiệp phương sai sai số trạng thái P
        self.P = np.array([
            [0.2, 0.0],
            [0.0, 1.0]
        ], dtype=np.float64)

        self.q_accel_noise = q_accel_noise
        self.r_us_default = r_ultrasonic
        self.r_cam_default = r_camera
        self.gating_threshold = gating_threshold

        # Ma trận đo lường H: chỉ đo trực tiếp khoảng cách d (không đo trực tiếp vận tốc)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        self.last_update_time = time.time()
        self.hits = 1
        self.updates_count = 0
        self.us_updates_count = 0

    def predict(self, dt: float):
        """
        Bước dự đoán (Prediction Step) với khoảng thời gian dt (giây):
        x = F * x
        P = F * P * F^T + Q
        """
        if dt <= 0.0:
            dt = 0.033  # Mặc định ~ 30 FPS

        # Ma trận chuyển trạng thái
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)

        # Ma trận nhiễu quá trình (mô hình gia tốc không đổi ngẫu nhiên)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = self.q_accel_noise
        Q = np.array([
            [0.25 * dt4 * q, 0.5 * dt3 * q],
            [0.5 * dt3 * q,  dt2 * q]
        ], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Giới hạn vật lý: khoảng cách không thể âm
        if self.x[0, 0] < 0.05:
            self.x[0, 0] = 0.05
            self.x[1, 0] = min(0.0, self.x[1, 0])

    def _update_measurement(self, z: float, R_scalar: float, enforce_gating: bool = True) -> bool:
        """
        Cập nhật trạng thái khi nhận một số đo z với phương sai R.
        Trả về True nếu cập nhật thành công, False nếu bị Gating loại trừ.
        """
        z_arr = np.array([[z]], dtype=np.float64)
        R = np.array([[R_scalar]], dtype=np.float64)

        # Đổi mới (Innovation / Residual): y = z - H * x
        y = z_arr - (self.H @ self.x)

        # Hiệp phương sai đổi mới: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R
        S_scalar = float(S[0, 0])

        # Ngưỡng Gating (Kiểm tra sai số chuẩn hoá Mahalanobis)
        if enforce_gating and S_scalar > 1e-9:
            mahalanobis_dist = abs(float(y[0, 0])) / math.sqrt(S_scalar)
            if mahalanobis_dist > self.gating_threshold:
                logger.debug(f"[GATING] Loại bỏ số đo ngoại lai z={z:.2f}m (M={mahalanobis_dist:.2f} > {self.gating_threshold})")
                return False

        # Hệ số khuếch đại Kalman: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Cập nhật trạng thái và hiệp phương sai
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float64)
        # Dạng Joseph đảm bảo tính đối xứng và xác định dương của P
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T

        # Giữ khoảng cách dương
        if self.x[0, 0] < 0.05:
            self.x[0, 0] = 0.05

        self.updates_count += 1
        return True

    def update_camera(self, z_cam: float, variance_override: Optional[float] = None) -> bool:
        """Cập nhật số đo từ Camera (phương sai lớn hơn)."""
        r = variance_override if variance_override is not None else self.r_cam_default
        return self._update_measurement(z_cam, r, enforce_gating=True)

    def update_ultrasonic(self, z_us: float, variance_override: Optional[float] = None, enforce_gating: bool = True) -> bool:
        """Cập nhật số đo từ Siêu âm HC-SR04 / Laser (phương sai nhỏ, độ tin cậy cao)."""
        r = variance_override if variance_override is not None else self.r_us_default
        success = self._update_measurement(z_us, r, enforce_gating=enforce_gating)
        if success:
            self.us_updates_count += 1
        return success

    @property
    def distance(self) -> float:
        """Khoảng cách lọc hiện tại (m)."""
        return float(self.x[0, 0])

    @property
    def velocity(self) -> float:
        """
        Vận tốc thay đổi khoảng cách d_dot (m/s).
        d_dot < 0: đối tượng đang tiến gần lại
        d_dot > 0: đối tượng đang di chuyển ra xa
        """
        return float(self.x[1, 0])

    @property
    def closing_speed(self) -> float:
        """Tốc độ tiến lại gần (-d_dot, m/s). Dương khi tiến lại."""
        return -float(self.x[1, 0])

    def get_ttc(self) -> float:
        """
        Tính Time-To-Collision (TTC):
        TTC = d / (-d_dot) = d / closing_speed
        - Nếu closing_speed <= 0.05 m/s (đứng yên hoặc đi xa): TTC = vô cùng (999.0s).
        """
        d = self.distance
        v_close = self.closing_speed

        if v_close > 0.05 and d > 0.01:
            ttc = d / v_close
            return max(0.01, min(ttc, 999.0))
        return 999.0


class OnlineScaleCalibrator:
    """
    Tự hiệu chỉnh thang đo trực tuyến (Online Scale Calibration):
    - Khi siêu âm và camera cùng nhìn thấy một track ổn định,
      cập nhật hệ số s = d_us / d_cam bằng trung bình trượt (EMA).
    - Camera tự "học" thang đo từ siêu âm để ước lượng chính xác các vật nằm ngoài chùm.
    """

    def __init__(self,
                 learning_rate: float = 0.05,
                 initial_scale: float = 1.0,
                 min_scale: float = 0.5,
                 max_scale: float = 2.0,
                 min_stable_frames: int = 3):
        self.learning_rate = learning_rate
        self.scale = initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_stable_frames = min_stable_frames
        self.total_calibrations = 0

    def update(self, d_us: float, d_cam: float, track_age: int) -> float:
        """
        Cập nhật hệ số s khi hội tụ đủ điều kiện ổn định.
        """
        if track_age < self.min_stable_frames or d_cam <= 0.1 or d_us <= 0.1:
            return self.scale

        # Sai lệch không được quá lớn (> 50%) để tránh học sai từ nhiễu
        ratio = d_us / d_cam
        if 0.4 <= ratio <= 2.5:
            # Trung bình trượt hàm mũ: s_new = (1 - alpha) * s_old + alpha * ratio
            self.scale = (1.0 - self.learning_rate) * self.scale + self.learning_rate * ratio
            self.scale = max(self.min_scale, min(self.scale, self.max_scale))
            self.total_calibrations += 1

        return self.scale
