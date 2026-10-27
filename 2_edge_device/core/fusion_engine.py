"""
fusion_engine.py - Động cơ hợp nhất cảm biến Siêu âm HC-SR04 & Camera YOLO (Mức B)
==================================================================================
Hiện thực hoá toàn bộ logic Fusion:
1. Xác định vật cản nằm trong chùm siêu âm (Beam Cone).
2. Xử lý triệt để 4 tình huống trong Bảng Mâu Thuẫn (Conflict Resolution Matrix).
3. Quy tắc an toàn khẩn cấp (Fail-Safe Emergency Rule).
4. Tính toán chỉ số nguy cơ va chạm (Risk Score) tích hợp TTC = d / (-ḋ).
5. Phân làn điều hướng 3 làn (Trái - Giữa - Phải).
"""

import time
import math
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .kalman_fusion import CameraDistanceEstimator, TrackDistanceKalmanFilter, OnlineScaleCalibrator
from modules.tracking.byte_tracker import TrackedItem

logger = logging.getLogger(__name__)


@dataclass
class FusedObstacleResult:
    """Kết quả hoàn chỉnh của một vật cản sau khi hợp nhất."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance_m: float
    velocity_mps: float         # d_dot: âm khi đang tiến lại gần
    closing_speed_mps: float    # -d_dot: dương khi tiến lại gần
    ttc_seconds: float          # Time-To-Collision
    lane: str                   # "left", "center", "right"
    risk_score: float           # [0.0 - 1.0]
    is_emergency: bool          # Nguy hiểm cận kề (< 0.8m hoặc TTC < 1.5s)
    source_status: str          # "two_sources_matched", "us_penalized", "camera_only", "unlabeled_obstacle"
    estimation_method: str      # "ground_plane", "pinhole_heuristic", "ultrasonic_direct"


class SensorFusionEngine:
    """
    Bộ não trung tâm điều phối dung hợp cảm biến siêu âm HC-SR04 và Camera.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 1. Khởi tạo bộ ước lượng khoảng cách camera
        cam_cfg = config.get("camera", {})
        ref_widths = config.get("reference_widths", {})
        self.cam_estimator = CameraDistanceEstimator(
            f_x=cam_cfg.get("f_x", 480.0),
            f_y=cam_cfg.get("f_y", 480.0),
            c_x=cam_cfg.get("c_x", 320.0),
            c_y=cam_cfg.get("c_y", 240.0),
            camera_height_m=cam_cfg.get("camera_height_m", 1.45),
            tilt_angle_deg=cam_cfg.get("tilt_angle_deg", 5.0),
            frame_height=cam_cfg.get("frame_height", 480),
            frame_width=cam_cfg.get("frame_width", 640),
            reference_widths=ref_widths
        )

        # 2. Bộ tự hiệu chỉnh thang đo trực tuyến (Online Scale Calibrator)
        scale_cfg = config.get("scale_calibration", {})
        self.scale_calibrator = OnlineScaleCalibrator(
            learning_rate=scale_cfg.get("learning_rate", 0.05),
            initial_scale=scale_cfg.get("initial_scale", 1.0),
            min_scale=scale_cfg.get("min_scale", 0.5),
            max_scale=scale_cfg.get("max_scale", 2.0),
            min_stable_frames=scale_cfg.get("min_stable_frames", 3)
        )

        # 3. Cấu hình chùm sóng siêu âm
        us_cfg = config.get("ultrasonic", {})
        self.beam_angle_rad = math.radians(us_cfg.get("beam_angle_deg", 30.0))
        self.half_beam_angle_rad = self.beam_angle_rad / 2.0
        self.emergency_distance_m = us_cfg.get("emergency_distance_m", 0.8)

        # 4. Cấu hình Kalman Filter
        self.kf_cfg = config.get("kalman_filter", {})
        self.q_accel_noise = self.kf_cfg.get("q_accel_noise", 0.5)
        self.r_us = self.kf_cfg.get("r_ultrasonic", 0.0025)
        self.r_cam = self.kf_cfg.get("r_camera", 0.09)
        self.r_us_penalized = self.kf_cfg.get("r_ultrasonic_penalized", 0.50)
        self.gating_threshold = self.kf_cfg.get("gating_threshold", 4.0)

        # 5. Cấu hình xung đột & rủi ro
        conflict_cfg = config.get("conflict_resolution", {})
        self.match_ratio = conflict_cfg.get("match_discrepancy_ratio", 0.25)

        self.risk_cfg = config.get("risk_scoring", {})
        self.danger_weights = config.get("class_danger_weights", {})

        self.last_process_time = time.time()
        self.unlabeled_trigger_count = 0

    def is_in_ultrasonic_beam(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Kiểm tra xem Bounding Box có giao cắt với góc nón phát sóng siêu âm hay không.
        HC-SR04 được gắn thẳng trục chính diện của camera.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0

        # Tính góc lệch phương vị trục ngang: theta_x = arctan((cx - c_x) / f_x)
        delta_x = cx - self.cam_estimator.c_x
        theta_x = abs(math.atan2(delta_x, self.cam_estimator.f_x))

        return theta_x <= self.half_beam_angle_rad

    def classify_lane(self, bbox: Tuple[int, int, int, int]) -> str:
        """Phân làn điều hướng 3 làn: Trái - Giữa - Phải."""
        cx = (bbox[0] + bbox[2]) / 2.0
        width = self.cam_estimator.frame_width
        third = width / 3.0

        if cx < third:
            return "left"
        elif cx < 2.0 * third:
            return "center"
        else:
            return "right"

    def compute_risk_score(self,
                           distance: float,
                           ttc: float,
                           bbox: Tuple[int, int, int, int],
                           class_name: str) -> float:
        """
        Tính điểm nguy cơ va chạm tổng hợp (Risk Score) theo yêu cầu Vấn đề 4:
        Risk = w_ttc * f(TTC) + w_d * f(d) + w_theta * f(theta) + w_class * f(class)
        """
        w_ttc = self.risk_cfg.get("w_ttc", 0.40)
        w_d = self.risk_cfg.get("w_distance", 0.30)
        w_theta = self.risk_cfg.get("w_heading", 0.15)
        w_class = self.risk_cfg.get("w_class", 0.15)

        # 1. Thành phần TTC (Cực kỳ nguy hiểm nếu TTC < 2 giây)
        if ttc < 900.0:
            # f_ttc tăng nhanh khi TTC nhỏ
            f_ttc = math.exp(-max(0.0, ttc) / 1.8)
        else:
            f_ttc = 0.0

        # 2. Thành phần cự ly (Proximity Decay)
        f_d = math.exp(-1.5 * max(0.05, distance))

        # 3. Thành phần góc lệch tâm (Heading Alignment): trung tâm nguy hiểm hơn hai rìa
        cx = (bbox[0] + bbox[2]) / 2.0
        norm_x = abs(cx - self.cam_estimator.c_x) / (self.cam_estimator.frame_width / 2.0)
        f_theta = max(0.0, 1.0 - norm_x * 0.7)

        # 4. Mức độ nguy hại theo lớp
        f_class = self.danger_weights.get(class_name, self.danger_weights.get("default", 0.50))

        risk = w_ttc * f_ttc + w_d * f_d + w_theta * f_theta + w_class * f_class
        return round(min(1.0, max(0.0, risk)), 3)

    def fuse(self,
             tracked_items: List[TrackedItem],
             raw_us_distance: Optional[float],
             dt: Optional[float] = None) -> Tuple[List[FusedObstacleResult], bool]:
        """
        Hàm thực thi chính của Sensor Fusion:
        Args:
            tracked_items: Danh sách các track từ Tracker
            raw_us_distance: Số đo khoảng cách hiện tại từ siêu âm (hoặc None nếu mất tín hiệu)
            dt: Khoảng thời gian từ lần xử lý trước (giây)

        Returns:
            (fused_obstacles_sorted_by_risk, global_emergency_flag)
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_process_time
        self.last_process_time = current_time

        fused_results: List[FusedObstacleResult] = []
        global_emergency = False

        # QUY TẮC AN TOÀN TUYỆT ĐỐI (FAIL-SAFE):
        # Bất kỳ khi nào cảm biến siêu âm báo gần hơn ngưỡng khẩn cấp (< 0.8m),
        # lập tức kích hoạt cờ khẩn cấp, không chờ AI gắn nhãn!
        if raw_us_distance is not None and 0.03 <= raw_us_distance < self.emergency_distance_m:
            global_emergency = True

        # =====================================================================
        # 1. BƯỚC 1: DỰ ĐOÁN VÀ ƯỚC LƯỢNG KHOẢNG CÁCH CAMERA CHO TỪNG TRACK
        # =====================================================================
        tracks_in_beam: List[Tuple[TrackedItem, float, str]] = []
        tracks_outside_beam: List[Tuple[TrackedItem, float, str]] = []

        curr_scale = self.scale_calibrator.scale

        for track in tracked_items:
            # Khởi tạo bộ lọc Kalman nếu là track mới
            d_cam, est_method = self.cam_estimator.estimate_distance(
                track.bbox, track.class_name, scale_factor=curr_scale
            )

            if track.kalman_dist is None:
                track.kalman_dist = TrackDistanceKalmanFilter(
                    initial_distance=d_cam,
                    q_accel_noise=self.q_accel_noise,
                    r_ultrasonic=self.r_us,
                    r_camera=self.r_cam,
                    gating_threshold=self.gating_threshold
                )
            else:
                # Bước dự đoán Kalman
                track.kalman_dist.predict(dt)

            # Cập nhật số đo Camera cho mọi track ở mọi frame
            track.kalman_dist.update_camera(d_cam)

            # Kiểm tra xem vật thể có nằm trong chùm phát siêu âm không
            in_beam = self.is_in_ultrasonic_beam(track.bbox)
            track.is_in_ultrasonic_beam = in_beam
            track.lane = self.classify_lane(track.bbox)

            if in_beam:
                tracks_in_beam.append((track, d_cam, est_method))
            else:
                tracks_outside_beam.append((track, d_cam, est_method))

        # =====================================================================
        # 2. BƯỚC 2: XỬ LÝ THEO BẢNG MÂU THUẪN (CONFLICT RESOLUTION MATRIX)
        # =====================================================================

        # Tình huống 3: Siêu âm báo gần, nhưng Camera không thấy bất kỳ vật nào trong chùm
        # (Ví dụ: kính trong suốt, vách tường trắng nhẵn, hoặc vật cản lạ không có nhãn COCO)
        if raw_us_distance is not None and raw_us_distance < 2.0 and len(tracks_in_beam) == 0:
            self.unlabeled_trigger_count += 1
            if self.unlabeled_trigger_count >= 2:
                # Tạo một vật cản không nhãn ở làn giữa để bảo vệ an toàn ngay
                is_emer = raw_us_distance < self.emergency_distance_m
                if is_emer:
                    global_emergency = True

                unlabeled_obs = FusedObstacleResult(
                    track_id=-1,
                    class_name="vật cản",
                    confidence=1.0,
                    bbox=(200, 150, 440, 450),
                    distance_m=round(raw_us_distance, 2),
                    velocity_mps=-0.5,
                    closing_speed_mps=0.5,
                    ttc_seconds=max(0.1, round(raw_us_distance / 0.5, 1)),
                    lane="center",
                    risk_score=0.95 if is_emer else 0.75,
                    is_emergency=is_emer,
                    source_status="unlabeled_obstacle",
                    estimation_method="ultrasonic_direct"
                )
                fused_results.append(unlabeled_obs)
        else:
            self.unlabeled_trigger_count = 0

        # Tình huống 4: Nhiều vật trong một chùm siêu âm
        # Chùm siêu âm phản xạ từ bề mặt gần nhất -> Gán echo cho bbox có d_cam gần nhất
        if len(tracks_in_beam) > 0 and raw_us_distance is not None:
            # Sắp xếp các vật trong chùm theo d_cam tăng dần
            tracks_in_beam.sort(key=lambda item: item[1])

            # Vật gần nhất nhận echo siêu âm
            closest_track, closest_d_cam, closest_method = tracks_in_beam[0]

            # Kiểm tra Tình huống 1 & Tình huống 2 đối với vật gần nhất này:
            ratio_diff = abs(raw_us_distance - closest_d_cam) / max(0.1, min(raw_us_distance, closest_d_cam))

            if ratio_diff <= self.match_ratio:
                # --- TÌNH HUỐNG 1: Hai nguồn khớp (lệch dưới 25%) ---
                # Cập nhật Kalman bình thường với phương sai chuẩn
                closest_track.kalman_dist.update_ultrasonic(raw_us_distance)
                # Tự hiệu chỉnh thang đo trực tuyến (Online Scale Calibration)
                self.scale_calibrator.update(raw_us_distance, closest_d_cam, closest_track.age)
                source_status = "two_sources_matched"
            elif raw_us_distance > closest_d_cam * (1.0 + self.match_ratio):
                # --- TÌNH HUỐNG 2: Siêu âm xa hơn camera nhiều (vật mềm tiêu âm, bề mặt nghiêng) ---
                # Phạt phương sai siêu âm, tin camera hơn, cảnh báo bảo thủ (lấy min)
                closest_track.kalman_dist.update_ultrasonic(raw_us_distance, variance_override=self.r_us_penalized)
                # Đảm bảo khoảng cách bảo thủ
                if closest_track.kalman_dist.distance > closest_d_cam:
                    closest_track.kalman_dist.x[0, 0] = closest_d_cam
                source_status = "us_penalized"
            else:
                # Siêu âm gần hơn camera một chút: Cập nhật siêu âm để đảm bảo an toàn
                closest_track.kalman_dist.update_ultrasonic(raw_us_distance)
                source_status = "two_sources_matched"

            # Đóng gói kết quả cho vật gần nhất
            d_final = closest_track.kalman_dist.distance
            ttc = closest_track.kalman_dist.get_ttc()
            v_dot = closest_track.kalman_dist.velocity
            risk = self.compute_risk_score(d_final, ttc, closest_track.bbox, closest_track.class_name)
            is_emer = (d_final < self.emergency_distance_m) or (ttc < 1.5 and d_final < 1.5)
            if is_emer:
                global_emergency = True

            fused_results.append(FusedObstacleResult(
                track_id=closest_track.track_id,
                class_name=closest_track.class_name,
                confidence=closest_track.confidence,
                bbox=closest_track.bbox,
                distance_m=round(d_final, 2),
                velocity_mps=round(v_dot, 2),
                closing_speed_mps=round(-v_dot, 2),
                ttc_seconds=round(ttc, 1),
                lane=closest_track.lane,
                risk_score=risk,
                is_emergency=is_emer,
                source_status=source_status,
                estimation_method=closest_method
            ))

            # Các vật còn lại trong chùm: Dùng ước lượng camera đã được scale bởi s
            for other_track, other_d_cam, other_method in tracks_in_beam[1:]:
                d_final = other_track.kalman_dist.distance
                ttc = other_track.kalman_dist.get_ttc()
                v_dot = other_track.kalman_dist.velocity
                risk = self.compute_risk_score(d_final, ttc, other_track.bbox, other_track.class_name)
                is_emer = (d_final < self.emergency_distance_m)
                if is_emer:
                    global_emergency = True

                fused_results.append(FusedObstacleResult(
                    track_id=other_track.track_id,
                    class_name=other_track.class_name,
                    confidence=other_track.confidence,
                    bbox=other_track.bbox,
                    distance_m=round(d_final, 2),
                    velocity_mps=round(v_dot, 2),
                    closing_speed_mps=round(-v_dot, 2),
                    ttc_seconds=round(ttc, 1),
                    lane=other_track.lane,
                    risk_score=risk,
                    is_emergency=is_emer,
                    source_status="camera_scaled",
                    estimation_method=other_method
                ))

        # Các vật nằm ngoài chùm siêu âm: dùng ước lượng camera (đã hưởng lợi từ thang đo s tự học)
        for out_track, out_d_cam, out_method in tracks_outside_beam:
            d_final = out_track.kalman_dist.distance
            ttc = out_track.kalman_dist.get_ttc()
            v_dot = out_track.kalman_dist.velocity
            risk = self.compute_risk_score(d_final, ttc, out_track.bbox, out_track.class_name)
            is_emer = (d_final < self.emergency_distance_m)
            if is_emer:
                global_emergency = True

            fused_results.append(FusedObstacleResult(
                track_id=out_track.track_id,
                class_name=out_track.class_name,
                confidence=out_track.confidence,
                bbox=out_track.bbox,
                distance_m=round(d_final, 2),
                velocity_mps=round(v_dot, 2),
                closing_speed_mps=round(-v_dot, 2),
                ttc_seconds=round(ttc, 1),
                lane=out_track.lane,
                risk_score=risk,
                is_emergency=is_emer,
                source_status="camera_scaled",
                estimation_method=out_method
            ))

        # Sắp xếp danh sách kết quả theo điểm rủi ro (Risk Score) giảm dần
        fused_results.sort(key=lambda x: x.risk_score, reverse=True)

        return fused_results, global_emergency
