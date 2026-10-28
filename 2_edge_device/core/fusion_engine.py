"""
fusion_engine.py - Động cơ hợp nhất cảm biến Siêu âm HC-SR04 & Camera (Mức B)
=============================================================================
Hiện thực hoá toàn diện theo đúng đặc tính vật lý CẢM BIẾN SIÊU ÂM:
1. Yêu cầu 2: Hợp nhất cảm biến đa phương thức (Multimodal Sensor Fusion)
   - Chùm sóng âm hình nón (Acoustic Beam Cone): HC-SR04 có góc mở ~30 độ (nửa góc 15 độ).
   - Sóng siêu âm phản xạ từ bề mặt gần nhất trong chùm phát -> Gán cự ly đo được của
     cảm biến siêu âm cho vật thể gần nhất nằm trong góc chùm sóng.
   - Gói tin hoàn chỉnh: [Tên vật thể + Tọa độ BBox + Khoảng cách siêu âm thực tế].
2. Yêu cầu 4: Đánh giá nguy cơ & Xếp hạng ưu tiên (Risk Prioritization)
   - Công thức tính điểm nguy cơ va chạm dựa trên 3 yếu tố:
     (1) Mức độ nguy hiểm loại vật cản (S_class: stairs, car, pole > chair, bottle)
     (2) Khoảng cách thực tế (S_dist: gần thì điểm cao)
     (3) Vị trí hướng đi thẳng (S_lane: làn giữa trung tâm > hai bên rìa)
   - Vật nguy hiểm nhất luôn được xếp đầu danh sách để ưu tiên cảnh báo trước.
3. Kế thừa Bộ lọc Kalman [d, ḋ], TTC = d / (-ḋ) và Tự hiệu chỉnh thang đo trực tuyến.
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
    """Gói tin kết quả hoàn chỉnh của vật cản sau khi hợp nhất đa phương thức."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance_m: float
    velocity_mps: float         # d_dot: âm khi đang tiến lại gần
    closing_speed_mps: float    # -d_dot: dương khi tiến lại gần
    ttc_seconds: float          # Time-To-Collision (s)
    lane: str                   # "left", "center", "right"
    risk_score: float           # [0.0 - 1.0]
    is_emergency: bool          # Nguy hiểm cận kề (< 1.0m hoặc TTC < 1.5s)
    source_status: str          # "two_sources_matched", "us_penalized", "camera_only", "unlabeled_obstacle", "camera_scaled"
    estimation_method: str      # "ground_plane", "pinhole_heuristic", "ultrasonic_direct"
    is_ultrasonic_target: bool = False  # True nếu là vật nhận phản xạ sóng siêu âm
    is_laser_target: bool = False       # Alias giữ tương thích


class SensorFusionEngine:
    """
    Bộ não điều phối hợp nhất dữ liệu không gian giữa Cảm biến Siêu âm HC-SR04 và Camera.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

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

        # Bộ tự hiệu chỉnh thang đo trực tuyến
        scale_cfg = config.get("scale_calibration", {})
        self.scale_calibrator = OnlineScaleCalibrator(
            learning_rate=scale_cfg.get("learning_rate", 0.05),
            initial_scale=scale_cfg.get("initial_scale", 1.0),
            min_scale=scale_cfg.get("min_scale", 0.5),
            max_scale=scale_cfg.get("max_scale", 2.0),
            min_stable_frames=scale_cfg.get("min_stable_frames", 3)
        )

        # Cấu hình chùm sóng nón siêu âm HC-SR04 (~30 độ)
        us_cfg = config.get("ultrasonic", {})
        self.beam_angle_deg = us_cfg.get("beam_angle_deg", 30.0)
        self.beam_angle_rad = math.radians(self.beam_angle_deg)
        self.half_beam_angle_rad = self.beam_angle_rad / 2.0
        self.emergency_distance_m = us_cfg.get("emergency_distance_m", 1.0)

        # Cấu hình Kalman Filter
        self.kf_cfg = config.get("kalman_filter", {})
        self.q_accel_noise = self.kf_cfg.get("q_accel_noise", 0.5)
        self.r_us = self.kf_cfg.get("r_ultrasonic", 0.0025)
        self.r_cam = self.kf_cfg.get("r_camera", 0.09)
        self.r_us_penalized = self.kf_cfg.get("r_ultrasonic_penalized", 0.50)
        self.gating_threshold = self.kf_cfg.get("gating_threshold", 4.0)

        # Cấu hình xung đột & xếp hạng rủi ro
        conflict_cfg = config.get("conflict_resolution", {})
        self.match_ratio = conflict_cfg.get("match_discrepancy_ratio", 0.25)
        self.risk_cfg = config.get("risk_scoring", {})
        self.danger_weights = config.get("class_danger_weights", {})

        self.last_process_time = time.time()
        self.unlabeled_trigger_count = 0

    def is_in_ultrasonic_beam(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Kiểm tra xem Bounding Box có nằm trong góc nón chùm sóng siêu âm HC-SR04 hay không.
        Góc lệch trục ngang: theta_x = arctan((cx - c_x) / f_x).
        HC-SR04 gắn chính diện trục giữa của camera.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        delta_x = cx - self.cam_estimator.c_x
        theta_x = abs(math.atan2(delta_x, self.cam_estimator.f_x))
        return theta_x <= self.half_beam_angle_rad

    def classify_lane(self, bbox: Tuple[int, int, int, int]) -> str:
        """Phân làn 3 vùng: Trái - Giữa - Phải."""
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
        Tính điểm nguy cơ va chạm (Risk Score) theo 3 yếu tố cốt lõi (Yêu cầu 4):
        1. Mức độ nguy hiểm của loại vật cản (S_class): cầu thang, xe, cột > bàn, ghế
        2. Khoảng cách thực tế (S_dist): càng gần điểm càng cao
        3. Vị trí hướng đi thẳng (S_lane / S_heading): ở giữa đường đi ưu tiên hơn hai rìa
        """
        w_dist = self.risk_cfg.get("w_distance", 0.40)
        w_class = self.risk_cfg.get("w_class", 0.30)
        w_heading = self.risk_cfg.get("w_heading", 0.20)
        w_ttc = self.risk_cfg.get("w_ttc", 0.10)

        # 1. Yếu tố khoảng cách thực tế (càng gần càng nguy hiểm)
        s_dist = math.exp(-1.1 * max(0.05, distance))

        # 2. Yếu tố loại vật cản
        s_class = self.danger_weights.get(class_name, self.danger_weights.get("default", 0.50))

        # 3. Yếu tố vị trí hướng đi thẳng (tâm góc nhìn là hướng di chuyển)
        cx = (bbox[0] + bbox[2]) / 2.0
        norm_offset = abs(cx - self.cam_estimator.c_x) / (self.cam_estimator.frame_width / 2.0)
        s_heading = max(0.2, 1.0 - norm_offset * 0.7)

        # Thành phần TTC bổ sung nếu đang tiến lại gần nhanh
        s_ttc = math.exp(-max(0.0, ttc) / 2.0) if ttc < 900.0 else 0.0

        total_risk = w_dist * s_dist + w_class * s_class + w_heading * s_heading + w_ttc * s_ttc
        return round(min(1.0, max(0.0, total_risk)), 3)

    def fuse(self,
             tracked_items: List[TrackedItem],
             raw_us_distance: Optional[float],
             dt: Optional[float] = None) -> Tuple[List[FusedObstacleResult], bool]:
        """
        Thực thi hợp nhất cảm biến siêu âm và xếp hạng ưu tiên:
        Returns:
            (fused_obstacles_sorted_by_risk, global_emergency_flag)
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_process_time
        self.last_process_time = current_time

        fused_results: List[FusedObstacleResult] = []
        global_emergency = False

        if raw_us_distance is not None and 0.03 <= raw_us_distance < self.emergency_distance_m:
            global_emergency = True

        curr_scale = self.scale_calibrator.scale
        tracks_in_beam: List[Tuple[TrackedItem, float, str]] = []
        tracks_outside_beam: List[Tuple[TrackedItem, float, str]] = []

        # 1. Cập nhật ước lượng Camera cho từng track
        for track in tracked_items:
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
                track.kalman_dist.predict(dt)

            track.kalman_dist.update_camera(d_cam)
            in_beam = self.is_in_ultrasonic_beam(track.bbox)
            track.is_in_ultrasonic_beam = in_beam
            track.lane = self.classify_lane(track.bbox)

            if in_beam:
                tracks_in_beam.append((track, d_cam, est_method))
            else:
                tracks_outside_beam.append((track, d_cam, est_method))

        # 2. Xử lý vật cản vô hình / không nhãn (Tình huống 3: Kính trong suốt, tường trơn)
        if raw_us_distance is not None and raw_us_distance < 2.0 and len(tracks_in_beam) == 0:
            self.unlabeled_trigger_count += 1
            if self.unlabeled_trigger_count >= 2:
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
                    estimation_method="ultrasonic_direct",
                    is_ultrasonic_target=True,
                    is_laser_target=True
                )
                fused_results.append(unlabeled_obs)
        else:
            self.unlabeled_trigger_count = 0

        # 3. Gắn kết khoảng cách siêu âm cho vật trong chùm sóng nón (Yêu cầu 2)
        # Sóng âm HC-SR04 phản xạ từ bề mặt gần nhất -> gán echo cho vật có d_cam gần nhất trong chùm
        if len(tracks_in_beam) > 0 and raw_us_distance is not None:
            tracks_in_beam.sort(key=lambda item: item[1])

            closest_track, closest_d_cam, closest_method = tracks_in_beam[0]
            ratio_diff = abs(raw_us_distance - closest_d_cam) / max(0.1, min(raw_us_distance, closest_d_cam))

            if raw_us_distance > closest_d_cam * (1.0 + self.match_ratio):
                # Tình huống 2: Cảm biến báo xa hơn camera nhiều (tiêu âm, phản xạ trượt) -> Phạt siêu âm
                closest_track.kalman_dist.update_ultrasonic(raw_us_distance, variance_override=self.r_us_penalized)
                if closest_track.kalman_dist.distance > closest_d_cam:
                    closest_track.kalman_dist.x[0, 0] = closest_d_cam
                source_status = "us_penalized"
            elif ratio_diff <= self.match_ratio:
                # Tình huống 1: Hai nguồn khớp -> Cập nhật siêu âm chuẩn và tự học thang đo scale s
                closest_track.kalman_dist.update_ultrasonic(raw_us_distance)
                self.scale_calibrator.update(raw_us_distance, closest_d_cam, closest_track.age)
                source_status = "two_sources_matched"
            else:
                # Siêu âm gần hơn camera một chút: Cập nhật siêu âm đảm bảo an toàn
                closest_track.kalman_dist.update_ultrasonic(raw_us_distance, enforce_gating=False)
                closest_track.kalman_dist.x[0, 0] = raw_us_distance
                source_status = "two_sources_matched"

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
                estimation_method=closest_method,
                is_ultrasonic_target=True,
                is_laser_target=True
            ))

            # Các vật còn lại trong chùm sóng nón: dùng khoảng cách camera đã được chuẩn hóa bởi scale s
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
                    estimation_method=other_method,
                    is_ultrasonic_target=False,
                    is_laser_target=False
                ))

        # 4. Các vật ngoài chùm siêu âm
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
                estimation_method=out_method,
                is_ultrasonic_target=False,
                is_laser_target=False
            ))

        # YÊU CẦU 4: XẾP HẠNG ƯU TIÊN THEO RISK SCORE GIẢM DẦN
        fused_results.sort(key=lambda x: x.risk_score, reverse=True)

        return fused_results, global_emergency
