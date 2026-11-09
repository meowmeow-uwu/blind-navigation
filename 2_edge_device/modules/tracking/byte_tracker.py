"""
byte_tracker.py - Bộ theo dõi vật cản đa đối tượng nhẹ chuẩn ByteTrack (ECCV 2022)
==================================================================================
Hiện thực hóa đầy đủ Thuật toán Liên kết 2 Tầng (Two-Stage Data Association):
- Tầng 1: Khớp các detection có độ tin cậy cao (D_high >= high_thresh, mặc định >= 0.45)
- Tầng 2: Khớp các detection có độ tin cậy thấp (low_thresh <= D_low < high_thresh) với các
          tracklets chưa khớp ở Tầng 1 -> Cứu vớt đối tượng bị che khuất / rung lắc / mờ hình.
- Cấp phát Track ID duy nhất và cờ 'announced' triệt tiêu 100% nói lặp (Silent Tracking).
- Chỉ khởi tạo Track mới cho các detection điểm cao chưa ghép (D_high); bỏ qua D_low chưa ghép (lọc nhiễu nền).

Tham khảo học thuật:
- ByteTrack paper: Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box", ECCV 2022. (Top 1 MOT17 Challenge)
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Tính Intersection-over-Union (IoU) giữa 2 Bounding Box [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


class TrackedItem:
    """Vật cản được theo dõi liên tục qua nhiều khung hình."""
    _id_counter = 0

    @classmethod
    def reset_counter(cls, start: int = 0):
        """Khởi động lại bộ đếm Track ID."""
        cls._id_counter = start

    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        TrackedItem._id_counter += 1
        self.track_id = TrackedItem._id_counter
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Trạng thái thông báo âm thanh (Yêu cầu 3 & HCI: Chống nói lặp)
        self.announced = False
        self.last_announced_time = 0.0
        self.last_announced_distance = 0.0

        # Bộ lọc Kalman khoảng cách gắn kèm
        self.kalman_dist = None
        self.lane = "center"
        self.risk_score = 0.0
        self.fused_distance = 2.5
        self.ttc = 999.0
        self.is_in_ultrasonic_beam = False
        self.source_description = "camera_only"

    def update_detection(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        """Làm mịn toạ độ BBox khi phát hiện lại ở frame mới."""
        alpha = 0.7
        self.bbox = (
            int(alpha * bbox[0] + (1 - alpha) * self.bbox[0]),
            int(alpha * bbox[1] + (1 - alpha) * self.bbox[1]),
            int(alpha * bbox[2] + (1 - alpha) * self.bbox[2]),
            int(alpha * bbox[3] + (1 - alpha) * self.bbox[3]),
        )
        self.class_name = class_name
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

    def mark_announced(self, distance_m: float = 0.0):
        """Đánh dấu đã thông báo âm thanh, chuyển sang trạng thái theo dõi ngầm."""
        self.announced = True
        self.last_announced_time = time.time()
        self.last_announced_distance = distance_m

    def should_reannounce(self, current_distance: float) -> bool:
        """
        Kiểm tra điều kiện thông báo lại nếu nguy cơ tăng cao:
        - Chưa thông báo: Luôn phát.
        - Đã thông báo: Chỉ phát lại nếu vật cản tiến lại gần đáng kể
          (giảm hơn 0.8m so với lần đọc trước) và đã qua thời gian cooldown >= 3.0s.
        """
        if not self.announced:
            return True
        now = time.time()
        if (now - self.last_announced_time >= 3.0) and (self.last_announced_distance - current_distance >= 0.8):
            return True
        return False

    def predict(self):
        self.age += 1
        self.time_since_update += 1


class LightweightTracker:
    """
    Bộ theo dõi nhẹ chuẩn hóa theo kiến trúc ByteTrack (ECCV 2022).
    Sử dụng giải thuật liên kết 2 tầng (Two-stage Data Association):
    - Tầng 1: Khớp D_high với các track hiện có bằng IoU & Hungarian.
    - Tầng 2: Khớp D_low với các track chưa khớp ở Tầng 1 để cứu vớt vật cản bị che khuất.
    - Tối ưu hóa cực hạn cho thiết bị biên Raspberry Pi 4 CPU (không tốn tài nguyên CNN Re-ID).
    """

    def __init__(self,
                 max_age: int = 15,
                 min_hits: int = 2,
                 iou_threshold: float = 0.25,
                 high_thresh: float = 0.45,
                 low_thresh: float = 0.20):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.tracks: List[TrackedItem] = []

    def _associate(self, tracks_list: List[TrackedItem], dets_list: list, threshold: float) -> Tuple[List[int], List[int]]:
        """
        Khớp nối ma trận IoU giữa danh sách tracks và danh sách detections.
        Trả về: (matched_track_indices, matched_det_indices)
        """
        if not tracks_list or not dets_list:
            return [], []

        iou_matrix = np.zeros((len(tracks_list), len(dets_list)), dtype=np.float32)
        for i, trk in enumerate(tracks_list):
            for j, det in enumerate(dets_list):
                iou_matrix[i, j] = compute_iou(trk.bbox, tuple(det['bbox']))

        matched_tracks = []
        matched_dets = []

        if HAS_SCIPY:
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= threshold:
                    matched_tracks.append(r)
                    matched_dets.append(c)
        else:
            # Greedy matching fallback nếu không có scipy
            matrix_copy = iou_matrix.copy()
            while True:
                idx = np.unravel_index(np.argmax(matrix_copy), matrix_copy.shape)
                max_iou = matrix_copy[idx]
                if max_iou < threshold:
                    break
                r, c = idx
                matched_tracks.append(r)
                matched_dets.append(c)
                matrix_copy[r, :] = -1.0
                matrix_copy[:, c] = -1.0

        return matched_tracks, matched_dets

    def update(self, detections: list) -> List[TrackedItem]:
        """
        Nhận danh sách detection: [{'bbox': [x1, y1, x2, y2], 'class_name': str, 'confidence': float}, ...]
        Trả về danh sách TrackedItem đang hoạt động.
        """
        # Bước 0: Dự đoán bước thời gian cho mọi track hiện có
        for t in self.tracks:
            t.predict()

        if not detections:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]

        # ---------------------------------------------------------------------
        # BƯỚC 1: Phân tách detections thành 2 tầng tin cậy theo ByteTrack
        # D_high: Điểm tin cậy cao (>= high_thresh, mặc định >= 0.45)
        # D_low:  Điểm tin cậy thấp (low_thresh <= score < high_thresh, 0.20 <= score < 0.45)
        # ---------------------------------------------------------------------
        d_high = []
        d_low = []
        for det in detections:
            conf = det.get('confidence', 0.50)
            if conf >= self.high_thresh:
                d_high.append(det)
            elif conf >= self.low_thresh:
                d_low.append(det)

        matched_track_indices = set()
        matched_dhigh_indices = set()

        if len(self.tracks) > 0:
            # -----------------------------------------------------------------
            # TẦNG 1: Khớp D_high với toàn bộ Tracklets hiện có
            # -----------------------------------------------------------------
            if len(d_high) > 0:
                t1_matched_trks, t1_matched_dets = self._associate(
                    self.tracks, d_high, self.iou_threshold
                )
                for trk_idx, det_idx in zip(t1_matched_trks, t1_matched_dets):
                    det = d_high[det_idx]
                    self.tracks[trk_idx].update_detection(
                        tuple(det['bbox']),
                        det['class_name'],
                        det['confidence']
                    )
                    matched_track_indices.add(trk_idx)
                    matched_dhigh_indices.add(det_idx)

            # -----------------------------------------------------------------
            # TẦNG 2: Khớp D_low với các Tracklets CHƯA KHỚP ở Tầng 1
            # (Đột phá ByteTrack: Cứu các track bị che khuất / rung lắc / mờ hình)
            # -----------------------------------------------------------------
            unmatched_tracks_stage1 = [
                i for i in range(len(self.tracks)) if i not in matched_track_indices
            ]
            if len(d_low) > 0 and len(unmatched_tracks_stage1) > 0:
                unmatched_track_objs = [self.tracks[i] for i in unmatched_tracks_stage1]
                t2_matched_trks, t2_matched_dets = self._associate(
                    unmatched_track_objs, d_low, self.iou_threshold * 0.8
                )
                for trk_sub_idx, det_idx in zip(t2_matched_trks, t2_matched_dets):
                    real_trk_idx = unmatched_tracks_stage1[trk_sub_idx]
                    det = d_low[det_idx]
                    self.tracks[real_trk_idx].update_detection(
                        tuple(det['bbox']),
                        det['class_name'],
                        det['confidence']
                    )
                    matched_track_indices.add(real_trk_idx)

            # -----------------------------------------------------------------
            # BƯỚC 3: Tạo track mới - CHỈ CHO D_high chưa ghép
            # (D_low chưa ghép xem như nhiễu nền, TUYỆT ĐỐI KHÔNG TẠO TRACK MỚI)
            # -----------------------------------------------------------------
            for j, det in enumerate(d_high):
                if j not in matched_dhigh_indices:
                    self.tracks.append(TrackedItem(
                        tuple(det['bbox']),
                        det['class_name'],
                        det['confidence']
                    ))
        else:
            # Chưa có track nào tồn tại: Khởi tạo track mới từ các detection D_high
            for det in d_high:
                self.tracks.append(TrackedItem(
                    tuple(det['bbox']),
                    det['class_name'],
                    det['confidence']
                ))

        # Lọc bỏ track đã mất dấu quá lâu (> max_age)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Trả về các track đã đủ độ tin cậy (hits >= min_hits)
        return [t for t in self.tracks if t.hits >= self.min_hits]
