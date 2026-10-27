"""
byte_tracker.py - Bộ theo dõi vật cản đa đối tượng nhẹ cho Raspberry Pi 4
========================================================================
Quản lý vòng đời của từng vật thể nhận dạng (Track ID), liên kết các Bounding Box
qua thời gian để duy trì bộ lọc Kalman khoảng cách và cờ chống lặp âm thanh.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Tính Intersection-over-Union giữa 2 Bounding Box [x1, y1, x2, y2]."""
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
    """Đối tượng vật cản được theo dõi liên tục qua các khung hình."""

    _id_counter = 0

    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        TrackedItem._id_counter += 1
        self.track_id = TrackedItem._id_counter
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.announced = False
        self.last_announced_time = 0.0
        self.last_announced_distance = 0.0

        # Tham chiếu bộ lọc Kalman khoảng cách (được Fusion Engine khởi tạo)
        self.kalman_dist = None

        # Vùng điều hướng làn đường ("left", "center", "right")
        self.lane = "center"
        self.risk_score = 0.0
        self.fused_distance = 2.5
        self.ttc = 999.0
        self.is_in_ultrasonic_beam = False
        self.source_description = "camera_only"

    def update_detection(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        """Cập nhật bounding box mới khi có match."""
        # Làm mịn toạ độ bounding box (alpha filter)
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

    def predict(self):
        """Tăng tuổi track."""
        self.age += 1
        self.time_since_update += 1


class LightweightTracker:
    """
    Theo dõi đa đối tượng thời gian thực sử dụng IoU và Hungarian Algorithm.
    Không dùng Re-ID để đảm bảo tốc độ cao nhất trên Raspberry Pi 4.
    """

    def __init__(self, max_age: int = 10, min_hits: int = 2, iou_threshold: float = 0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[TrackedItem] = []

    def update(self, detections: list) -> List[TrackedItem]:
        """
        Nhận danh sách detections từ YOLO:
        [{'bbox': [x1, y1, x2, y2], 'class_name': 'chair', 'confidence': 0.8}, ...]
        Trả về danh sách TrackedItem còn sống.
        """
        # 1. Dự đoán cho các track hiện tại
        for t in self.tracks:
            t.predict()

        if not detections:
            # Xoá các track đã mất dấu quá lâu
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]

        # 2. Ghép cặp IoU giữa tracks và detections mới
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = compute_iou(trk.bbox, tuple(det['bbox']))

            # Giải bài toán gán Hungarian
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self.tracks[r].update_detection(
                        tuple(detections[c]['bbox']),
                        detections[c]['class_name'],
                        detections[c]['confidence']
                    )
                    matched_tracks.add(r)
                    matched_dets.add(c)

            # Tạo track mới cho detection chưa ghép được
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    new_item = TrackedItem(
                        tuple(det['bbox']),
                        det['class_name'],
                        det['confidence']
                    )
                    self.tracks.append(new_item)
        else:
            # Khởi tạo toàn bộ track mới
            for det in detections:
                new_item = TrackedItem(
                    tuple(det['bbox']),
                    det['class_name'],
                    det['confidence']
                )
                self.tracks.append(new_item)

        # 3. Lọc bỏ track cũ
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Trả về các track đã xác thực (đủ min_hits và vừa được cập nhật)
        return [t for t in self.tracks if t.hits >= self.min_hits]
