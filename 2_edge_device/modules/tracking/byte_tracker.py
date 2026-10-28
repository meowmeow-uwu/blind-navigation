"""
byte_tracker.py - Bộ theo dõi vật cản đa đối tượng nhẹ (Real-time Object Tracking)
==================================================================================
Yêu cầu 3: Bám vết đối tượng thời gian thực & Triệt tiêu 100% nói lặp
- Cấp phát Track ID duy nhất cho từng vật cản từ khi xuất hiện đến khi biến mất.
- Quản lý trạng thái thông báo: Mỗi Track ID chỉ phát âm thanh thông báo ĐÚNG 1 LẦN,
  sau đó chuyển sang trạng thái theo dõi ngầm (Silent Tracking), triệt tiêu hoàn toàn
  hiện tượng loa lặp đi lặp lại ("Cửa... Cửa... Cửa").
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

    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        TrackedItem._id_counter += 1
        self.track_id = TrackedItem._id_counter
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Trạng thái thông báo âm thanh (Yêu cầu 3: Triệt tiêu nói lặp)
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

    def predict(self):
        self.age += 1
        self.time_since_update += 1


class LightweightTracker:
    """
    Bộ theo dõi nhẹ dựa trên IoU và thuật toán Hungarian.
    Cung cấp Track ID duy nhất với chi phí tính toán cực thấp trên Raspberry Pi 4.
    """

    def __init__(self, max_age: int = 15, min_hits: int = 2, iou_threshold: float = 0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[TrackedItem] = []

    def update(self, detections: list) -> List[TrackedItem]:
        """
        Nhận danh sách detection: [{'bbox': [x1, y1, x2, y2], 'class_name': str, 'confidence': float}, ...]
        Trả về danh sách TrackedItem đang hoạt động.
        """
        for t in self.tracks:
            t.predict()

        if not detections:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]

        if len(self.tracks) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = compute_iou(trk.bbox, tuple(det['bbox']))

            matched_tracks = set()
            matched_dets = set()

            if HAS_SCIPY:
                cost_matrix = 1.0 - iou_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_threshold:
                        self.tracks[r].update_detection(
                            tuple(detections[c]['bbox']),
                            detections[c]['class_name'],
                            detections[c]['confidence']
                        )
                        matched_tracks.add(r)
                        matched_dets.add(c)
            else:
                # Greedy matching fallback nếu không có scipy
                while True:
                    idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    max_iou = iou_matrix[idx]
                    if max_iou < self.iou_threshold:
                        break
                    r, c = idx
                    self.tracks[r].update_detection(
                        tuple(detections[c]['bbox']),
                        detections[c]['class_name'],
                        detections[c]['confidence']
                    )
                    matched_tracks.add(r)
                    matched_dets.add(c)
                    iou_matrix[r, :] = -1.0
                    iou_matrix[:, c] = -1.0

            # Tạo track mới cho detection chưa ghép
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    self.tracks.append(TrackedItem(
                        tuple(det['bbox']),
                        det['class_name'],
                        det['confidence']
                    ))
        else:
            for det in detections:
                self.tracks.append(TrackedItem(
                    tuple(det['bbox']),
                    det['class_name'],
                    det['confidence']
                ))

        # Lọc bỏ track đã mất dấu quá lâu
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Trả về các track đã đủ độ tin cậy
        return [t for t in self.tracks if t.hits >= self.min_hits]
