"""
trackers.py - Cài đặt thống nhất 4 cấu hình Tracker cho thực nghiệm đối sánh (T4: PBL6)
========================================================================================
Cung cấp cùng một giao diện đầu vào/đầu ra `update(detections, frame=None) -> List[TrackedItem]`
để chạy thực nghiệm công bằng trên cùng một chuỗi video/detections:
1. RawNoTracker (`raw_yolo`): Không bám vết, gán ID mới cho mọi detection trên từng frame.
2. SORTTracker (`sort`): Bewley et al. (ICIP 2016) - Kalman 2D + Hungarian IoU 1 tầng,
   chỉ dùng detection tin cậy cao (>= 0.45), bỏ qua detection mờ/thấp.
3. DeepSORTTracker (`deepsort`): Wojke et al. (ICIP 2017) - Kalman 2D + Matching Cascade
   kết hợp Cosine Distance của vector ngoại hình 128-D (hỗ trợ ONNX Re-ID qua cv2.dnn
   hoặc bộ trích xuất HOG+Color fallback được gắn nhãn tường minh).
4. EdgeByteTrackWrapper (`edge_bytetrack`): Bộ bám vết 2 tầng IoU của dự án (LightweightTracker).
"""

import os
import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

EDGE_ROOT = Path(__file__).resolve().parents[2]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from modules.tracking.byte_tracker import TrackedItem, LightweightTracker, compute_iou


def _bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Chuyển [x1, y1, x2, y2] -> vector quan sát [cx, cy, area, aspect_ratio]^T."""
    w = max(1.0, float(bbox[2] - bbox[0]))
    h = max(1.0, float(bbox[3] - bbox[1]))
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / h
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def _x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
    """Chuyển vector trạng thái Kalman [cx, cy, s, r, ...] -> [x1, y1, x2, y2]."""
    cx = float(x[0, 0])
    cy = float(x[1, 0])
    s = max(1.0, float(x[2, 0]))
    r = max(0.05, float(x[3, 0]))
    w = math.sqrt(s * r)
    h = s / max(1e-3, w)
    return (
        int(round(cx - w / 2.0)),
        int(round(cy - h / 2.0)),
        int(round(cx + w / 2.0)),
        int(round(cy + h / 2.0)),
    )


def _linear_assignment(cost_matrix: np.ndarray, max_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Giải bài toán gán cặp tối ưu (Hungarian hoặc Greedy fallback) với ngưỡng chi phí."""
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    matches = []
    matched_r = set()
    matched_c = set()

    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_cost:
                matches.append((int(r), int(c)))
                matched_r.add(int(r))
                matched_c.add(int(c))
    else:
        work = cost_matrix.copy()
        while True:
            r, c = np.unravel_index(np.argmin(work), work.shape)
            if work[r, c] > max_cost:
                break
            matches.append((int(r), int(c)))
            matched_r.add(int(r))
            matched_c.add(int(c))
            work[r, :] = max_cost + 1e6
            work[:, c] = max_cost + 1e6

    unmatched_r = [i for i in range(n_rows) if i not in matched_r]
    unmatched_c = [j for j in range(n_cols) if j not in matched_c]
    return matches, unmatched_r, unmatched_c


class KalmanBox2D:
    """Bộ lọc Kalman 7 trạng thái [cx, cy, s, r, d_cx, d_cy, d_s] chuẩn SORT / DeepSORT."""

    def __init__(self, bbox: Tuple[int, int, int, int]):
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[:4, :4] = np.eye(4, dtype=np.float64)

        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = _bbox_to_z(bbox)

    def predict(self) -> Tuple[int, int, int, int]:
        """Dự báo vị trí Bounding Box ở khung hình kế tiếp."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _x_to_bbox(self.x)

    def update(self, bbox: Tuple[int, int, int, int]):
        """Cập nhật bộ lọc bằng quan sát mới."""
        z = _bbox_to_z(bbox)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(7) - K @ self.H
        # Cập nhật hiệp phương sai dạng đối xứng Joseph
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


class ReIDFeatureExtractor:
    """
    Bộ trích xuất vector đặc trưng ngoại hình 128-D cho DeepSORT.
    Ưu tiên nạp mô hình CNN ONNX (ví dụ osnet_x0_25.onnx / mars-small128.onnx) qua cv2.dnn.
    Nếu chưa có tệp ONNX, dùng bộ trích xuất HOG + HSV Spatial 128-D (gắn nhãn tường minh).
    """

    def __init__(self, onnx_path: Optional[str] = None):
        self.onnx_path = onnx_path
        self.net = None
        self.backend_label = "Sobel+HSV_Fallback (Chưa nạp ONNX CNN)"
        self.is_real_cnn = False

        default_candidates = [
            onnx_path,
            str(EDGE_ROOT / "modules/tracking/models/osnet_x0_25.onnx"),
            str(EDGE_ROOT / "modules/tracking/models/mars_small128.onnx"),
            str(EDGE_ROOT / "modules/tracking/models/reid_128d.onnx"),
        ]
        for path in default_candidates:
            if path and os.path.exists(path):
                try:
                    self.net = cv2.dnn.readNetFromONNX(path)
                    self.onnx_path = path
                    self.backend_label = f"CNN 128-D ({os.path.basename(path)})"
                    self.is_real_cnn = True
                    break
                except Exception:
                    self.net = None

    def extract(self, frame: Optional[np.ndarray], bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Trích xuất vector đặc trưng L2-normalized cho danh sách các BBox trên frame."""
        if not bboxes:
            return []

        if frame is None or frame.size == 0:
            return [np.ones(128, dtype=np.float32) / math.sqrt(128.0) for _ in bboxes]

        h_img, w_img = frame.shape[:2]
        features: List[np.ndarray] = []

        for x1, y1, x2, y2 in bboxes:
            cx1 = max(0, min(w_img - 1, int(x1)))
            cy1 = max(0, min(h_img - 1, int(y1)))
            cx2 = max(cx1 + 2, min(w_img, int(x2)))
            cy2 = max(cy1 + 2, min(h_img, int(y2)))
            crop = frame[cy1:cy2, cx1:cx2]

            if self.net is not None:
                # Chuẩn bị đầu vào chuẩn Re-ID (height=128/256, width=64/128)
                blob = cv2.dnn.blobFromImage(
                    crop, scalefactor=1.0 / 255.0, size=(64, 128),
                    mean=(0.485, 0.456, 0.406), swapRB=True, crop=False
                )
                self.net.setInput(blob)
                feat = self.net.forward().flatten().astype(np.float32)
            else:
                # Fallback Sobel Gradient + Spatial HSV Descriptor (128 chiều, tương thích OpenCV 4/5)
                resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)
                hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                hist_top = cv2.calcHist([hsv[:64, :]], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
                hist_bot = cv2.calcHist([hsv[64:, :]], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                grad_top = cv2.calcHist([ang[:64, :]], [0], None, [32], [0, 360]).flatten() * (mag[:64, :].mean() + 1e-3)
                grad_bot = cv2.calcHist([ang[64:, :]], [0], None, [32], [0, 360]).flatten() * (mag[64:, :].mean() + 1e-3)

                feat = np.concatenate([
                    hist_top / (np.linalg.norm(hist_top) + 1e-6),
                    hist_bot / (np.linalg.norm(hist_bot) + 1e-6),
                    grad_top / (np.linalg.norm(grad_top) + 1e-6),
                    grad_bot / (np.linalg.norm(grad_bot) + 1e-6),
                ]).astype(np.float32)

            norm = float(np.linalg.norm(feat))
            if norm > 1e-6:
                feat = feat / norm
            features.append(feat)

        return features


class BaseBenchmarkTracker:
    """Lớp cơ sở thống nhất giao diện cho mọi tracker thực nghiệm."""
    name: str = "base"
    reid_info: str = "Không"

    def reset(self):
        self.tracks: List[TrackedItem] = []

    def update(self, detections: list, frame: Optional[np.ndarray] = None) -> List[TrackedItem]:
        raise NotImplementedError


class RawNoTracker(BaseBenchmarkTracker):
    """Cấu hình 1: Raw YOLO - Không dùng tracker, mọi detection đều sinh Track ID mới."""
    name = "raw_yolo"
    reid_info = "Không"

    def __init__(self, conf_thresh: float = 0.20):
        self.conf_thresh = conf_thresh
        self.tracks: List[TrackedItem] = []

    def reset(self):
        TrackedItem.reset_counter(0)
        self.tracks = []

    def update(self, detections: list, frame: Optional[np.ndarray] = None) -> List[TrackedItem]:
        self.tracks = []
        for det in detections:
            conf = float(det.get("confidence", 0.5))
            if conf >= self.conf_thresh:
                item = TrackedItem(tuple(det["bbox"]), det["class_name"], conf)
                item.hits = 2  # Đưa thẳng qua bộ lọc downstream như Raw YOLO không tracker
                self.tracks.append(item)
        return list(self.tracks)


class SORTTracker(BaseBenchmarkTracker):
    """
    Cấu hình 2: SORT (Bewley et al., ICIP 2016).
    - Chỉ dùng detection điểm tin cậy cao (>= 0.45), vứt bỏ detection mờ/bị che khuất.
    - Dự báo Kalman 2D và gán cặp 1 tầng bằng IoU + Hungarian.
    """
    name = "sort"
    reid_info = "Không"

    def __init__(self, max_age: int = 12, min_hits: int = 2, iou_threshold: float = 0.25, conf_thresh: float = 0.45):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.conf_thresh = conf_thresh
        self.tracks: List[TrackedItem] = []
        self._kf_boxes: Dict[int, KalmanBox2D] = {}

    def reset(self):
        TrackedItem.reset_counter(0)
        self.tracks = []
        self._kf_boxes = {}

    def update(self, detections: list, frame: Optional[np.ndarray] = None) -> List[TrackedItem]:
        for t in self.tracks:
            t.predict()
            kf = self._kf_boxes.get(t.track_id)
            if kf is not None:
                t.bbox = kf.predict()

        high_dets = [d for d in detections if float(d.get("confidence", 0.5)) >= self.conf_thresh]

        if self.tracks and high_dets:
            cost = np.ones((len(self.tracks), len(high_dets)), dtype=np.float64)
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(high_dets):
                    cost[i, j] = 1.0 - compute_iou(trk.bbox, tuple(det["bbox"]))
            matches, _, unmatched_d = _linear_assignment(cost, max_cost=1.0 - self.iou_threshold)

            for r, c in matches:
                det = high_dets[c]
                trk = self.tracks[r]
                self._kf_boxes[trk.track_id].update(tuple(det["bbox"]))
                trk.update_detection(tuple(det["bbox"]), det["class_name"], float(det["confidence"]))
        else:
            unmatched_d = list(range(len(high_dets)))

        for j in unmatched_d:
            det = high_dets[j]
            item = TrackedItem(tuple(det["bbox"]), det["class_name"], float(det["confidence"]))
            self._kf_boxes[item.track_id] = KalmanBox2D(tuple(det["bbox"]))
            self.tracks.append(item)

        alive = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive.append(t)
            else:
                self._kf_boxes.pop(t.track_id, None)
        self.tracks = alive

        return [t for t in self.tracks if t.hits >= self.min_hits]


class DeepSORTTracker(BaseBenchmarkTracker):
    """
    Cấu hình 3: DeepSORT (Wojke et al., ICIP 2017).
    - Kalman 2D + Matching Cascade dùng khoảng cách Cosine trên đặc trưng ngoại hình 128-D
      kết hợp vòng vớt IoU cho các track chưa ghép.
    """
    name = "deepsort"

    def __init__(self,
                 max_age: int = 12,
                 min_hits: int = 2,
                 iou_threshold: float = 0.25,
                 conf_thresh: float = 0.45,
                 max_cosine_distance: float = 0.35,
                 nn_budget: int = 30,
                 onnx_reid_path: Optional[str] = None):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.conf_thresh = conf_thresh
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        self.extractor = ReIDFeatureExtractor(onnx_reid_path)
        self.reid_info = self.extractor.backend_label

        self.tracks: List[TrackedItem] = []
        self._kf_boxes: Dict[int, KalmanBox2D] = {}
        self._galleries: Dict[int, List[np.ndarray]] = {}

    def reset(self):
        TrackedItem.reset_counter(0)
        self.tracks = []
        self._kf_boxes = {}
        self._galleries = {}

    def _cosine_distance(self, trk_id: int, feat: np.ndarray) -> float:
        gallery = self._galleries.get(trk_id)
        if not gallery:
            return 1.0
        # Lấy khoảng cách Cosine nhỏ nhất trong gallery đặc trưng gần nhất
        sims = [float(np.dot(g, feat)) for g in gallery]
        return max(0.0, 1.0 - max(sims))

    def _record_feature(self, trk_id: int, feat: np.ndarray):
        gal = self._galleries.setdefault(trk_id, [])
        gal.append(feat)
        if len(gal) > self.nn_budget:
            gal.pop(0)

    def update(self, detections: list, frame: Optional[np.ndarray] = None) -> List[TrackedItem]:
        for t in self.tracks:
            t.predict()
            kf = self._kf_boxes.get(t.track_id)
            if kf is not None:
                t.bbox = kf.predict()

        high_dets = [d for d in detections if float(d.get("confidence", 0.5)) >= self.conf_thresh]
        det_boxes = [tuple(d["bbox"]) for d in high_dets]
        det_feats = self.extractor.extract(frame, det_boxes)

        matched_trks = set()
        matched_dets = set()

        # Vòng 1: Matching Cascade theo ngoại hình (Cosine) kết hợp cổng IoU/khoảng cách cho các track gần đây
        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.hits >= self.min_hits]
        if confirmed_indices and high_dets:
            unmatched_d_pool = list(range(len(high_dets)))
            # Cascade ưu tiên các track vừa cập nhật (age_gap nhỏ trước)
            for age_gap in range(self.max_age + 1):
                if not unmatched_d_pool:
                    break
                tier_trks = [i for i in confirmed_indices if i not in matched_trks and (self.tracks[i].time_since_update - 1) == age_gap]
                if not tier_trks:
                    continue

                cost = np.full((len(tier_trks), len(unmatched_d_pool)), 1e5, dtype=np.float64)
                for r_idx, t_idx in enumerate(tier_trks):
                    trk = self.tracks[t_idx]
                    for c_idx, d_idx in enumerate(unmatched_d_pool):
                        cos_d = self._cosine_distance(trk.track_id, det_feats[d_idx])
                        iou = compute_iou(trk.bbox, det_boxes[d_idx])
                        # Kết hợp trọng số cosine + hình học để tránh khớp nhầm hai vật giống nhau ở xa
                        if cos_d <= self.max_cosine_distance or iou >= 0.5:
                            cost[r_idx, c_idx] = 0.7 * cos_d + 0.3 * (1.0 - iou)

                matches, _, _ = _linear_assignment(cost, max_cost=0.65)
                newly_matched_d = set()
                for r_idx, c_idx in matches:
                    t_idx = tier_trks[r_idx]
                    d_idx = unmatched_d_pool[c_idx]
                    det = high_dets[d_idx]
                    trk = self.tracks[t_idx]
                    self._kf_boxes[trk.track_id].update(det_boxes[d_idx])
                    trk.update_detection(det_boxes[d_idx], det["class_name"], float(det["confidence"]))
                    self._record_feature(trk.track_id, det_feats[d_idx])
                    matched_trks.add(t_idx)
                    matched_dets.add(d_idx)
                    newly_matched_d.add(d_idx)
                unmatched_d_pool = [d for d in unmatched_d_pool if d not in newly_matched_d]

        # Vòng 2: Ghép IoU fallback cho track mới (tentative) hoặc track rớt chưa ghép
        rem_trks = [i for i in range(len(self.tracks)) if i not in matched_trks and self.tracks[i].time_since_update <= 2]
        rem_dets = [j for j in range(len(high_dets)) if j not in matched_dets]
        if rem_trks and rem_dets:
            iou_cost = np.ones((len(rem_trks), len(rem_dets)), dtype=np.float64)
            for r_idx, t_idx in enumerate(rem_trks):
                for c_idx, d_idx in enumerate(rem_dets):
                    iou_cost[r_idx, c_idx] = 1.0 - compute_iou(self.tracks[t_idx].bbox, det_boxes[d_idx])
            matches, _, _ = _linear_assignment(iou_cost, max_cost=1.0 - self.iou_threshold)
            for r_idx, c_idx in matches:
                t_idx = rem_trks[r_idx]
                d_idx = rem_dets[c_idx]
                det = high_dets[d_idx]
                trk = self.tracks[t_idx]
                self._kf_boxes[trk.track_id].update(det_boxes[d_idx])
                trk.update_detection(det_boxes[d_idx], det["class_name"], float(det["confidence"]))
                self._record_feature(trk.track_id, det_feats[d_idx])
                matched_trks.add(t_idx)
                matched_dets.add(d_idx)

        # Tạo track mới cho detections chưa ghép
        for d_idx, det in enumerate(high_dets):
            if d_idx not in matched_dets:
                item = TrackedItem(det_boxes[d_idx], det["class_name"], float(det["confidence"]))
                self._kf_boxes[item.track_id] = KalmanBox2D(det_boxes[d_idx])
                self._record_feature(item.track_id, det_feats[d_idx])
                self.tracks.append(item)

        alive = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive.append(t)
            else:
                self._kf_boxes.pop(t.track_id, None)
                self._galleries.pop(t.track_id, None)
        self.tracks = alive

        return [t for t in self.tracks if t.hits >= self.min_hits]


class EdgeByteTrackWrapper(BaseBenchmarkTracker):
    """Cấu hình 4: Edge Two-Stage ByteTrack của dự án PBL6 (LightweightTracker)."""
    name = "edge_bytetrack"
    reid_info = "Không (2-Stage IoU)"

    def __init__(self,
                 max_age: int = 12,
                 min_hits: int = 2,
                 iou_threshold: float = 0.25,
                 high_thresh: float = 0.45,
                 low_thresh: float = 0.20):
        self.tracker = LightweightTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            high_thresh=high_thresh,
            low_thresh=low_thresh,
        )

    @property
    def tracks(self) -> List[TrackedItem]:
        return self.tracker.tracks

    def reset(self):
        TrackedItem.reset_counter(0)
        self.tracker.tracks = []

    def update(self, detections: list, frame: Optional[np.ndarray] = None) -> List[TrackedItem]:
        return self.tracker.update(detections)


def create_tracker(tracker_name: str, onnx_reid_path: Optional[str] = None) -> BaseBenchmarkTracker:
    """Factory khởi tạo tracker theo tên cấu hình chuẩn."""
    key = tracker_name.lower()
    if key == "raw_yolo":
        return RawNoTracker(conf_thresh=0.20)
    elif key == "sort":
        return SORTTracker(max_age=12, min_hits=2, iou_threshold=0.25, conf_thresh=0.45)
    elif key == "deepsort":
        return DeepSORTTracker(max_age=12, min_hits=2, iou_threshold=0.25, conf_thresh=0.45, onnx_reid_path=onnx_reid_path)
    elif key == "edge_bytetrack":
        return EdgeByteTrackWrapper(max_age=12, min_hits=2, iou_threshold=0.25, high_thresh=0.45, low_thresh=0.20)
    raise ValueError(f"Tracker không hợp lệ: {tracker_name}")
