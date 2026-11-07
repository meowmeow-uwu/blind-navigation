# 2_edge_device/modules/decision/priority_scorer.py
"""
Tính điểm ưu tiên nguy hiểm (Priority Score) cho từng vật thể đang được track.

Công thức:
    score = w_class * C_i
           + w_pos   * (1 - |cx_obj - cx_frame| / cx_frame)
           + w_size  * (area_bbox / area_frame)

Giải thích:
    - C_i        : Trọng số nguy hiểm theo class (từ class_weights.yaml).
    - w_pos term : = 1.0 khi vật thể ở chính giữa, = 0.0 khi ở cạnh trái/phải cùng.
    - w_size term: Tỉ lệ diện tích bbox / diện tích toàn frame, đại diện độ gần.

Kết quả nằm trong khoảng [0, 1].
"""
from typing import Dict, List, Optional, Any
import numpy as np
import yaml
import os


class PriorityScorer:
    """
    Tính điểm ưu tiên và chọn Top-1 target nguy hiểm nhất trong một frame.
    """

    def __init__(
        self,
        class_weights_path: str,
        w_class: float = 0.40,
        w_position: float = 0.35,
        w_size: float = 0.25,
    ):
        """
        Args:
            class_weights_path: Đường dẫn đến file class_weights.yaml.
            w_class: Trọng số đóng góp của loại vật thể.
            w_position: Trọng số đóng góp của vị trí trung tâm.
            w_size: Trọng số đóng góp của kích thước bbox.
        """
        assert abs(w_class + w_position + w_size - 1.0) < 1e-6, \
            "Tổng w_class + w_position + w_size phải = 1.0"

        self.w_class = w_class
        self.w_position = w_position
        self.w_size = w_size

        self._class_weights: Dict[str, float] = {}
        self._default_weight: float = 0.5
        self._load_class_weights(class_weights_path)

    def _load_class_weights(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[PriorityScorer] Không tìm thấy: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._default_weight = float(data.get("default_weight", 0.5))
        self._class_weights = {
            k: float(v) for k, v in data.get("weights", {}).items()
        }

    def class_weight(self, class_name: str) -> float:
        return self._class_weights.get(class_name, self._default_weight)

    def score(
        self,
        obj: Dict[str, Any],
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Tính điểm ưu tiên cho một vật thể đang track.

        Args:
            obj: Dict từ ByteTrackerWrapper.update() với key: bbox, class_name, class_id.
            frame_width: Chiều rộng khung hình gốc (pixel).
            frame_height: Chiều cao khung hình gốc (pixel).

        Returns:
            Điểm ưu tiên trong khoảng [0.0, 1.0].
        """
        x1, y1, x2, y2 = obj["bbox"]
        cx_obj = (x1 + x2) / 2.0
        cy_obj = (y1 + y2) / 2.0  # noqa: F841 (dự phòng)

        cx_frame = frame_width / 2.0
        area_frame = float(frame_width * frame_height)

        # Thành phần class
        c_i = self.class_weight(obj["class_name"])

        # Thành phần vị trí: tâm vật thể càng gần trục giữa → điểm càng cao
        pos_score = 1.0 - abs(cx_obj - cx_frame) / cx_frame if cx_frame > 0 else 0.0
        pos_score = float(np.clip(pos_score, 0.0, 1.0))

        # Thành phần kích thước: bbox càng lớn → vật thể càng gần → điểm càng cao
        bbox_area = float(max(0, (x2 - x1)) * max(0, (y2 - y1)))
        size_score = float(np.clip(bbox_area / area_frame, 0.0, 1.0)) if area_frame > 0 else 0.0

        return self.w_class * c_i + self.w_position * pos_score + self.w_size * size_score

    def rank(
        self,
        tracked: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> List[Dict[str, Any]]:
        """
        Sắp xếp danh sách tracked objects theo điểm ưu tiên giảm dần.
        Mỗi dict được bổ sung key 'priority_score'.

        Returns:
            Danh sách đã sắp xếp, phần tử đầu tiên là Top-1 (nguy hiểm nhất).
        """
        scored = []
        for obj in tracked:
            s = self.score(obj, frame_width, frame_height)
            scored.append({**obj, "priority_score": round(s, 4)})
        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        return scored

    def top1(
        self,
        tracked: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Optional[Dict[str, Any]]:
        """
        Trả về vật thể nguy hiểm nhất (Top-1), hoặc None nếu danh sách trống.
        """
        ranked = self.rank(tracked, frame_width, frame_height)
        return ranked[0] if ranked else None
