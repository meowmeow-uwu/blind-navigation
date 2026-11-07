# 2_edge_device/modules/utils/visualizer.py
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

# Bảng màu HSV cố định theo class_id – đảm bảo mỗi class luôn cùng một màu
_COLOR_POOL = [
    (0, 220, 255),    # Vàng sáng
    (0, 165, 255),    # Cam
    (0, 128, 255),    # Cam đậm
    (255, 100, 100),  # Hồng
    (100, 100, 255),  # Đỏ nhạt
    (100, 255, 100),  # Xanh lá nhạt
    (255, 255, 100),  # Cyan nhạt
    (255, 100, 255),  # Tím nhạt
]


def _get_color(idx: int) -> Tuple[int, int, int]:
    return _COLOR_POOL[idx % len(_COLOR_POOL)]


def draw_tracked_detections(
    frame: np.ndarray,
    tracked: List[Dict[str, Any]],
    top1_track_id: Optional[int] = None
) -> np.ndarray:
    """
    Vẽ bounding box + Track ID + class/confidence cho từng tracked object.
    Object được chọn làm Top-1 (nguy hiểm nhất) sẽ được tô đậm + viền kép màu đỏ.

    Args:
        frame: Ảnh BGR (H, W, 3).
        tracked: Danh sách dict từ ByteTrackerWrapper.update().
        top1_track_id: track_id của mục tiêu nguy hiểm nhất (Top-1) để highlight.

    Returns:
        frame đã được annotate.
    """
    for obj in tracked:
        x1, y1, x2, y2 = obj["bbox"]
        tid = obj["track_id"]
        cls_id = obj["class_id"]
        label = f"#{tid} {obj['class_name']} {obj['confidence']:.2f}"
        is_top1 = (tid == top1_track_id)

        base_color = (0, 0, 220) if is_top1 else _get_color(cls_id)
        thickness = 3 if is_top1 else 2

        # Vẽ bbox chính
        cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, thickness)

        # Vẽ viền ngoài trắng mỏng cho Top-1 để nổi bật hơn
        if is_top1:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

        # Background cho label
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = y1 - 8 if y1 > lh + 10 else y2 + lh + 4
        cv2.rectangle(
            frame,
            (x1, label_y - lh - baseline),
            (x1 + lw + 4, label_y + baseline),
            base_color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame


def draw_lane_overlay(
    frame: np.ndarray,
    alpha: float = 0.10
) -> np.ndarray:
    """
    Vẽ 3 làn (trái/giữa/phải) dạng overlay bán trong suốt để debug policy.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    lane_w = w // 3

    colors = [(60, 60, 200), (60, 200, 60), (60, 60, 200)]  # Trái-Xanh Giữa-Lục Phải-Xanh
    labels_txt = ["TRAI", "GIUA", "PHAI"]

    for i in range(3):
        x_start = i * lane_w
        x_end = (i + 1) * lane_w if i < 2 else w
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), colors[i], -1)
        cv2.putText(
            frame,
            labels_txt[i],
            (x_start + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_alert_banner(
    frame: np.ndarray,
    alert_text: str,
    color: Tuple[int, int, int] = (0, 0, 220)
) -> np.ndarray:
    """
    Vẽ banner cảnh báo điều hướng ở góc dưới khung hình.
    """
    if not alert_text:
        return frame

    h, w = frame.shape[:2]
    banner_h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    tx = max(0, (w - tw) // 2)
    ty = h - (banner_h - th) // 2 - 2

    cv2.putText(frame, alert_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
