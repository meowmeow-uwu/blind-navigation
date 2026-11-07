# 2_edge_device/modules/vision_local/tracker.py
from typing import List, Dict, Any, Optional
import numpy as np
import supervision as sv

try:
    from trackers import ByteTrackTracker
    _BYTETRACK_AVAILABLE = True
except ImportError:
    _BYTETRACK_AVAILABLE = False


class ByteTrackerWrapper:
    """
    Wrapper tích hợp ByteTrack qua thư viện `trackers` (tương thích supervision 0.30+).

    - supervision <= 0.24 : dùng sv.ByteTrack trực tiếp
    - supervision >= 0.25 : ByteTrack tách ra package `trackers` (pip install trackers)
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        if not _BYTETRACK_AVAILABLE:
            raise RuntimeError(
                "[ByteTrackerWrapper] Cần cài thư viện 'trackers': uv pip install trackers"
            )

        self.classes: Optional[List[str]] = None

        # ByteTrackTracker sử dụng supervision Detections nội bộ
        self.tracker = ByteTrackTracker(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_iou_threshold=match_thresh,
            frame_rate=frame_rate
        )

    def update(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Cập nhật tracker từ kết quả phát hiện của detector.

        Args:
            boxes_xyxy: (N, 4) float32 – [x1, y1, x2, y2]
            confidences: (N,) float32 – score từng detection
            class_ids: (N,) int – class index từng detection
            classes: Danh sách tên lớp COCO (tùy chọn, để điền class_name)

        Returns:
            Danh sách dict đã có track_id:
            [{'track_id': 1, 'class_name': 'person', 'class_id': 0,
              'bbox': [x1,y1,x2,y2], 'confidence': 0.88}, ...]
        """
        if len(boxes_xyxy) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=boxes_xyxy.astype(np.float32),
                confidence=confidences.astype(np.float32),
                class_id=class_ids.astype(int)
            )

        tracked: sv.Detections = self.tracker.update(detections)

        results: List[Dict[str, Any]] = []
        if tracked.tracker_id is None or len(tracked) == 0:
            return results

        conf_arr = tracked.confidence if tracked.confidence is not None \
            else np.ones(len(tracked), dtype=np.float32)

        for xyxy, conf, cls_id, trk_id in zip(
            tracked.xyxy,
            conf_arr,
            tracked.class_id,
            tracked.tracker_id
        ):
            c_name = "unknown"
            if classes and 0 <= int(cls_id) < len(classes):
                c_name = classes[int(cls_id)]

            results.append({
                "track_id": int(trk_id),
                "class_name": c_name,
                "class_id": int(cls_id),
                "bbox": [int(round(v)) for v in xyxy],
                "confidence": float(conf)
            })

        return results

    def reset(self) -> None:
        """Reset toàn bộ tracklet — dùng khi chuyển nguồn video hoặc tua lại file."""
        self.tracker.reset()
