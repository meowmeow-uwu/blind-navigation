"""Tests for detection postprocessing."""

import numpy as np
import pytest

from models.base_detector import Detection
from models.onnx_detector import nms


def test_nms_removes_duplicate_boxes():
    """Test that NMS removes overlapping boxes with lower scores."""
    # Two boxes with high IoU
    boxes = np.array([
        [10, 10, 50, 50],  # Box 1
        [15, 15, 55, 55],  # Box 2 - overlaps with Box 1
    ])
    scores = np.array([0.9, 0.7])
    
    keep = nms(boxes, scores, iou_threshold=0.5)
    
    # Should keep only the first box (higher score)
    assert len(keep) == 1
    assert keep[0] == 0


def test_nms_keeps_non_overlapping_boxes():
    """Test that NMS keeps boxes with low IoU."""
    # Two boxes with no overlap
    boxes = np.array([
        [10, 10, 50, 50],
        [100, 100, 150, 150],
    ])
    scores = np.array([0.9, 0.8])
    
    keep = nms(boxes, scores, iou_threshold=0.5)
    
    # Should keep both
    assert len(keep) == 2


def test_detection_dataclass_validation():
    """Test that Detection validates fields."""
    # Valid detection
    det = Detection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
    )
    
    assert det.confidence == 0.95
    assert det.bbox_xyxy == (10.0, 20.0, 100.0, 200.0)
    
    # Invalid confidence
    with pytest.raises(ValueError):
        Detection(
            class_id=0,
            class_name="person",
            confidence=1.5,  # > 1.0
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
        )
    
    # Invalid bbox (x2 <= x1)
    with pytest.raises(ValueError):
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox_xyxy=(100.0, 20.0, 10.0, 200.0),  # x2 < x1
        )


def test_nms_sorts_by_score():
    """Test that NMS prioritizes higher-scoring boxes."""
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
        [14, 14, 54, 54],
    ])
    # Middle box has highest score
    scores = np.array([0.7, 0.9, 0.6])
    
    keep = nms(boxes, scores, iou_threshold=0.5)
    
    # Should keep the middle box (index 1)
    assert 1 in keep
    # Should suppress others
    assert len(keep) == 1
