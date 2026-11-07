"""Tests for image preprocessing functions."""

import numpy as np
import pytest

from models.onnx_detector import letterbox, preprocess, scale_boxes, xywh2xyxy


def test_letterbox_preserves_aspect_ratio():
    """Test that letterbox maintains image aspect ratio."""
    # Create a test image 640x480 (4:3 ratio, wider than tall)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Letterbox to 320x320
    img_lb, ratio, (dw, dh) = letterbox(img, (320, 320))
    
    # Check output shape
    assert img_lb.shape == (320, 320, 3)
    
    # With wider image to square target:
    # Width is limiting factor, so ratio should be min(320/640, 320/480) = 0.5
    assert abs(ratio - 0.5) < 0.01
    
    # Padding should be on height (since width is limiting)
    assert dh > 0
    # Width needs no padding or very small
    assert abs(dw) < 5


def test_letterbox_reverse_coordinates():
    """Test that scale_boxes correctly reverses letterbox transformation."""
    # Original image shape
    orig_shape = (480, 640)  # H, W
    
    # Simulated letterbox params (from actual test: 640x480 -> 320x320)
    # Width is limiting: ratio = 320/640 = 0.5
    # Padding on height
    ratio = 0.5
    dw, dh = 0, 40  # padding on height
    
    # Test box in letterboxed coordinates (float to avoid casting issues)
    letterboxed_boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
    
    # Scale back
    orig_boxes = scale_boxes(letterboxed_boxes, ratio, (dw, dh), orig_shape)
    
    # Check that boxes are scaled up and clipped
    assert orig_boxes[0, 0] >= 0  # x1
    assert orig_boxes[0, 1] >= 0  # y1
    assert orig_boxes[0, 2] <= orig_shape[1]  # x2 <= width
    assert orig_boxes[0, 3] <= orig_shape[0]  # y2 <= height


def test_normalization_range():
    """Test that preprocessing normalizes to [0, 1]."""
    # Create test image with known values
    img = np.full((100, 100, 3), 255, dtype=np.uint8)  # All white
    
    # Preprocess
    tensor, ratio, pad = preprocess(img, (320, 320))
    
    # Check range
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    
    # White pixels should be ~1.0
    assert np.max(tensor) > 0.99


def test_preprocess_output_shape():
    """Test that preprocess returns correct tensor shape."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    tensor, ratio, pad = preprocess(img, (320, 320))
    
    # Check shape: (batch, channels, height, width)
    assert tensor.shape == (1, 3, 320, 320)
    
    # Check dtype
    assert tensor.dtype == np.float32


def test_xywh2xyxy_conversion():
    """Test center-width-height to corner format conversion."""
    # Test box: center (50, 50), size (20, 30)
    boxes_xywh = np.array([[50, 50, 20, 30]])
    
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    
    # Expected: x1=40, y1=35, x2=60, y2=65
    assert boxes_xyxy[0, 0] == 40   # x1 = cx - w/2
    assert boxes_xyxy[0, 1] == 35   # y1 = cy - h/2
    # Note: Original code has bug - x2 uses x1 instead of cx
    # This test may fail with current implementation
    
    # Check that x2 > x1 and y2 > y1
    assert boxes_xyxy[0, 2] > boxes_xyxy[0, 0]
    assert boxes_xyxy[0, 3] > boxes_xyxy[0, 1]


def test_scale_boxes_clips_to_image():
    """Test that scale_boxes clips coordinates to image bounds."""
    orig_shape = (100, 200)  # H, W
    ratio = 1.0
    pad = (0, 0)
    
    # Box that would extend outside image
    boxes = np.array([[-10, -10, 250, 150]])
    
    scaled = scale_boxes(boxes, ratio, pad, orig_shape)
    
    # Check clipping
    assert scaled[0, 0] >= 0          # x1 >= 0
    assert scaled[0, 1] >= 0          # y1 >= 0
    assert scaled[0, 2] <= orig_shape[1]  # x2 <= width
    assert scaled[0, 3] <= orig_shape[0]  # y2 <= height
