"""Base detector interface for unified outputs."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    """Unified detection output.
    
    Attributes:
        class_id: Integer class ID
        class_name: Human-readable class name
        confidence: Confidence score [0.0, 1.0]
        bbox_xyxy: Bounding box as (x1, y1, x2, y2) in original image coordinates
    """
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    
    def __post_init__(self):
        """Validate detection fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {self.bbox_xyxy}")


class Detector(Protocol):
    """Protocol for object detectors."""
    
    def predict(self, image_path: str, conf: float = 0.25, iou: float = 0.45) -> list[Detection]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            iou: NMS IoU threshold
        
        Returns:
            List of Detection objects
        """
        ...
