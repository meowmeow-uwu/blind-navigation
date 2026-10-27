from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Initialize the runtime and load model weights."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return detections with class, confidence, bbox and position."""
        raise NotImplementedError