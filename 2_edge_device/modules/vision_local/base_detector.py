from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def load_model(self):
        """Khởi tạo và tải trọng số mô hình vào RAM/NCNN"""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """
        Nhận vào một frame ảnh, trả về danh sách các vật thể.
        Đầu ra kỳ vọng: [{'class_name': 'person', 'bbox': [x1, y1, x2, y2], 'confidence': 0.85}, ...]
        """
        pass