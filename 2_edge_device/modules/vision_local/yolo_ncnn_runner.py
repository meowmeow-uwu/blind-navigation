import ncnn
import cv2
import numpy as np
from .base_detector import BaseDetector

class YoloNcnnRunner(BaseDetector):
    def __init__(self, param_path: str, bin_path: str, target_size: int = 320):
        self.param_path = param_path
        self.bin_path = bin_path
        self.target_size = target_size # Thu nhỏ tensor để tăng FPS
        self.net = ncnn.Net()
        self.load_model()
        
        # Labels cho các vật cản cơ bản (COCO format)
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def load_model(self):
        # Kích hoạt tính toán đa luồng NEON SIMD
        self.net.opt.use_vulkan_compute = False # Pi 4 không có GPU mạnh, dùng CPU
        self.net.opt.num_threads = 4            # Tận dụng 4 lõi Cortex-A72
        
        self.net.load_param(self.param_path)
        self.net.load_model(self.bin_path)

    def detect(self, frame: np.ndarray) -> list:
        img_h, img_w = frame.shape[:2]
        
        # Tiền xử lý: Resize và chuẩn hóa theo chuẩn NCNN
        mat_in = ncnn.Mat.from_pixels_resize(
            frame, 
            ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
            img_w, img_h, 
            self.target_size, self.target_size
        )
        
        mean_vals = [103.53, 116.28, 123.675]
        norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        # Trích xuất đặc trưng (Inference)
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        
        ret, mat_out = ex.extract("out0")
        
        results = []
        if ret == 0:
            for i in range(mat_out.h):
                values = mat_out.row(i)
                class_id = int(values[0])
                confidence = values[1]
                
                # Lọc ngưỡng Confidence
                if confidence > 0.45:
                    x1 = int(values[2] * img_w)
                    y1 = int(values[3] * img_h)
                    x2 = int(values[4] * img_w)
                    y2 = int(values[5] * img_h)
                    
                    results.append({
                        'class_name': self.classes[class_id] if class_id < len(self.classes) else "unknown",
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence)
                    })
        return results