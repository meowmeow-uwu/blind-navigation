# 2_edge_device/modules/vision_local/yolo_ncnn_runner.py
import ncnn
import cv2
import numpy as np
from .base_detector import BaseDetector

class YoloNcnnRunner(BaseDetector):
    def __init__(self, param_path: str, bin_path: str, target_size: int = 320, conf_thresh: float = 0.45, nms_thresh: float = 0.45):
        self.param_path = param_path
        self.bin_path = bin_path
        self.target_size = target_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        self.net = ncnn.Net()
        self.load_model()
        
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

        # Khởi tạo ma trận anchors tương ứng với target_size
        self.anchors, self.strides = self._generate_anchors(self.target_size)
        self.dfl_weights = np.arange(16, dtype=np.float32)

    def _make_grid(self, nx, ny):
        grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        return np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) + 0.5

    def _generate_anchors(self, target_size):
        strides = [8, 16, 32]
        anchors_list = []
        strides_list = []
        
        for stride in strides:
            grid_size = target_size // stride
            grid = self._make_grid(grid_size, grid_size) * stride
            anchors_list.append(grid)
            strides_list.append(np.full(grid.shape[0], stride))
            
        return np.concatenate(anchors_list, axis=0), np.concatenate(strides_list, axis=0)

    def load_model(self):
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4                    # Khai thác tối đa 4 lõi CPU
        self.net.opt.use_fp16_packed = True             # Ép kiểu FP16 nén để giảm nghẽn RAM
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_packing_layout = True          # Sắp xếp lại bộ nhớ ma trận cho SIMD NEON
        self.net.load_param(self.param_path)
        self.net.load_model(self.bin_path)

    def detect(self, frame: np.ndarray) -> list:
        img_h, img_w = frame.shape[:2]
        
        mat_in = ncnn.Mat.from_pixels_resize(
            frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
            img_w, img_h, self.target_size, self.target_size
        )
        mat_in.substract_mean_normalize([], [1.0/255.0, 1.0/255.0, 1.0/255.0])

        ex = self.net.create_extractor()
        ex.input("images", mat_in)
        ret, mat_out = ex.extract("output")
        if ret != 0:
            return []

        out_data = np.array(mat_out)
        
        # BẮT BUỘC: Đảm bảo shape luôn là (N_anchors, 144)
        # N_anchors sẽ là 2100 nếu target_size=320, hoặc 8400 nếu target_size=640
        if out_data.shape[1] != 144 and out_data.shape[0] == 144:
            out_data = out_data.T
            
        # Kiểm tra nếu số lượng anchors trong model lệch với target_size được cài đặt
        num_anchors = out_data.shape[0]
        if self.anchors.shape[0] != num_anchors:
            # Tự động tính toán lại anchors nếu có sự thay đổi kích thước đột ngột
            self.anchors, self.strides = self._generate_anchors(self.target_size)

        # 1. Tách DFL (64 kênh) và Scores (80 kênh)
        box_dfl = out_data[:, :64]
        raw_scores = out_data[:, 64:]

        # 2. Sigmoid đưa logits về xác suất [0, 1]
        classes_scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -20.0, 20.0)))
        max_scores = np.max(classes_scores, axis=1)
        
        # Tạo mask lọc trên trục 0 (số anchors)
        mask = max_scores >= self.conf_thresh
        if not np.any(mask):
            return []

        filtered_dfl = box_dfl[mask]
        confidences = max_scores[mask].tolist()
        class_ids = np.argmax(classes_scores[mask], axis=1).tolist()
        
        anchors = self.anchors[mask]
        strides = self.strides[mask]

        # 3. Giải mã DFL (Softmax + Weighted Sum)
        dfl_reshaped = filtered_dfl.reshape(-1, 4, 16)
        dfl_exp = np.exp(dfl_reshaped - np.max(dfl_reshaped, axis=2, keepdims=True))
        dfl_softmax = dfl_exp / np.sum(dfl_exp, axis=2, keepdims=True)
        dist = np.sum(dfl_softmax * self.dfl_weights, axis=2)

        # 4. Tọa độ trên ảnh target_size
        x1 = anchors[:, 0] - dist[:, 0] * strides
        y1 = anchors[:, 1] - dist[:, 1] * strides
        x2 = anchors[:, 0] + dist[:, 2] * strides
        y2 = anchors[:, 1] + dist[:, 3] * strides

        # 5. Quy đổi về tỷ lệ khung hình thật của Camera
        x_factor = img_w / self.target_size
        y_factor = img_h / self.target_size

        boxes = np.column_stack((
            x1 * x_factor, 
            y1 * y_factor, 
            (x2 - x1) * x_factor, 
            (y2 - y1) * y_factor
        )).astype(int).tolist()

        # 6. Triệt tiêu Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                b = boxes[i]
                results.append({
                    'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown",
                    'bbox': [
                        max(0, b[0]),
                        max(0, b[1]),
                        min(img_w, b[0] + b[2]),
                        min(img_h, b[1] + b[3])
                    ],
                    'confidence': float(confidences[i])
                })
        return results