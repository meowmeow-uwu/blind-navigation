# 2_edge_device/modules/vision_local/yolo_onnx_runner.py
from typing import List, Tuple, Dict, Any, Optional
import os
import cv2
import numpy as np
import onnxruntime as ort

from .base_detector import BaseDetector

COCO_CLASSES = [
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


def letterbox(
    img: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize và padding ảnh bảo toàn tỷ lệ khung hình (Letterbox) chuẩn cho YOLO.

    Args:
        img: Ảnh nguồn BGR (H, W, C).
        target_shape: (target_h, target_w).
        color: Màu viền padding (mặc định viền xám 114).
        auto: Làm tròn padding sao cho chia hết cho stride (tiết kiệm compute).
        scale_fill: Kéo giãn ép kích thước (bỏ letterbox).
        scale_up: Cho phép phóng to ảnh nhỏ hơn target_shape.
        stride: Bước chia hết của mạng YOLO.

    Returns:
        padded_img: Ảnh sau khi resize và padding (target_h, target_w, C).
        ratio: Tỷ lệ co giãn (new / old).
        (dw, dh): Khoảng cách padding 2 bên (width, height) tính bằng pixel.
    """
    shape = img.shape[:2]  # [height, width]
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)

    # Tính toán hệ số tỷ lệ co giãn
    r = min(target_shape[0] / shape[0], target_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    # Kích thước unpadded mới
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = target_shape[1] - new_unpad[0]
    dh = target_shape[0] - new_unpad[1]

    if auto:  # Đảm bảo padding tối thiểu chia hết cho stride
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (target_shape[1], target_shape[0])
        r = (target_shape[1] / shape[1], target_shape[0] / shape[0])

    dw /= 2.0  # Chia đều 2 bên
    dh /= 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, r, (dw, dh)


class YoloOnnxRunner(BaseDetector):
    """
    YOLOv8 Detector sử dụng ONNX Runtime.
    Tối ưu hóa:
      - Letterbox chuẩn giữ tỷ lệ khung hình.
      - Vectorized post-processing và OpenCV DNN NMS.
      - Hỗ trợ cả CPU (với đa luồng intra-op) và GPU CUDA.
    """

    def __init__(
        self,
        model_path: str,
        target_size: int = 640,
        conf_thresh: float = 0.45,
        nms_thresh: float = 0.45,
        classes: Optional[List[str]] = None,
        num_threads: int = 4,
        providers: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.target_size = target_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.classes = classes or COCO_CLASSES
        self.num_threads = num_threads
        self.providers = providers

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.input_shape: List[Any] = []
        self.output_names: List[str] = []

        self.load_model()

    def load_model(self) -> None:
        """Khởi tạo phiên thực thi ONNX Runtime với tối ưu CPU/GPU."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model ONNX không tồn tại tại: {self.model_path}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        if self.providers is None:
            # Ưu tiên CUDA nếu có, nếu không thì CPU
            if "CUDAExecutionProvider" in available_providers:
                target_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                target_providers = ["CPUExecutionProvider"]
        else:
            target_providers = [p for p in self.providers if p in available_providers]
            if not target_providers:
                target_providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=target_providers
        )

        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape

        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Letterbox, chuẩn hóa màu BGR->RGB, HWC->CHW, scale về [0, 1].
        """
        padded_img, ratio, (dw, dh) = letterbox(
            frame, target_shape=(self.target_size, self.target_size), auto=False, scale_up=True
        )

        # Chuyển BGR sang RGB
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        blob = rgb_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        # Thêm batch dimension: (1, 3, target_size, target_size)
        blob = np.expand_dims(blob, axis=0)
        return np.ascontiguousarray(blob), ratio, (dw, dh)

    def detect_raw(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Thực hiện phát hiện và trả về mảng numpy thuận tiện cho Tracker (ByteTrack / Supervision).

        Returns:
            boxes_xyxy: np.ndarray shape (N, 4) dạng [x1, y1, x2, y2]
            confidences: np.ndarray shape (N,)
            class_ids: np.ndarray shape (N,) kiểu int
        """
        img_h, img_w = frame.shape[:2]
        blob, ratio, (dw, dh) = self._preprocess(frame)

        # Chạy inference
        preds = self.session.run(self.output_names, {self.input_name: blob})[0]

        # Chuẩn hóa shape đầu ra: kỳ vọng (1, 84, 8400) -> chuyển thành (8400, 84)
        if preds.shape[1] < preds.shape[2]:
            preds = preds[0].T  # (N_candidates, 84)
        else:
            preds = preds[0]

        # Tách BBox (4 kênh: cx, cy, w, h) và Class Scores (80 kênh)
        raw_boxes = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        pred_class_ids = np.argmax(class_scores, axis=1)

        # Lọc ngưỡng Confidence
        mask = max_scores >= self.conf_thresh
        if not np.any(mask):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

        filtered_boxes = raw_boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = pred_class_ids[mask]

        # Giải mã BBox từ (cx, cy, w, h) sang (x1, y1, x2, y2) trên ảnh đã Letterbox
        cx = filtered_boxes[:, 0]
        cy = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]

        x1 = cx - (w / 2.0)
        y1 = cy - (h / 2.0)
        x2 = cx + (w / 2.0)
        y2 = cy + (h / 2.0)

        # Map ngược về tọa độ ảnh gốc (trừ padding dw, dh và chia tỷ lệ ratio)
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        # Clip tọa độ nằm gọn trong khung hình gốc
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)

        # Chuẩn bị cho cv2.dnn.NMSBoxes (yêu cầu dạng [x1, y1, width, height])
        nms_boxes = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()
        nms_scores = filtered_scores.tolist()

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=nms_scores,
            score_threshold=float(self.conf_thresh),
            nms_threshold=float(self.nms_thresh)
        )

        if len(indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

        keep_indices = np.array(indices).flatten()
        final_boxes_xyxy = np.column_stack([x1[keep_indices], y1[keep_indices], x2[keep_indices], y2[keep_indices]]).astype(np.float32)
        final_scores = filtered_scores[keep_indices].astype(np.float32)
        final_classes = filtered_classes[keep_indices].astype(int)

        return final_boxes_xyxy, final_scores, final_classes

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Thực hiện phát hiện và trả về định dạng chuẩn BaseDetector:
        [{'class_name': 'person', 'class_id': 0, 'bbox': [x1, y1, x2, y2], 'confidence': 0.85}, ...]
        """
        boxes_xyxy, confidences, class_ids = self.detect_raw(frame)
        results: List[Dict[str, Any]] = []

        for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
            c_name = self.classes[cls_id] if 0 <= cls_id < len(self.classes) else f"class_{cls_id}"
            results.append({
                "class_name": c_name,
                "class_id": int(cls_id),
                "bbox": [int(round(coord)) for coord in box],
                "confidence": float(conf)
            })

        return results
