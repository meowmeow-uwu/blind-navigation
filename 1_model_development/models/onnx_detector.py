"""ONNX Runtime detector with preprocessing and postprocessing."""

import cv2
import numpy as np
import onnxruntime as ort

from .base_detector import Detection


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (320, 320),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with aspect ratio preserved using letterboxing.
    
    Args:
        image: Input image (H, W, C), BGR
        new_shape: Target (height, width)
        color: Padding color
    
    Returns:
        resized: Letterboxed image
        ratio: Scale ratio
        (dw, dh): Padding width and height
    """
    shape = image.shape[:2]  # current (H, W)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W, H)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    
    return image, r, (int(dw), int(dh))


def preprocess(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Preprocess image for ONNX inference.
    
    Args:
        image: Input BGR image (H, W, C)
        input_size: Model input (H, W)
    
    Returns:
        tensor: Preprocessed tensor (1, 3, H, W) float32
        ratio: Scale ratio
        pad: (dw, dh) padding
    """
    # Letterbox
    img_letterbox, ratio, pad = letterbox(image, input_size)
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    
    # HWC to CHW
    img_chw = img_rgb.transpose(2, 0, 1)
    
    # Normalize to [0, 1]
    img_float = img_chw.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_float, axis=0)
    
    return img_batch, ratio, pad


def scale_boxes(boxes: np.ndarray, ratio: float, pad: tuple[int, int], orig_shape: tuple[int, int]) -> np.ndarray:
    """Scale boxes from letterboxed image back to original image.
    
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] in letterboxed coordinates
        ratio: Scale ratio from letterbox
        pad: (dw, dh) padding from letterbox
        orig_shape: Original image (H, W)
    
    Returns:
        boxes: (N, 4) array scaled to original coordinates
    """
    boxes = boxes.astype(np.float32).copy()
    
    # Remove padding
    boxes[:, [0, 2]] -= pad[0]  # x
    boxes[:, [1, 3]] -= pad[1]  # y
    
    # Scale
    boxes /= ratio
    
    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])  # width
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])  # height
    
    return boxes


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2].
    
    Args:
        boxes: (N, 4) array in xywh format
    
    Returns:
        boxes: (N, 4) array in xyxy format
    """
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) array [x1, y1, x2, y2]
        scores: (N,) array of scores
        iou_threshold: IoU threshold
    
    Returns:
        keep: Indices of boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


class ONNXDetector:
    """ONNX Runtime detector for YOLO models."""
    
    def __init__(
        self,
        onnx_path: str,
        class_names: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ):
        """Initialize ONNX detector.
        
        Args:
            onnx_path: Path to ONNX model
            class_names: List of class names (default: COCO 80 classes)
        """
        self.onnx_path = onnx_path
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        
        # Get input metadata
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # Input shape: [batch, channels, height, width]
        self.batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        # Get output metadata
        self.output_name = self.session.get_outputs()[0].name
        output_shape = self.session.get_outputs()[0].shape
        
        # Detect output format
        # Format 1: [1, num_classes+4, num_boxes] - raw YOLO output (e.g. [1, 84, 2100])
        # Format 2: [1, max_dets, 6] - end-to-end NMS (e.g. [1, 300, 6])
        if len(output_shape) == 3:
            if output_shape[2] == 6:
                self.output_format = "end2end"
            else:
                self.output_format = "raw"
        else:
            self.output_format = "raw"
        
        # Class names (COCO 80 classes by default)
        if class_names is None:
            self.class_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
            ]
        else:
            self.class_names = class_names
    
    def predict(self, image_path: str, conf: float = 0.25, iou: float = 0.45) -> list[Detection]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image
            conf: Confidence threshold
            iou: NMS IoU threshold
        
        Returns:
            List of Detection objects
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        orig_shape = image.shape[:2]  # (H, W)
        
        # Preprocess
        input_tensor, ratio, pad = preprocess(image, (self.input_height, self.input_width))
        
        # Inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )
        
        output = outputs[0]
        
        # Postprocess based on format
        if self.output_format == "end2end":
            detections = self._postprocess_end2end(output, ratio, pad, orig_shape, conf)
        else:
            detections = self._postprocess_raw(output, ratio, pad, orig_shape, conf, iou)
        
        return detections
    
    def _postprocess_end2end(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
        conf_threshold: float,
    ) -> list[Detection]:
        """Postprocess end-to-end NMS output.
        
        Args:
            output: End-to-end NMS output [1, max_dets, 6]
                    Format: [x1, y1, x2, y2, score, class_id]
            ratio: Scale ratio
            pad: Padding (dw, dh)
            orig_shape: Original image shape (H, W)
            conf_threshold: Confidence threshold
        
        Returns:
            List of Detection objects
        """
        # Remove batch dimension
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]  # (max_dets, 6)
        
        detections = []
        
        for det in output:
            x1, y1, x2, y2, score, class_id = det
            
            # Filter by confidence
            if score < conf_threshold:
                continue
            
            # Validate
            if np.isnan([x1, y1, x2, y2, score, class_id]).any():
                continue
            if np.isinf([x1, y1, x2, y2, score, class_id]).any():
                continue
            if not (0.0 <= score <= 1.0):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Scale from letterboxed to original
            boxes = np.array([[x1, y1, x2, y2]])
            boxes_scaled = scale_boxes(boxes, ratio, pad, orig_shape)[0]
            x1, y1, x2, y2 = boxes_scaled
            
            # Get class name
            class_id_int = int(class_id)
            if 0 <= class_id_int < len(self.class_names):
                class_name = self.class_names[class_id_int]
            else:
                class_name = f"class_{class_id_int}"
            
            detections.append(Detection(
                class_id=class_id_int,
                class_name=class_name,
                confidence=float(score),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            ))
        
        return detections
    
    def _postprocess_raw(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        """Postprocess raw YOLO output to Detection objects.
        
        Args:
            output: Model output tensor
            ratio: Scale ratio
            pad: Padding (dw, dh)
            orig_shape: Original image shape (H, W)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        
        Returns:
            List of Detection objects
        """
        # Handle different output formats
        # Common formats:
        # - (1, 84, 8400): YOLOv8 format [cx, cy, w, h, conf_class0, conf_class1, ...]
        # - (1, 8400, 84): Transposed
        # - End-to-end NMS models may have different output
        
        # Remove batch dimension
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        
        # Ensure shape is (num_boxes, num_attrs)
        if output.shape[0] < output.shape[1]:
            output = output.T  # Transpose to (num_boxes, num_attrs)
        
        # Expected: (num_boxes, 4 + num_classes)
        num_boxes, num_attrs = output.shape
        num_classes = num_attrs - 4
        
        # Extract boxes and class scores
        boxes_xywh = output[:, :4]  # (num_boxes, 4) [cx, cy, w, h]
        class_scores = output[:, 4:]  # (num_boxes, num_classes)
        
        # Get max class score and class ID for each box
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        
        # Filter by confidence
        mask = max_scores >= conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes_xywh) == 0:
            return []
        
        # Convert to xyxy
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        
        # Scale to original image
        boxes_scaled = scale_boxes(boxes_xyxy, ratio, pad, orig_shape)
        
        # NMS per class
        detections = []
        unique_classes = np.unique(class_ids)
        
        for cls_id in unique_classes:
            cls_mask = class_ids == cls_id
            cls_boxes = boxes_scaled[cls_mask]
            cls_scores = max_scores[cls_mask]
            
            # NMS
            keep_indices = nms(cls_boxes, cls_scores, iou_threshold)
            
            for idx in keep_indices:
                x1, y1, x2, y2 = cls_boxes[idx]
                confidence = float(cls_scores[idx])
                class_id = int(cls_id)
                
                # Validate
                if not (0.0 <= confidence <= 1.0):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                if np.isnan([x1, y1, x2, y2, confidence]).any():
                    continue
                if np.isinf([x1, y1, x2, y2, confidence]).any():
                    continue
                
                # Get class name
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                ))
        
        return detections
