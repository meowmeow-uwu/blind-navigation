# Native NCNN FP32 640 Accuracy Verification

- Dataset: `rod_clean_v1/test` (1,615 images)
- Runtime: direct `ncnn.Net()` `.param/.bin` inference; Ultralytics NMS/matching/AP evaluator
- Preprocessing: letterbox 640, RGB, divide by 255; confidence 0.001; NMS IoU 0.7

| Model | mAP50 | mAP50-95 | Precision | Recall | Dangerous recall | Critical min | FN | ΔmAP50 | ΔmAP50-95 | ΔRecall | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolov8n | 0.8990 | 0.7280 | 0.8848 | 0.8621 | 0.8462 | 0.7568 | 339 | -0.0000 | -0.0000 | +0.0000 | PASS |
| yolov9t | 0.9134 | 0.7526 | 0.9000 | 0.8684 | 0.8471 | 0.7027 | 344 | +0.0000 | +0.0000 | +0.0000 | PASS |
| yolov12n | 0.9071 | 0.7515 | 0.9079 | 0.8748 | 0.8607 | 0.7590 | 330 | +0.0000 | -0.0002 | +0.0000 | PASS |
| yolov26n | 0.9109 | 0.7573 | 0.8962 | 0.8843 | 0.8762 | 0.7838 | 288 | +0.0045 | +0.0042 | +0.0167 | FAIL |

Per-class recall, class counts, hashes, and exact gate results are recorded in the JSON report.
