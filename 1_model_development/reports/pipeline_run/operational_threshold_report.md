# Operational Threshold Selection

- Dataset: `rod_clean_v1/valid` (3448 images)
- Input: 640x640; NMS IoU: 0.70
- Selection: maximum macro recall with macro precision >= 0.70; ties prefer precision, then higher confidence.
- Metrics use IoU 0.50 for TP/FP/FN matching; dangerous recall is macro recall over configured critical classes.

| Model | Threshold | Precision | Recall | Dangerous Recall | Critical Min |
|---|---:|---:|---:|---:|---:|
| yolov12n | 0.10 | 0.7591 | 0.9175 | 0.9004 | 0.7195 |
| yolov26n | 0.05 | 0.7107 | 0.9334 | 0.9215 | 0.7805 |

## Selected operating metrics

| Model | False negatives | False positives/image |
|---|---:|---:|
| yolov12n | 550 | 0.6293 |
| yolov26n | 463 | 0.9635 |

Full metrics for all thresholds, including per-class values, are in the JSON report.
