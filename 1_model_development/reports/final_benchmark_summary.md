# Final 320 Benchmark Summary

Frozen from the currently available full-test accuracy reports and Raspberry Pi benchmarks. This report contains all **12 screening configurations** and the **6 main-benchmark configurations**.

## Measurement Scope

- Accuracy: 320px `rod_clean_v1/test`, 1,615 images. Dangerous recall is the macro mean across the nine configured critical classes; critical minimum is the lowest recall among them.
- Pi latency: preprocessing + inference + postprocessing; decode is excluded. Screening: 30 warm-up, 300 measured runs × 3 repeats. Main: 50 warm-up, 1,000 measured runs × 3 repeats. All use 4 threads and completed with valid status.
- p50/p95/p99 are the median of the three repeat-level percentiles. RAM is the maximum sampled process RSS; CPU is system CPU mean/p95; temperature is the median start/end across repeats. Size is ONNX file size or NCNN `.param` + `.bin` size.
- Accuracy recall/FN come from full-test evaluator reports at a confidence floor of 0.001 and NMS IoU 0.7. Per-class recall is selected by the evaluator's F1 operating point. It is **not** a locked deployment threshold. The available operational-threshold report is 640px validation data for YOLO12n/YOLO26n only, so it is not applied to these 320px rows. False negatives are estimated from rounded per-class recall; false positives/image are not available in the 320 reports.

## Combined Results

Recall and critical minimum are percentages. CPU columns are system utilization percentages. Temperature is `start/end` in °C. `Main` and `Screening` rows are separate runs of the same configuration.

| Run | Model | Runtime | Danger Recall | Critical Min (class) | p50 / p95 / p99 (ms) | RAM (MB) | CPU mean / p95 | Temp start / end | Size (MB) | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Screening | YOLOv8n | ONNX FP32 | 79.017 | 67.798 (Car) | 119.399 / 163.160 / 238.041 | 114.98 | 99.62 / 100.00 | 54.04 / 55.02 | 11.595 | verified |
| Screening | YOLOv8n | ONNX INT8 | 77.585 | 66.736 (Car) | 102.273 / 129.732 / 156.725 | 106.18 | 99.80 / 100.00 | 56.97 / 57.94 | 3.186 | verified |
| Screening | YOLOv8n | NCNN FP32 | 79.017 | 67.797 (Car) | 98.360 / 132.923 / 157.806 | 143.79 | 93.73 / 95.70 | 55.99 / 55.50 | 11.524 | verified native 320 |
| Screening | YOLOv9t | ONNX FP32 | 78.870 | 68.946 (Car) | 134.078 / 159.014 / 182.775 | 113.62 | 99.76 / 100.00 | 57.94 / 58.43 | 7.803 | verified |
| Screening | YOLOv9t | ONNX INT8 | 78.493 | 66.640 (Car) | 130.296 / 153.830 / 168.304 | 116.76 | 99.80 / 100.00 | 59.89 / 59.40 | 2.651 | verified |
| Screening | YOLOv9t | NCNN FP32 | 78.870 | 68.946 (Car) | 112.052 / 205.335 / 521.549 | 135.52 | 93.82 / 95.50 | 57.45 / 56.97 | 7.605 | verified native 320 |
| Screening | YOLO12n | ONNX FP32 | 80.208 | 68.000 (Person) | 122.202 / 216.887 / 280.169 | 117.53 | 99.75 / 100.00 | 58.91 / 59.40 | 9.973 | verified |
| Screening | YOLO12n | ONNX INT8 | 79.790 | 71.563 (Person) | 118.179 / 125.089 / 146.994 | 117.29 | 99.81 / 100.00 | 58.91 / 59.89 | 3.104 | verified |
| Screening | YOLO12n | NCNN FP32 | 80.208 | 68.000 (Person) | 122.639 / 144.662 / 178.728 | 120.46 | 94.66 / 95.90 | 58.43 / 57.94 | 9.831 | verified native 320 |
| Screening | YOLO26n | ONNX FP32 | 75.748 | 62.500 (Person) | 96.068 / 104.680 / 142.615 | 114.56 | 99.81 / 100.00 | 59.40 / 59.89 | 9.249 | verified |
| Screening | YOLO26n | ONNX INT8 | 79.260 | 72.840 (Car) | 99.549 / 104.247 / 108.819 | 118.20 | 99.81 / 100.00 | 59.89 / 59.40 | 2.770 | verified |
| Screening | YOLO26n | NCNN FP32 | 80.342 | 73.205 (Car) | 84.505 / 102.259 / 193.278 | 119.08 | 93.31 / 94.30 | 57.94 / 58.43 | 9.127 | verified native 320 |
| Main | YOLOv8n | NCNN FP32 | 79.017 | 67.797 (Car) | 98.599 / 133.014 / 186.984 | 143.85 | 93.75 / 95.10 | 56.97 / 56.97 | 11.524 | verified native 320 |
| Main | YOLOv8n | ONNX INT8 | 77.585 | 66.736 (Car) | 101.919 / 123.036 / 153.244 | 107.20 | 99.74 / 100.00 | 60.37 / 60.37 | 3.186 | verified |
| Main | YOLO12n | ONNX INT8 | 79.790 | 71.563 (Person) | 118.949 / 142.969 / 171.419 | 117.90 | 99.81 / 100.00 | 60.37 / 60.37 | 3.104 | verified |
| Main | YOLO26n | ONNX FP32 | 75.748 | 62.500 (Person) | 96.700 / 123.179 / 177.451 | 114.68 | 99.74 / 100.00 | 58.91 / 59.40 | 9.249 | verified |
| Main | YOLO26n | ONNX INT8 | 79.260 | 72.840 (Car) | 100.174 / 123.857 / 156.218 | 117.66 | 99.77 / 100.00 | 59.89 / 60.37 | 2.770 | verified |
| Main | YOLO26n | NCNN FP32 | 80.342 | 73.205 (Car) | 84.289 / 107.747 / 168.306 | 119.14 | 93.41 / 94.30 | 57.45 / 57.94 | 9.127 | verified native 320 |

## Pareto Frontiers

Frontiers use measured 320 accuracy and Pi screening/main results. Accuracy Pareto maximizes dangerous recall and critical minimum. Speed Pareto minimizes p50, p95, p99, RAM, and model size. A configuration is retained if no other configuration is at least as good on every listed objective and strictly better on one.

| Frontier | Configurations |
|---|---|
| Accuracy | YOLO26n NCNN FP32 (80.342% dangerous recall; 73.205% critical minimum) |
| Speed, screening | YOLO12n ONNX INT8; YOLO26n NCNN FP32; YOLO26n ONNX FP32; YOLO26n ONNX INT8; YOLOv8n ONNX INT8; YOLOv9t ONNX INT8 |
| Speed, main | YOLO26n NCNN FP32; YOLO26n ONNX FP32; YOLO26n ONNX INT8; YOLOv8n ONNX INT8 |

## Model Selection

- **Recall-first by current full-test metric:** YOLO26n NCNN FP32. It has the highest measured dangerous macro recall (80.342%) and critical minimum (73.205%) among the 12 tested configs. Treat this as an evaluation-metric lead, not a deployment safety claim, until a 320 operational confidence threshold is selected on validation and retested on test.
- **Edge-first with direct accuracy evidence:** YOLO26n ONNX INT8. Main p50 is 100.174 ms, artifact size 2.77 MB, dangerous recall 79.260%, and critical minimum 72.840%.
- **Fastest:** YOLO26n NCNN FP32. Main p50 is 84.289 ms. Its native 320 report is available, so it is no longer proxy/pending at 320. However, its recall differs materially from same-model ONNX FP32: 80.342% vs 75.748% dangerous macro (+4.594 percentage points); critical minimum is 73.205% (Car) vs 62.500% (Person). This is a backend/evaluator output discrepancy to investigate, not proof either backend is universally safer. For YOLOv8n/YOLOv9t/YOLO12n, NCNN and ONNX FP32 dangerous macro recalls agree to the report's precision.

## Evidence and Limitations

- Native NCNN 320 was evaluated by direct `ncnn.Net()` `.param/.bin` inference on all 1,615 test images; see `scripts/evaluate_ncnn_320.py` and `reports/pipeline_run/ncnn_fp32_320_accuracy_report.json`. NCNN 640 remains proxy unless separately evaluated natively.
- The 320 verification audit predates the full-test ONNX and native NCNN 320 reports; its 320 `missing_evidence` classifications are stale relative to these artifacts.
- `operational_threshold_report.json` is 640px validation data for YOLO12n and YOLO26n only. No locked 320 threshold, 320 test-at-operating-threshold metrics, or 320 false positives/image are available in this freeze. The next safety gate is 320 validation threshold selection followed by a locked-threshold test for all candidate runtimes.
- Source data: `blind-navigation/1_model_development/reports/pipeline_run/{onnx_fp32_320_accuracy_report.json,int8_ptq_320_accuracy_report.json,ncnn_fp32_320_accuracy_report.json}` and `pi_deployment_package/benchmarks_pi/{benchmark_screening_full_320,benchmark_main_320}/`.
- Machine-readable joined data and source metadata: `reports/final_benchmark_summary.json`.
