# 320 Recall and Raspberry Pi Pareto Summary

## Scope and method

This joins dangerous/critical-class recall for the 320 profile with Pi screening measurements. Pi latency is total model p50 (preprocess + inference + postprocess, excluding video/image decode), median across three repeats of 300 runs. RAM is the maximum sampled process RSS across repeats. Artifact size is ONNX file size, or NCNN `.param` + `.bin` size.

Accuracy source: `reports/pipeline_run/profile_comparison_320.json`, generated 2026-11-11 16:54Z on `rod_clean_v1/test` (1,615 images). Its dangerous macro is the mean recall over the nine `critical` classes in `configs/danger_classes.yaml`; the critical minimum is included to expose a weak class.

The 320 verification audit was generated earlier (2026-11-11 16:05 +07) and marked 320 evidence missing. The later full-test report supersedes that finding for ONNX FP32 and INT8, but the audit has not been regenerated. The report JSON does not persist the actual device: `run_640_profile.py` selects CUDA device 0 when available for FP32, while its INT8 validation explicitly uses CPU. It evaluates at `conf=0.001, iou=0.7` and uses Ultralytics' selected-F1 per-class recall, so these values are not a direct measurement at the Pi operating thresholds (`conf=0.25, iou=0.45`). Validate at operating thresholds before making a safety claim.

## Joined results

| Configuration | Dangerous macro recall (%) | Min critical recall (%) | Pi total p50 (ms) | Peak RAM (MB) | Model size (MB) | Accuracy evidence |
|---|---:|---:|---:|---:|---:|---|
| YOLO12n ONNX FP32 | 80.208 | 67.999 (Person) | 122.20 | 117.5 | 9.97 | Full-test ONNX; device not recorded |
| YOLO12n ONNX INT8 | 79.790 | 71.563 (Person) | 118.18 | 117.3 | 3.10 | Full-test ONNX INT8; CPU validation |
| YOLO12n NCNN FP32 | 80.208* | 67.999* | 122.64 | 120.5 | 9.83 | `proxy/pending` |
| YOLO26n ONNX FP32 | 75.748 | 62.500 (Person) | 96.07 | 114.6 | 9.25 | Full-test ONNX; device not recorded |
| YOLO26n ONNX INT8 | 79.260 | 72.840 (Car) | 99.55 | 118.2 | 2.77 | Full-test ONNX INT8; CPU validation |
| YOLO26n NCNN FP32 | 75.748* | 62.500* | 84.50 | 119.1 | 9.13 | `proxy/pending` |
| YOLOv8n ONNX FP32 | 79.017 | 67.797 (Car) | 119.40 | 115.0 | 11.60 | Full-test ONNX; device not recorded |
| YOLOv8n ONNX INT8 | 77.585 | 66.736 (Car) | 102.27 | 106.2 | 3.19 | Full-test ONNX INT8; CPU validation |
| YOLOv8n NCNN FP32 | 79.017* | 67.797* | 98.36 | 143.8 | 11.52 | `proxy/pending` |
| YOLOv9t ONNX FP32 | 78.870 | 68.946 (Car) | 134.08 | 113.6 | 7.80 | Full-test ONNX; device not recorded |
| YOLOv9t ONNX INT8 | 78.493 | 66.640 (Car) | 130.30 | 116.8 | 2.65 | Full-test ONNX INT8; CPU validation |
| YOLOv9t NCNN FP32 | 78.870* | 68.946* | 112.05 | 135.5 | 7.61 | `proxy/pending` |

`*` NCNN recall is copied as a proxy from the corresponding ONNX FP32 model. No native 320 full-test evaluation from the NCNN engine was found. Do not treat these values as equivalent accuracy evidence or use them to clear NCNN for safety.

## Pareto and picks

For the eight ONNX configurations with direct full-test accuracy evidence, the four-objective Pareto dimensions are dangerous macro recall (maximize), Pi total p50 (minimize), peak RAM (minimize), and model size (minimize). All eight remain non-dominated: each trades at least one of recall, latency, memory, or footprint. NCNN is excluded from the safety Pareto because its recall is proxy/pending.

- **Recall priority: YOLO12n ONNX FP32.** Highest dangerous macro recall (80.208%). Check the weaker critical-class result: Person recall is 67.999%. Its p50 is 122.20 ms and its artifact is 9.97 MB.
- **Speed/footprint with direct accuracy evidence: YOLO26n ONNX INT8.** Total p50 is 99.55 ms, model size 2.77 MB, dangerous macro recall 79.260%, and minimum critical recall 72.840% (Car). It is a compact, fast compromise, not the absolute fastest ONNX configuration.
- **Performance-only leader: YOLO26n NCNN FP32** at 84.50 ms p50. Accuracy is `proxy/pending`; it is not a safety-cleared recommendation until native NCNN full-test evaluation is completed.

The Pi screening values come from `pi_deployment_package/benchmarks_pi/benchmark_screening_full_320/`; the accuracy source is the report named above. Re-run full-test accuracy using the deployment thresholds and native NCNN inference before final safety selection.
