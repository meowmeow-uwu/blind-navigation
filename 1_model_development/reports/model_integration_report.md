# Model Integration Report

**Purpose: team handoff.** This report assigns integration verification to the team member responsible for edge deployment and defines the evidence to return to the report requester. It is not asking the requester to run the checks. No inference, benchmark, training, model conversion, or soak test was run to produce this report. Existing Pi benchmark records are reproduced as baseline context only. Runtime integration checks below remain pending.

## Member Assignment and Handoff

**Owner:** team member assigned to edge-model integration (name to be filled in by the team).  
**Recipient:** the requester of this report.  
**Scope:** integrate and verify the three selected artifacts below without changing model weights or rerunning accuracy/training.  
**Return:** update this report's pending fields and provide the completed JSON, run logs, target device/OS/runtime versions, artifact SHA results, and any failures to the recipient.

The member should execute the integration checklist on the target environment. The requester only reviews the returned evidence; the baseline figures in this report are supplied for comparison and are not proof that integration passed.

## Selected Configurations

| Model | Runtime | Existing baseline p50 / p95 (ms) | FPS equivalent | Peak RAM (MB) | System CPU mean / p95 (%) | Temperature start / end (°C) | Stable for 30 min |
|---|---|---:|---:|---:|---:|---:|---|
| YOLO12n ONNX FP32 320 | ONNX Runtime FP32 | 122.202 / 216.887 | 8.18* | 117.53 | 99.75 / 100.00 | 58.91 / 59.40 | Not assessed; soak not run |
| YOLO26n ONNX INT8 320 | ONNX Runtime INT8 | 100.174 / 123.857 | 9.98* | 117.66 | 99.77 / 100.00 | 59.89 / 60.37 | Not assessed; soak not run |
| YOLO26n NCNN FP32 320 | NCNN FP32 | 84.289 / 107.747 | 11.86* | 119.14 | 93.41 / 94.30 | 57.45 / 57.94 | Not assessed; soak not run |

`*` FPS equivalent is calculated as `1000 / p50_ms`; it is not measured end-to-end video FPS. Existing benchmark timing covers preprocessing + inference + postprocessing and excludes image decode. YOLO12n values come from the valid 30 warm-up, 300 runs × 3 screening record. YOLO26n values come from valid 50 warm-up, 1000 runs × 3 main records. RAM is the sampled process-RSS peak; CPU and temperature are readings from those prior benchmarks, not from an integration soak.

Existing benchmark telemetry reported no throttling/undervoltage flags at sampled run boundaries for these records. This does not establish that power/thermal events will not occur during a 30-minute continuous run.

## Artifact and Runtime Status

The selected entries are present in `pi_deployment_package/weights/manifest.json`:

| Configuration | Manifest key | Artifact | Static manifest check | Runtime load in this report |
|---|---|---|---|---|
| YOLO12n ONNX FP32 320 | `yolov12n.onnx_fp32_320` | `weights/onnx/yolov12n_320_fp32.onnx` | Entry and SHA recorded; `accuracy_passed: true` | Not run |
| YOLO26n ONNX INT8 320 | `yolov26n.int8_ptq_320` | `weights/int8/yolov26n_320_int8.onnx` | Entry and SHA recorded; `accuracy_passed: true` | Not run |
| YOLO26n NCNN FP32 320 | `yolov26n.ncnn_fp32_320` | `weights/ncnn/yolov26n_ncnn_fp32/model.ncnn.param` + `.bin` | Entry and both SHAs recorded; `accuracy_passed: true` | Not run |

Manifest registration is not the same as successfully loading the model on the target device. The load checks below must still be performed in the deployed environment.

## Required Integration Checks

| Check | YOLO12n ONNX FP32 | YOLO26n ONNX INT8 | YOLO26n NCNN FP32 |
|---|---|---|---|
| Load artifact and verify SHA from manifest | Not run | Not run | Not run |
| Output fields/types at runtime | Not run | Not run | Not run |
| NaN/Inf scan at runtime | Not run | Not run | Not run |
| Bounding-box validity at runtime | Not run | Not run | Not run |
| Tracker receives detections | Not verified; tracker integration absent | Not verified; tracker integration absent | Not verified; tracker integration absent |
| Continuous 30-minute soak | Not run | Not run | Not run |
| Crash count | Not measured | Not measured | Not measured |
| Frame/decode/inference errors | Not measured | Not measured | Not measured |
| Empty-detection anomalies | Not measured | Not measured | Not measured |
| Throttling/undervoltage during soak | Not measured | Not measured | Not measured |

The `modules/tracking/` directory currently contains no tracker Python implementation. Therefore the detector-to-tracker handoff cannot be validated from this checkout. `deployment_config.yaml` and the legacy NCNN runner also describe an older `bbox` schema; the best-model inference template emits `bbox_xyxy`. The integration boundary must use one agreed schema before connecting a tracker.

## Output Contract

The detector should return a list of dictionaries:

```python
{
    "class_id": int,          # 0..24
    "class_name": str,       # exact ROD class label
    "confidence": float,     # finite and in [0, 1]
    "bbox_xyxy": [x1, y1, x2, y2],  # finite source-frame pixel coordinates
}
```

Validate every detection before passing it to a tracker:

```python
import math


def validate_detections(detections, frame_width, frame_height):
    for detection in detections:
        assert isinstance(detection["class_id"], int)
        assert 0 <= detection["class_id"] < 25
        assert isinstance(detection["class_name"], str)
        confidence = detection["confidence"]
        assert isinstance(confidence, (int, float)) and math.isfinite(confidence)
        assert 0.0 <= confidence <= 1.0

        box = detection["bbox_xyxy"]
        assert len(box) == 4 and all(math.isfinite(value) for value in box)
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= frame_width
        assert 0 <= y1 < y2 <= frame_height
```

For YOLO26n ONNX the artifact uses end-to-end NMS; confidence filtering is applied after runtime output and the graph controls NMS. For NCNN the artifact produces raw predictions and the current inference template applies CPU NMS. Do not assume thresholds or backend outputs are interchangeable.

## Work Instructions for Assigned Member

1. On the target, read each selected manifest entry and verify the ONNX SHA-256, or both NCNN param/bin SHA-256 values, before loading. Return the actual hashes and load result for each artifact.
2. Load with `2_edge_device/best_model_inference.py` and run the agreed fixed image suite. Return detections and schema, NaN/Inf scan, and bounding-box validity results per model.
3. Coordinate with the tracking owner to identify the tracker's accepted input contract. Return the tracker version/commit, adapter used, and accepted/rejected detection counts. If no tracker implementation is available, report this as a blocker rather than marking handoff passed.
4. Run each model continuously for at least 30 minutes on a representative, labeled video or camera scene. Return crash count, frame/decode/inference error count, total frames, empty-detection frames, and empty anomalies on known-positive frames. A zero-detection frame by itself is not necessarily anomalous.
5. Sample temperature and `vcgencmd get_throttled` throughout the soak, not only at start/end. Return start/end state, any event, and recovery behavior.
6. Mark `Stable = yes` only after the full soak completes with zero crashes, no unexplained frame errors, valid output schema/boxes throughout, tracker handoff confirmed, and power/thermal status acceptable. Otherwise record failure or partial duration and attach the logs.

**Return package:** completed `model_integration_report.json` and `.md`, raw run logs, target environment details, hashes, and any known issues. Do not mark an unexecuted check as passed.

## Existing Baseline Sources

- YOLO12n ONNX FP32: `pi_deployment_package/benchmarks_pi/benchmark_screening_full_320/yolov12n_onnx_fp32_320_benchmark_results.json`
- YOLO26n ONNX INT8: `pi_deployment_package/benchmarks_pi/benchmark_main_320/yolov26n_onnx_int8_320_benchmark_results.json`
- YOLO26n NCNN FP32: `pi_deployment_package/benchmarks_pi/benchmark_main_320/yolov26n_ncnn_fp32_320_benchmark_results.json`
- Artifact manifest: `pi_deployment_package/weights/manifest.json`
- Inference template: `blind-navigation/2_edge_device/best_model_inference.py`
