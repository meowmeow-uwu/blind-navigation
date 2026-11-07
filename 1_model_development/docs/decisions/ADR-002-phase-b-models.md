# ADR-002: Model Selection and Qualification for Phase B Fine-Tuning

## Context
Phase A focused on acquiring official YOLO checkpoints, exporting standardized FP32 ONNX graphs (static batch=1, imgsz=320), verifying taxonomy alignment with the Real-Time Obstacle Detection (ROD) dataset, profiling host latency across 1,000 runs, and hardening the Raspberry Pi benchmark harness.

Four candidates were evaluated:
- **YOLOv8n**: Established edge baseline (raw output)
- **YOLO11n**: Recent stable architecture (raw output)
- **YOLO12n**: Attention-driven architecture targeting high recall (raw output)
- **YOLO26n**: Edge candidate with built-in end-to-end NMS (shape `[1, 300, 6]`)

## Inclusion Criteria for Phase B
To qualify for Phase B fine-tuning on the ROD dataset, a model must satisfy all of the following technical gates:
1. Official checkpoint downloaded with validated SHA-256 and metadata.
2. Standardized ONNX FP32 export succeeds without custom hacks.
3. ONNX graph passes `onnx.checker` with static `[1, 3, 320, 320]` FP32 input.
4. Output equivalence between PyTorch and ONNX verified (**pending**: `validate_export.py` not yet implemented).
5. Zero crashes across 1,000 consecutive inference cycles.
6. Evaluator compatibility with post-processing adapters.
7. Host runtime latency <= 25 ms p50 on CPU.

## Decision

| Model | Role | Phase B Status | Rationale |
|---|---|---|---|
| **YOLOv8n** | Baseline | **QUALIFIED (Retained)** | Stable reference point, mature mobile/edge deployments, 16.3 ms host p50. |
| **YOLO12n** | Recall Candidate | **QUALIFIED (Retained)** | Highest potential recall for sidewalk collision hazards; 17.3 ms host p50. |
| **YOLO26n** | Edge / NMS-Free Candidate | **QUALIFIED (Retained)** | Built-in end-to-end NMS head eliminates post-processing overhead; fastest host p50 (14.4 ms). |
| **YOLO11n** | Secondary Candidate | **QUALIFIED (Retained)** | Fully valid ONNX graph and 15.6 ms host p50; retained for ablation if compute permits. |

**No model is eliminated solely based on a 1–2 ms latency variance on the x86 host.** Final model selection will be determined after domain fine-tuning and evaluation on actual Raspberry Pi hardware.

## Status
**Provisional.** Models are retained for Phase B on the basis of export validity, ONNX graph checks, smoke tests and host latency only. Criterion 4 and real-data accuracy evaluation are still open (see `reports/phase_a_report.md`).

## Limitations and Future Work
- Accuracy evaluation ran on an 11-image synthetic placeholder, not the real ROD dataset.
- Raspberry Pi on-device benchmarking is currently **Pending** physical hardware execution (standalone harness packaged in `artifacts/pi_phase_a/`).
- Pretrained evaluation only assesses the 11 classes shared with COCO. The 14 domain-only obstacle classes (Stairs, Manhole, Electrical Pole, etc.) will be evaluated post fine-tuning in Phase B.
