# 2_edge_device/experiments/exp_onnx_tracker_pipeline.py
"""
Pipeline tích hợp đầy đủ Giai đoạn 1+2+3:
    YOLOv8 ONNX → ByteTrack → PriorityScorer → TrackStateManager → NavigationPolicy
    → Profiler → Visualizer

Cách dùng:
    python experiments/exp_onnx_tracker_pipeline.py --source 0          # Webcam
    python experiments/exp_onnx_tracker_pipeline.py --source video.mp4  # Video offline
    python experiments/exp_onnx_tracker_pipeline.py --source 0 --no-display
"""
import os
import sys
import cv2
import argparse
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, EDGE_DIR)

from modules.vision_local.yolo_onnx_runner import YoloOnnxRunner, COCO_CLASSES
from modules.vision_local.tracker import ByteTrackerWrapper
from modules.decision.priority_scorer import PriorityScorer
from modules.decision.track_state_manager import TrackStateManager
from modules.decision.navigation_policy import generate_alert
from modules.utils.profiler import PipelineProfiler
from modules.utils.stream_loader import StreamLoader
from modules.utils.visualizer import (
    draw_tracked_detections,
    draw_lane_overlay,
    draw_alert_banner,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Full Pipeline Demo (Stage 1+2+3)")
    parser.add_argument("--source", default="0")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-lane", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config_path = os.path.join(EDGE_DIR, args.config)
    cfg = load_config(config_path)

    source_raw = args.source
    source = int(source_raw) if source_raw.isdigit() else source_raw

    # ── Khởi tạo các module ────────────────────────────────────────────────────
    model_path = os.path.join(EDGE_DIR, cfg["model"]["onnx_path"])
    if not os.path.exists(model_path):
        print(f"[ERROR] Không tìm thấy model ONNX: {model_path}")
        print("        Chạy trước: python experiments/export_onnx.py")
        sys.exit(1)

    print("[INFO] Khởi tạo YOLOv8 ONNX Runner...")
    detector = YoloOnnxRunner(
        model_path=model_path,
        target_size=cfg["model"]["target_size"],
        conf_thresh=cfg["model"]["conf_threshold"],
        nms_thresh=cfg["model"]["nms_threshold"],
        num_threads=cfg["model"]["num_threads"]
    )

    print("[INFO] Khởi tạo ByteTrack Tracker...")
    tracker = ByteTrackerWrapper(
        track_thresh=cfg["tracker"]["track_threshold"],
        track_buffer=cfg["tracker"]["track_buffer"],
        match_thresh=cfg["tracker"]["match_threshold"]
    )

    weights_path = os.path.join(EDGE_DIR, "configs/class_weights.yaml")
    policy_cfg = cfg["policy"]
    w = policy_cfg["weights"]

    print("[INFO] Khởi tạo PriorityScorer và TrackStateManager...")
    scorer = PriorityScorer(
        class_weights_path=weights_path,
        w_class=w["w_class"],
        w_position=w["w_position"],
        w_size=w["w_size"]
    )
    state_mgr = TrackStateManager(
        cooldown_seconds=policy_cfg["cooldown_seconds"],
        re_alert_delta=policy_cfg["re_alert_delta_priority"]
    )
    profiler = PipelineProfiler(buffer_size=30)

    print(f"[INFO] Mở nguồn video: {source!r}")
    stream = StreamLoader(
        source=source,
        width=cfg["source"]["width"],
        height=cfg["source"]["height"],
        fps_sync=True
    )
    stream.open()
    meta = stream.meta
    print(f"[INFO] Stream: {meta.get('width')}x{meta.get('height')} @ {meta.get('fps'):.1f} FPS")

    window_name = "Pipeline Demo | 'q' = Thoát"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_alert = ""
    frame_count = 0
    print("[INFO] Đang chạy pipeline... Nhấn 'q' để thoát.\n")

    try:
        for frame in stream:
            frame_count += 1

            # ── Capture ────────────────────────────────────────────────────────
            with profiler.measure("capture"):
                img_bgr = frame.copy()
            img_h, img_w = img_bgr.shape[:2]

            # ── Inference ─────────────────────────────────────────────────────
            with profiler.measure("inference"):
                boxes, confs, cls_ids = detector.detect_raw(img_bgr)

            # ── Tracking ──────────────────────────────────────────────────────
            with profiler.measure("tracking"):
                tracked = tracker.update(boxes, confs, cls_ids, classes=COCO_CLASSES)

            # ── Policy: Priority + Cooldown + Alert ───────────────────────────
            with profiler.measure("policy"):
                # Dọn dẹp track cũ định kỳ (mỗi 150 frame)
                if frame_count % 150 == 0:
                    evicted = state_mgr.evict_stale()
                    if evicted:
                        print(f"[GC] Đã xóa {evicted} track cũ")

                top1 = scorer.top1(tracked, img_w, img_h)

                new_alert = ""
                if top1 is not None:
                    tid = top1["track_id"]
                    score = top1["priority_score"]

                    # Cập nhật trạng thái cho các track không phải Top-1
                    for obj in tracked:
                        if obj["track_id"] != tid:
                            state_mgr.touch(obj["track_id"], 0.0)

                    if state_mgr.should_alert(tid, score):
                        new_alert = generate_alert(top1, tracked, img_w)
                        if new_alert:
                            current_alert = new_alert
                            record = state_mgr.get_record(tid)
                            print(
                                f"[ALERT] #{record.alert_count} track={tid} "
                                f"score={score:.3f}: {new_alert}"
                            )

            # ── Render ────────────────────────────────────────────────────────
            with profiler.measure("render"):
                if not args.no_display:
                    if not args.no_lane:
                        draw_lane_overlay(img_bgr)
                    top1_id = top1["track_id"] if top1 else None
                    draw_tracked_detections(img_bgr, tracked, top1_track_id=top1_id)
                    draw_alert_banner(img_bgr, current_alert)
                    profiler.draw_hud(img_bgr)

            profiler.tick_frame()

            if not args.no_display:
                cv2.imshow(window_name, img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        stats = profiler.get_stats()
        print("\n" + "=" * 58)
        print("  PIPELINE PROFILE SUMMARY (rolling avg 30 frames)")
        print("=" * 58)
        print(f"  FPS (thực tế):    {stats['fps']:.1f}")
        print(f"  Capture:          {stats.get('capture', 0):.2f} ms")
        print(f"  Inference (ONNX): {stats.get('inference', 0):.2f} ms")
        print(f"  Tracking (BT):    {stats.get('tracking', 0):.2f} ms")
        print(f"  Policy:           {stats.get('policy', 0):.2f} ms")
        print(f"  Render:           {stats.get('render', 0):.2f} ms")
        print(f"  Total Pipeline:   {stats.get('total_pipeline_ms', 0):.2f} ms")
        print("=" * 58)


if __name__ == "__main__":
    main()
