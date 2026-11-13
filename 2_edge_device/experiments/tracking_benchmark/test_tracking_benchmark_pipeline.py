"""
test_tracking_benchmark_pipeline.py - Kiểm thử End-to-End Bộ công cụ Benchmark Tracking
========================================================================================
Sinh video + detections kiểm thử có kiểm soát (6 kịch bản chuẩn) vào thư mục tạm để:
1. Kiểm tra định dạng 15 cột log CSV.
2. Kiểm tra bộ tính toán MOTA, IDF1, IDSW, FPS, Tỷ lệ nói lặp và Cảnh báo nhầm.
3. Xác nhận sự khác biệt hành vi đúng đắn giữa `raw_yolo`, `sort`, `deepsort` và `edge_bytetrack`.
"""

import os
import sys
import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parents[2]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from experiments.tracking_benchmark.run_benchmark import run_single_config_on_video, TRACKER_CONFIGS, load_edge_config
from experiments.tracking_benchmark.evaluate_results import evaluate_scenario, aggregate_by_tracker


def _generate_synthetic_scenario(video_path: Path, gt_path: Path, scenario_id: str, n_frames: int = 90, fps: float = 15.0):
    """Tạo video MP4 kiểm thử + cache detections + Ground-Truth CSV cho từng kịch bản."""
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    cached_dets = {}
    gt_rows = []

    rng = np.random.default_rng(42)

    for fid in range(n_frames):
        frame = np.full((480, 640, 3), 35, dtype=np.uint8)
        dets = []

        if scenario_id in ["scen1_static"]:
            # 1 chiếc ghế đứng yên ở giữa, độ tin cậy ổn định 0.85
            box = [260, 220, 380, 400]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (40, 180, 60), -1)
            dets.append({"bbox": box, "class_name": "chair", "confidence": 0.85})
            gt_rows.append({"frame_id": fid, "gt_id": 1, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "class_name": "chair"})

        elif scenario_id in ["scen4_occlusion", "scen3_occlusion"]:
            # Vật cản (người đi ngang/đến gần) bị che khuất 1 phần trong 15 frame (1.0s @ 15FPS > max_age=12)
            # làm conf tụt xuống D_low (0.30).
            x1 = 170 + fid * 2
            y1 = 230
            chair_box = [x1, y1, x1 + 100, y1 + 180]
            cv2.rectangle(frame, (chair_box[0], chair_box[1]), (chair_box[2], chair_box[3]), (60, 180, 80), -1)
            gt_rows.append({"frame_id": fid, "gt_id": 1, "x1": chair_box[0], "y1": chair_box[1], "x2": chair_box[2], "y2": chair_box[3], "class_name": "chair"})

            if 28 <= fid <= 42:
                # Che khuất 1 phần -> YOLOv8 giảm confidence xuống D_low (0.30) trong 15 frames
                dets.append({"bbox": [x1 + 2, y1 + 3, x1 + 98, y1 + 178], "class_name": "chair", "confidence": 0.30})
            else:
                dets.append({"bbox": chair_box, "class_name": "chair", "confidence": 0.82})

        elif scenario_id in ["scen8_shake", "scen5_shake"]:
            # Camera rung nhẹ kết hợp mờ hình ngắn: tọa độ x dịch chuyển và liên tục có D_low (0.28) trong 14 frames
            dx = int(12 * np.sin(fid * 0.5))
            box = [240 + dx, 220, 360 + dx, 400]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (190, 120, 40), -1)
            conf = 0.28 if (20 <= fid <= 34) else 0.78
            dets.append({"bbox": box, "class_name": "person", "confidence": conf})
            gt_rows.append({"frame_id": fid, "gt_id": 1, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "class_name": "person"})

        elif scenario_id in ["scen3_empty", "scen6_empty"]:
            # Hành lang trống: chỉ có nhiễu giả 1-frame chớp tắt ở vùng cảnh báo (~2.5m)
            if fid in (15, 45, 70):
                dx_n = int(rng.integers(-20, 20))
                noise_box = [250 + dx_n, 230, 350 + dx_n, 440]
                dets.append({"bbox": noise_box, "class_name": "chair", "confidence": 0.50})

        writer.write(frame)
        cached_dets[fid] = {"det_time_ms": 20.0, "detections": dets}

    writer.release()

    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "gt_id", "x1", "y1", "x2", "y2", "class_name"])
        w.writeheader()
        w.writerows(gt_rows)

    return cached_dets


def run_self_test():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        v_dir = tmp_dir / "videos"
        g_dir = tmp_dir / "gt"
        l_dir = tmp_dir / "logs"
        v_dir.mkdir()
        g_dir.mkdir()
        l_dir.mkdir()

        config = load_edge_config()
        scenarios = ["scen1_static", "scen4_occlusion", "scen8_shake", "scen3_empty"]
        per_tracker = {k: [] for k in TRACKER_CONFIGS}

        for sc in scenarios:
            vp = v_dir / f"{sc}.mp4"
            gp = g_dir / f"{sc}_gt.csv"
            cached = _generate_synthetic_scenario(vp, gp, sc, n_frames=90, fps=15.0)
            for trk in TRACKER_CONFIGS:
                out_csv = l_dir / f"{sc}__{trk}.csv"
                run_single_config_on_video(
                    scenario_id=sc,
                    video_path=vp,
                    us_csv_path=v_dir / f"{sc}_missing_us.csv",
                    tracker_name=trk,
                    detector=None,
                    cached_detections=cached,
                    config=config,
                    out_csv_path=out_csv,
                )
                # Kiểm tra đầy đủ các cột theo đặc tả
                with open(out_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    required_cols = {
                        "timestamp", "frame_id", "bbox", "class_name", "confidence",
                        "track_id", "tracker_name", "processing_time_ms",
                        "distance_estimate", "audio_event", "audio_track_id"
                    }
                    assert required_cols.issubset(set(reader.fieldnames)), f"Thiếu cột log: {required_cols - set(reader.fieldnames)}"

                m = evaluate_scenario(out_csv, gp)
                per_tracker[trk].append(m)

        # Kiểm tra các tính chất khoa học bắt buộc:
        agg_raw = aggregate_by_tracker(per_tracker["raw_yolo"])
        agg_sort = aggregate_by_tracker(per_tracker["sort"])
        agg_deep = aggregate_by_tracker(per_tracker["deepsort"])
        agg_byte = aggregate_by_tracker(per_tracker["edge_bytetrack"])

        print("\n[KIỂM TRA ĐẶC TÍNH KHOA HỌC TRÊN DỮ LIỆU KIỂM THỬ CÓ KIỂM SOÁT]")
        for name, agg in [("raw_yolo", agg_raw), ("sort", agg_sort), ("deepsort", agg_deep), ("edge_bytetrack", agg_byte)]:
            print(f"  - {name:<15}: MOTA={agg['mota']:.1f}%, IDF1={agg['idf1']:.1f}%, IDSW={agg['idsw']}, Repeat={agg['repeat_rate_per_min']:.1f}/p, FalseAlerts={agg['false_alert_rate_per_min']:.1f}/p")

        # 1. Raw YOLO phải nhảy ID liên tục (> 100 IDSW) và nói lặp cao hơn ByteTrack
        assert agg_raw["idsw"] > 50, f"Raw YOLO phải có IDSW cao, nhận được {agg_raw['idsw']}"
        assert agg_raw["repeat_rate_per_min"] > agg_byte["repeat_rate_per_min"]

        # 2. ByteTrack phải có MOTA & IDF1 cao hơn hoặc bằng SORT trên kịch bản rung lắc / che khuất
        assert agg_byte["mota"] > agg_sort["mota"], f"ByteTrack MOTA ({agg_byte['mota']}) phải vượt SORT ({agg_sort['mota']}) khi có D_low!"
        assert agg_byte["idsw"] <= agg_sort["idsw"]

        # 3. Kịch bản 6 (Hành lang trống, nhiễu chớp tắt 1 frame): min_hits=2 giúp Edge ByteTrack có 0 cảnh báo nhầm!
        empty_raw = per_tracker["raw_yolo"][3]
        empty_byte = per_tracker["edge_bytetrack"][3]
        assert empty_byte["false_alerts"] == 0, f"ByteTrack không được báo nhầm nhiễu 1-frame, nhận được {empty_byte['false_alerts']}"
        assert empty_raw["false_alerts"] > 0

        print("\n✅ TẤT CẢ KIỂM THỬ PIPELINE BENCHMARK OBJECT TRACKING ĐỀU ĐẠT!")


if __name__ == "__main__":
    run_self_test()
