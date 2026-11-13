"""
run_benchmark.py - Thực thi thực nghiệm có kiểm soát cho 4 cấu hình Tracking (T4: PBL6)
========================================================================================
Chạy lần lượt 4 cấu hình (`raw_yolo`, `sort`, `deepsort`, `edge_bytetrack`) trên CÙNG
tập video kịch bản chuẩn, đi qua YOLOv8 NCNN -> Tracker -> SensorFusionEngine -> Audio Logic,
và xuất log chi tiết từng frame/sự kiện ra CSV để đánh giá định lượng.

Cột log xuất ra (đúng chuẩn đặc tả):
  timestamp, frame_id, scenario_id, tracker_name, reid_info, bbox, class_name,
  confidence, track_id, det_time_ms, track_time_ms, processing_time_ms,
  distance_estimate, audio_event, audio_track_id

Ví dụ chạy trên Raspberry Pi 4 hoặc Mac:
  python run_benchmark.py --videos-dir data/videos --out-dir results/logs
"""

import os
import sys
import csv
import json
import time
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parents[2]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
from core.fusion_engine import SensorFusionEngine
from experiments.tracking_benchmark.trackers import create_tracker

TRACKER_CONFIGS = ["raw_yolo", "sort", "deepsort", "edge_bytetrack"]


class BenchmarkAudioSimulator:
    """
    Mô phỏng chính xác logic điều phối âm thanh của `main.py` + `AudioManager`
    dựa trên đồng hồ thời gian thực của video (`video_time_s = frame_id / video_fps`),
    đảm bảo tính nhất quán khi replay nhanh/chậm trên các máy tính khác nhau.
    """

    def __init__(self, guidance_cooldown_s: float = 2.5, reannounce_dist_delta: float = 0.8, reannounce_cooldown_s: float = 3.0):
        self.guidance_cooldown_s = guidance_cooldown_s
        self.reannounce_dist_delta = reannounce_dist_delta
        self.reannounce_cooldown_s = reannounce_cooldown_s
        self.last_voice_time_s: float = -999.0
        self.track_announced_state: Dict[int, dict] = {}
        self.announced_track_ids = set()

    def decide_frame_audio(self, fused_results: list, raw_us_d: Optional[float], global_emergency: bool, video_time_s: float):
        """
        Trả về dict: {track_id: audio_event_str}, spoken_track_id
        Các giá trị audio_event_str:
          - 'emergency': Khẩn cấp < 1.0m hoặc TTC < 1.5s
          - 'guidance': Phát âm thanh dẫn hướng (lần đầu hoặc tái cảnh báo hợp lệ)
          - 'suppressed_cooldown': Track muốn đọc nhưng bị khóa cooldown 2.5s
          - 'suppressed_silent': Track đã đọc trước đó -> Bám vết ngầm im lặng
          - 'none': Ngoài vùng dẫn hướng 1.0m - 3.0m
        """
        events_by_track: Dict[int, str] = {}
        spoken_track_id = ""

        if global_emergency or (raw_us_d is not None and raw_us_d < 1.0):
            # Vùng khẩn cấp chiếm quyền tức thì
            top_id = fused_results[0].track_id if fused_results else -1
            events_by_track[top_id] = "emergency"
            return events_by_track, top_id

        guidance_dispatched = False
        for obs in fused_results:
            tid = obs.track_id
            if tid == -1 or not (1.0 <= obs.distance_m <= 3.0):
                events_by_track[tid] = "none"
                continue

            if guidance_dispatched:
                events_by_track[tid] = "none"
                continue

            state = self.track_announced_state.get(tid)
            wants_speak = False
            force = False
            if state is None:
                wants_speak = True
            else:
                # Ngoại lệ an toàn khi vật tiến gần >= 0.8m và sau >= 3.0s
                if (video_time_s - state["time_s"] >= self.reannounce_cooldown_s) and (
                    state["dist_m"] - obs.distance_m >= self.reannounce_dist_delta
                ):
                    wants_speak = True
                    force = True

            if not wants_speak:
                events_by_track[tid] = "suppressed_silent"
                continue

            if (tid in self.announced_track_ids and not force):
                events_by_track[tid] = "suppressed_silent"
                continue

            if (video_time_s - self.last_voice_time_s) < self.guidance_cooldown_s:
                events_by_track[tid] = "suppressed_cooldown"
                continue

            # Phát âm thanh dẫn hướng thành công
            self.last_voice_time_s = video_time_s
            self.announced_track_ids.add(tid)
            self.track_announced_state[tid] = {"time_s": video_time_s, "dist_m": obs.distance_m}
            events_by_track[tid] = "guidance"
            spoken_track_id = tid
            guidance_dispatched = True

        return events_by_track, spoken_track_id


def load_edge_config() -> dict:
    cfg_path = EDGE_ROOT / "configs/config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_ultrasonic_csv(us_csv_path: Path) -> Dict[int, Optional[float]]:
    readings: Dict[int, Optional[float]] = {}
    if not us_csv_path.exists():
        return readings
    with open(us_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            val = row.get("us_distance_m", "").strip()
            readings[fid] = float(val) if val else None
    return readings


def run_single_config_on_video(
    scenario_id: str,
    video_path: Path,
    us_csv_path: Path,
    tracker_name: str,
    detector: Optional[YoloNcnnRunner],
    cached_detections: Optional[Dict[int, dict]],
    config: dict,
    out_csv_path: Path,
    onnx_reid_path: Optional[str] = None,
):
    """Chạy đúng 1 cấu hình tracker trên 1 video kịch bản và lưu log CSV từng frame/event."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 1e-3 or video_fps > 120:
        video_fps = 15.0

    us_readings = load_ultrasonic_csv(us_csv_path)
    tracker = create_tracker(tracker_name, onnx_reid_path=onnx_reid_path)
    tracker.reset()

    fusion_engine = SensorFusionEngine(config)
    audio_sim = BenchmarkAudioSimulator(guidance_cooldown_s=2.5)

    rows = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        video_time_s = frame_id / video_fps
        raw_us_d = us_readings.get(frame_id, 2.0)  # Mặc định 2.0m nếu không có file siêu âm

        # 1. Phát hiện vật thể (hoặc dùng detections đã cache cho cùng video)
        if cached_detections is not None and frame_id in cached_detections:
            dets = cached_detections[frame_id]["detections"]
            det_time_ms = float(cached_detections[frame_id]["det_time_ms"])
        else:
            t_det_start = time.perf_counter()
            dets = detector.detect(frame) if detector is not None else []
            det_time_ms = (time.perf_counter() - t_det_start) * 1000.0

        # 2. Thực thi thuật toán Tracking (bao gồm trích xuất đặc trưng Re-ID nếu là DeepSORT)
        t_trk_start = time.perf_counter()
        tracked_items = tracker.update(dets, frame=frame)
        track_time_ms = (time.perf_counter() - t_trk_start) * 1000.0

        # 3. Hợp nhất đa phương thức & Đánh giá rủi ro
        t_fus_start = time.perf_counter()
        fused_results, global_emergency = fusion_engine.fuse(tracked_items, raw_us_d)
        fus_time_ms = (time.perf_counter() - t_fus_start) * 1000.0

        total_proc_ms = det_time_ms + track_time_ms + fus_time_ms

        # 4. Điều phối sự kiện âm thanh
        events_by_track, spoken_tid = audio_sim.decide_frame_audio(
            fused_results, raw_us_d, global_emergency, video_time_s
        )

        # Ghi log bản ghi từng đối tượng (hoặc 1 bản ghi trống nếu frame không có đối tượng)
        if not fused_results:
            rows.append({
                "timestamp": round(video_time_s, 4),
                "frame_id": frame_id,
                "scenario_id": scenario_id,
                "tracker_name": tracker_name,
                "reid_info": tracker.reid_info,
                "bbox": "",
                "class_name": "",
                "confidence": "",
                "track_id": -1,
                "det_time_ms": round(det_time_ms, 3),
                "track_time_ms": round(track_time_ms, 3),
                "processing_time_ms": round(total_proc_ms, 3),
                "distance_estimate": "",
                "audio_event": events_by_track.get(-1, "none"),
                "audio_track_id": spoken_tid if spoken_tid != "" else "",
            })
        else:
            for obs in fused_results:
                bbox_str = f"{obs.bbox[0]},{obs.bbox[1]},{obs.bbox[2]},{obs.bbox[3]}"
                rows.append({
                    "timestamp": round(video_time_s, 4),
                    "frame_id": frame_id,
                    "scenario_id": scenario_id,
                    "tracker_name": tracker_name,
                    "reid_info": tracker.reid_info,
                    "bbox": bbox_str,
                    "class_name": obs.class_name,
                    "confidence": round(obs.confidence, 4),
                    "track_id": obs.track_id,
                    "det_time_ms": round(det_time_ms, 3),
                    "track_time_ms": round(track_time_ms, 3),
                    "processing_time_ms": round(total_proc_ms, 3),
                    "distance_estimate": round(obs.distance_m, 3),
                    "audio_event": events_by_track.get(obs.track_id, "none"),
                    "audio_track_id": spoken_tid if spoken_tid != "" else "",
                })

        frame_id += 1

    cap.release()

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "frame_id", "scenario_id", "tracker_name", "reid_info",
        "bbox", "class_name", "confidence", "track_id",
        "det_time_ms", "track_time_ms", "processing_time_ms",
        "distance_estimate", "audio_event", "audio_track_id"
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_ms = np.mean([r["processing_time_ms"] for r in rows]) if rows else 0.0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    print(f"   ✓ [{tracker_name:<14}] {scenario_id:<22} | {frame_id} frames | Avg: {avg_ms:.2f}ms ({avg_fps:.1f} FPS) | Re-ID: {tracker.reid_info}")


def build_or_load_detection_cache(video_path: Path, cache_path: Path, detector: YoloNcnnRunner) -> Dict[int, dict]:
    """Chạy YOLO 1 lần và cache detections + thời gian det_time_ms để replay siêu nhanh nếu muốn."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}

    cap = cv2.VideoCapture(str(video_path))
    cache: Dict[int, dict] = {}
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cache[fid] = {"det_time_ms": round(dt_ms, 3), "detections": dets}
        fid += 1
    cap.release()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return cache


def main():
    parser = argparse.ArgumentParser(description="Chạy Benchmark đối sánh 4 cấu hình Tracking trên cùng tập video chuẩn")
    parser.add_argument("--videos-dir", type=str, default=str(Path(__file__).resolve().parent / "data/videos"))
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "results/logs"))
    parser.add_argument("--model", type=str, choices=["yolov8n", "yolov8s"], default="yolov8n", help="Mô hình YOLO NCNN")
    parser.add_argument("--trackers", nargs="+", default=TRACKER_CONFIGS, choices=TRACKER_CONFIGS)
    parser.add_argument("--scenarios", nargs="+", default=None, help="Danh sách kịch bản (mặc định chạy tất cả .mp4 trong videos-dir)")
    parser.add_argument("--cache-dets", action="store_true", help="Cache detection YOLO qua các lần replay (tiết kiệm thời gian)")
    parser.add_argument("--onnx-reid", type=str, default=None, help="Đường dẫn file .onnx mô hình Re-ID thật cho DeepSORT")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)

    param_path = EDGE_ROOT / f"modules/vision_local/models/{args.model}.param"
    bin_path = EDGE_ROOT / f"modules/vision_local/models/{args.model}.bin"
    if not param_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình YOLO NCNN tại {param_path}")

    print(f"🔧 Nạp mô hình {args.model.upper()} NCNN (conf_thresh=0.20)...")
    detector = YoloNcnnRunner(str(param_path), str(bin_path), target_size=320, conf_thresh=0.20)
    config = load_edge_config()

    video_files = sorted(videos_dir.glob("*.mp4"))
    if args.scenarios:
        video_files = [v for v in video_files if v.stem in args.scenarios]

    if not video_files:
        print(f"❌ Không tìm thấy video .mp4 nào trong {videos_dir}")
        return

    print(f"🚀 Bắt đầu Benchmark trên {len(video_files)} kịch bản x {len(args.trackers)} cấu hình tracker:\n")

    for vpath in video_files:
        scen_id = vpath.stem
        us_csv = videos_dir / f"{scen_id}_ultrasonic.csv"
        cache_data = None
        if args.cache_dets:
            cache_file = videos_dir / "cache" / f"{scen_id}_{args.model}_dets.json"
            cache_data = build_or_load_detection_cache(vpath, cache_file, detector)

        print(f"📹 Kịch bản: {scen_id} ({vpath.name})")
        for trk_name in args.trackers:
            out_csv = out_dir / f"{scen_id}__{trk_name}.csv"
            run_single_config_on_video(
                scenario_id=scen_id,
                video_path=vpath,
                us_csv_path=us_csv,
                tracker_name=trk_name,
                detector=detector if cache_data is None else None,
                cached_detections=cache_data,
                config=config,
                out_csv_path=out_csv,
                onnx_reid_path=args.onnx_reid,
            )

    print(f"\n✅ Đã xuất toàn bộ log CSV ra: {out_dir}")


if __name__ == "__main__":
    main()

