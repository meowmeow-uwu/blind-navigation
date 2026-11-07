# 2_edge_device/main.py
"""
Điểm khởi chạy chính cho hệ thống điều hướng người khiếm thị.

Cách dùng:
    python main.py                          # Webcam, có hiển thị
    python main.py --source video.mp4       # File video offline
    python main.py --source 0 --no-display  # Chỉ profile, không hiển thị
    python main.py --source 0 --no-audio    # Tắt Audio Broker
    python main.py --source 0 --device cpu  # Ép buộc dùng CPU
    python main.py --profile                # In profile mỗi 30 frame ra console

Cấu hình được đọc từ configs/default_config.yaml.
Các tham số CLI sẽ ghi đè giá trị trong file config.
"""
import os
import sys
import argparse
import yaml

# Đảm bảo chạy được từ bất kỳ thư mục nào
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.vision_local.yolo_onnx_runner import YoloOnnxRunner
from modules.vision_local.tracker import ByteTrackerWrapper
from modules.decision.priority_scorer import PriorityScorer
from modules.decision.track_state_manager import TrackStateManager
from modules.utils.profiler import PipelineProfiler
from modules.utils.stream_loader import StreamLoader
from pipeline_runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PBL6 Blind Navigation – ONNX + ByteTrack Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source", default=None,
        help="Nguồn video: 0/1/... (webcam index) hoặc path file video (.mp4, .avi, ...)"
    )
    parser.add_argument(
        "--config", default="configs/default_config.yaml",
        help="Đường dẫn file cấu hình YAML"
    )
    parser.add_argument(
        "--model", default=None,
        help="Ghi đè đường dẫn file ONNX trong config"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Ghi đè ngưỡng confidence"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Ghi đè Execution Provider (cpu | cuda)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Chạy không hiển thị cửa sổ OpenCV (headless / SSH)"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Tắt Audio Broker (chỉ dùng khi debug)"
    )
    parser.add_argument(
        "--no-lane", action="store_true",
        help="Tắt overlay phân làn trên màn hình"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="In bảng latency chi tiết ra console mỗi 30 frame"
    )
    parser.add_argument(
        "--cooldown", type=float, default=None,
        help="Ghi đè cooldown cảnh báo (giây)"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        print(f"[ERROR] Không tìm thấy file config: {full_path}")
        sys.exit(1)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_on_alert(no_audio: bool, profile_mode: bool):
    """
    Xây dựng callback on_alert:
      - Nếu có AudioBroker: gọi speak().
      - Nếu no_audio: chỉ in console.
    """
    if no_audio:
        def _console_alert(text: str) -> None:
            print(f"[ALERT] {text}")
        return _console_alert, None

    try:
        from modules.audio.audio_broker import AsyncAudioBroker
        broker = AsyncAudioBroker(port=8080)
        print("[INFO] Audio Broker đang phục vụ trên cổng 8080")
        print("[INFO] Mở Chrome trên điện thoại: http://<IP_CỦA_MÁY>:8080")

        def _audio_alert(text: str) -> None:
            broker.speak(text, cooldown=0.1)

        return _audio_alert, broker

    except Exception as exc:
        print(f"[WARN] Không thể khởi tạo AudioBroker ({exc}). Chuyển sang console-only.")

        def _fallback_alert(text: str) -> None:
            print(f"[ALERT] {text}")

        return _fallback_alert, None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ── Xác định source ────────────────────────────────────────────────────────
    source_raw = args.source or str(cfg["source"].get("input", 0))
    source = int(source_raw) if source_raw.isdigit() else source_raw

    # ── Xây dựng đường dẫn model ───────────────────────────────────────────────
    model_path = os.path.join(
        BASE_DIR,
        args.model or cfg["model"]["onnx_path"]
    )
    if not os.path.exists(model_path):
        print(f"[ERROR] Không tìm thấy model ONNX: {model_path}")
        print("        Chạy: python experiments/export_onnx.py")
        sys.exit(1)

    # ── Chọn Execution Provider ────────────────────────────────────────────────
    providers = None
    if args.device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ── Khởi tạo tất cả module ────────────────────────────────────────────────
    print("[INIT] Khởi tạo YOLOv8 ONNX Detector...")
    detector = YoloOnnxRunner(
        model_path=model_path,
        target_size=cfg["model"]["target_size"],
        conf_thresh=args.conf or cfg["model"]["conf_threshold"],
        nms_thresh=cfg["model"]["nms_threshold"],
        num_threads=cfg["model"].get("num_threads", 4),
        providers=providers
    )

    print("[INIT] Khởi tạo ByteTrack Tracker...")
    tracker = ByteTrackerWrapper(
        track_thresh=cfg["tracker"]["track_threshold"],
        track_buffer=cfg["tracker"]["track_buffer"],
        match_thresh=cfg["tracker"]["match_threshold"]
    )

    weights_path = os.path.join(BASE_DIR, "configs/class_weights.yaml")
    pcfg = cfg["policy"]
    w = pcfg["weights"]

    print("[INIT] Khởi tạo Decision Policy...")
    scorer = PriorityScorer(
        class_weights_path=weights_path,
        w_class=w["w_class"],
        w_position=w["w_position"],
        w_size=w["w_size"]
    )
    state_mgr = TrackStateManager(
        cooldown_seconds=args.cooldown or pcfg["cooldown_seconds"],
        re_alert_delta=pcfg["re_alert_delta_priority"]
    )

    print(f"[INIT] Mở nguồn video: {source!r}")
    stream = StreamLoader(
        source=source,
        width=cfg["source"].get("width", 640),
        height=cfg["source"].get("height", 480),
        fps_sync=True
    )
    stream.open()
    meta = stream.meta
    print(f"[INIT] Stream meta: {meta.get('width')}x{meta.get('height')} "
          f"@ {meta.get('fps', 0):.1f} FPS")

    profiler = PipelineProfiler(buffer_size=30)

    # ── Audio callback ─────────────────────────────────────────────────────────
    on_alert_cb, audio_broker = build_on_alert(args.no_audio, args.profile)

    # ── Chạy pipeline ──────────────────────────────────────────────────────────
    print("\n[INFO] Pipeline sẵn sàng. Nhấn 'q' để thoát.\n" + "=" * 40)

    runner = PipelineRunner(
        detector=detector,
        tracker=tracker,
        scorer=scorer,
        state_mgr=state_mgr,
        stream=stream,
        profiler=profiler,
        on_alert=on_alert_cb,
        show_display=not args.no_display,
        show_lane=not args.no_lane,
    )
    runner.run()

    # ── Dọn dẹp Audio Broker ──────────────────────────────────────────────────
    if audio_broker is not None:
        audio_broker.stop()


if __name__ == "__main__":
    main()