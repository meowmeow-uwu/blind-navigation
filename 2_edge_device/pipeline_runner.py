# 2_edge_device/pipeline_runner.py
"""
PipelineRunner — Đóng gói toàn bộ vòng lặp xử lý thành một lớp có thể tái sử dụng.

Không phụ thuộc vào bất kỳ framework nào, không có global state.
Mọi phụ thuộc được inject qua constructor (Dependency Injection).
"""
from typing import Optional, Callable, Any
import cv2
import time

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


class PipelineRunner:
    """
    Vòng lặp pipeline chính, kết nối tuần tự:
        Capture → Inference → Tracking → Policy → Render → Alert

    Args:
        detector: Instance của YoloOnnxRunner.
        tracker: Instance của ByteTrackerWrapper.
        scorer: Instance của PriorityScorer.
        state_mgr: Instance của TrackStateManager.
        stream: Instance của StreamLoader.
        profiler: Instance của PipelineProfiler.
        on_alert: Callback(text: str) được gọi khi có cảnh báo mới.
                  Mặc định in ra console. Truyền audio_broker.speak để phát âm thanh.
        show_display: Hiển thị cửa sổ OpenCV hay không.
        show_lane: Vẽ overlay phân làn.
        evict_interval: Số frame giữa 2 lần tự dọn TTL-track cũ.
    """

    def __init__(
        self,
        detector: YoloOnnxRunner,
        tracker: ByteTrackerWrapper,
        scorer: PriorityScorer,
        state_mgr: TrackStateManager,
        stream: StreamLoader,
        profiler: Optional[PipelineProfiler] = None,
        on_alert: Optional[Callable[[str], Any]] = None,
        show_display: bool = True,
        show_lane: bool = True,
        evict_interval: int = 150,
        window_title: str = "Blind Navigation | q = Thoát"
    ):
        self.detector = detector
        self.tracker = tracker
        self.scorer = scorer
        self.state_mgr = state_mgr
        self.stream = stream
        self.profiler = profiler or PipelineProfiler()
        self.on_alert = on_alert or (lambda text: print(f"[ALERT] {text}"))
        self.show_display = show_display
        self.show_lane = show_lane
        self.evict_interval = evict_interval
        self.window_title = window_title

        self._current_alert: str = ""
        self._frame_count: int = 0
        self._running: bool = False

    def run(self) -> None:
        """Bắt đầu vòng lặp chính. Chặn đến khi người dùng thoát hoặc video hết."""
        self._running = True

        if self.show_display:
            cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

        try:
            for frame in self.stream:
                if not self._running:
                    break
                self._process_frame(frame)

                if self.show_display:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Dừng vòng lặp từ luồng khác (thread-safe)."""
        self._running = False

    def _process_frame(self, frame) -> None:
        self._frame_count += 1

        # ── Capture ────────────────────────────────────────────────────────────
        with self.profiler.measure("capture"):
            img_bgr = frame.copy()
        img_h, img_w = img_bgr.shape[:2]

        # ── Inference ──────────────────────────────────────────────────────────
        with self.profiler.measure("inference"):
            boxes, confs, cls_ids = self.detector.detect_raw(img_bgr)

        # ── Tracking ───────────────────────────────────────────────────────────
        with self.profiler.measure("tracking"):
            tracked = self.tracker.update(boxes, confs, cls_ids, classes=COCO_CLASSES)

        # ── Policy ─────────────────────────────────────────────────────────────
        with self.profiler.measure("policy"):
            if self._frame_count % self.evict_interval == 0:
                evicted = self.state_mgr.evict_stale()
                if evicted:
                    print(f"[GC] Đã xóa {evicted} track cũ. "
                          f"Track đang hoạt động: {self.state_mgr.active_track_count}")

            top1 = self.scorer.top1(tracked, img_w, img_h)

            if top1 is not None:
                for obj in tracked:
                    if obj["track_id"] != top1["track_id"]:
                        self.state_mgr.touch(obj["track_id"], 0.0)

                if self.state_mgr.should_alert(top1["track_id"], top1["priority_score"]):
                    alert_text = generate_alert(top1, tracked, img_w)
                    if alert_text:
                        self._current_alert = alert_text
                        self.on_alert(alert_text)

        # ── Render ─────────────────────────────────────────────────────────────
        with self.profiler.measure("render"):
            if self.show_display:
                if self.show_lane:
                    draw_lane_overlay(img_bgr)
                draw_tracked_detections(
                    img_bgr, tracked,
                    top1_track_id=top1["track_id"] if top1 else None
                )
                draw_alert_banner(img_bgr, self._current_alert)
                self.profiler.draw_hud(img_bgr)
                cv2.imshow(self.window_title, img_bgr)

        self.profiler.tick_frame()

    def _cleanup(self) -> None:
        self.stream.close()
        if self.show_display:
            cv2.destroyAllWindows()

        stats = self.profiler.get_stats()
        print("\n" + "=" * 58)
        print("  PROFILE SUMMARY (rolling avg 30 frames)")
        print("=" * 58)
        print(f"  FPS:              {stats['fps']:.1f}")
        print(f"  Capture:          {stats.get('capture', 0):.2f} ms")
        print(f"  Inference (ONNX): {stats.get('inference', 0):.2f} ms")
        print(f"  Tracking (BT):    {stats.get('tracking', 0):.2f} ms")
        print(f"  Policy:           {stats.get('policy', 0):.2f} ms")
        print(f"  Render:           {stats.get('render', 0):.2f} ms")
        print(f"  Total Pipeline:   {stats.get('total_pipeline_ms', 0):.2f} ms")
        print("=" * 58)
