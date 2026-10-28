"""
ultrasonic_hcsr04.py - Driver cảm biến khoảng cách siêu âm / Laser (30-50 Hz)
=============================================================================
Yêu cầu 1: Kiến trúc xử lý bất đồng bộ đa luồng (Multi-threaded Asynchronous)
- Luồng cảm biến chạy độc lập ở tần số cao (30-50 Hz).
- Tích hợp Callback ngắt khẩn cấp trực tiếp (< 30ms) khi phát hiện cự ly < ngưỡng,
  hoàn toàn không bị phụ thuộc hay nghẽn bởi tốc độ suy luận của mô hình AI!
- Bộ đệm dữ liệu an toàn luồng (Thread-safe) và cơ chế Non-blocking Queue.
- Hỗ trợ cả phần cứng thật (RPi.GPIO) lẫn chế độ mô phỏng (Simulation/Keyboard/Web).
"""

import time
import queue
import threading
import logging
from typing import Optional, Tuple, Callable
from collections import deque

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except (ImportError, RuntimeError):
    HAS_RPI_GPIO = False


class HCSR04Sensor:
    """
    Driver cảm biến khoảng cách chạy đa luồng độc lập (30-50 Hz).
    Phản ứng khẩn cấp tức thời (< 30ms) qua callback độc lập với luồng AI.
    """

    def __init__(self,
                 trig_pin: int = 23,
                 echo_pin: int = 24,
                 min_range_m: float = 0.03,
                 max_range_m: float = 4.0,
                 poll_rate_hz: int = 40,
                 emergency_distance_m: float = 1.0,
                 filter_window: int = 3):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.poll_interval = 1.0 / max(10, min(poll_rate_hz, 100))
        self.emergency_distance_m = emergency_distance_m
        self.filter_window = filter_window

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Dữ liệu đo mới nhất
        self._latest_distance_m: float = 2.5
        self._latest_timestamp: float = time.time()
        self._history = deque(maxlen=filter_window)
        self._is_emergency: bool = False

        # Hàng đợi bất đồng bộ (Non-blocking Queue)
        self.reading_queue: queue.Queue = queue.Queue(maxsize=10)

        # Callback khẩn cấp trực tiếp phản ứng < 30ms (Yêu cầu 1)
        self._emergency_callback: Optional[Callable[[float], None]] = None
        self._last_emergency_cb_time = 0.0

        # Mô phỏng (Simulation)
        self._simulated_distance: float = 2.5
        self._is_simulated = not HAS_RPI_GPIO
        self.is_manually_overridden = False

        if HAS_RPI_GPIO:
            self._setup_gpio()

    def set_emergency_callback(self, cb: Callable[[float], None]):
        """Đăng ký callback gọi tức thời (< 30ms) khi cảm biến phát hiện cự ly khẩn cấp."""
        self._emergency_callback = cb

    def _setup_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, GPIO.LOW)
            time.sleep(0.05)
        except Exception as e:
            logger.error(f"[DISTANCE SENSOR] Lỗi GPIO: {e}. Chuyển sang mô phỏng.")
            self._is_simulated = True

    def _raw_measure(self) -> Optional[float]:
        if self._is_simulated:
            return self._simulated_distance

        try:
            GPIO.output(self.trig_pin, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(self.trig_pin, GPIO.LOW)

            pulse_start = time.time()
            timeout = pulse_start + 0.03

            while GPIO.input(self.echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return None

            pulse_end = time.time()
            while GPIO.input(self.echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return None

            duration = pulse_end - pulse_start
            distance_m = (duration * 343.0) / 2.0
            if self.min_range_m <= distance_m <= self.max_range_m:
                return distance_m
            return None
        except Exception:
            return None

    def _poll_loop(self):
        """Vòng lặp đọc cảm biến tốc độ cao (30-50 Hz) chạy độc lập."""
        while self._running:
            t0 = time.perf_counter()
            dist = self._raw_measure()

            if dist is not None:
                with self._lock:
                    self._history.append(dist)
                    sorted_vals = sorted(self._history)
                    median_val = sorted_vals[len(sorted_vals) // 2]
                    self._latest_distance_m = median_val
                    self._latest_timestamp = time.time()
                    is_emer = (median_val < self.emergency_distance_m)
                    self._is_emergency = is_emer

                # Đẩy vào Non-blocking Queue
                try:
                    self.reading_queue.put_nowait((median_val, self._latest_timestamp, is_emer))
                except queue.Full:
                    try:
                        self.reading_queue.get_nowait()
                        self.reading_queue.put_nowait((median_val, self._latest_timestamp, is_emer))
                    except Exception:
                        pass

                # KÍCH HOẠT KHẨN CẤP TỨC THỜI (< 30ms):
                # Không chờ AI, gọi thẳng callback để phát tín hiệu an toàn ngay lập tức
                if is_emer and self._emergency_callback:
                    now = time.time()
                    if now - self._last_emergency_cb_time >= 0.15:
                        self._last_emergency_cb_time = now
                        try:
                            self._emergency_callback(median_val)
                        except Exception as e:
                            logger.error(f"[DISTANCE SENSOR] Lỗi gọi emergency callback: {e}")

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, self.poll_interval - elapsed)
            time.sleep(sleep_time)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, name="Sensor-Thread", daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if HAS_RPI_GPIO and not self._is_simulated:
            try:
                GPIO.cleanup([self.trig_pin, self.echo_pin])
            except Exception:
                pass

    def get_latest_reading(self) -> Tuple[float, float, bool]:
        """Lấy giá trị đo mới nhất: (distance_m, timestamp, is_emergency)"""
        with self._lock:
            return self._latest_distance_m, self._latest_timestamp, self._is_emergency

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    def set_simulated_distance(self, distance_m: float, manual: bool = True):
        if manual:
            self.is_manually_overridden = True
        self._simulated_distance = distance_m
        is_emer = (distance_m < self.emergency_distance_m)
        with self._lock:
            self._history.clear()
            self._history.append(distance_m)
            self._latest_distance_m = distance_m
            self._latest_timestamp = time.time()
            self._is_emergency = is_emer

        if is_emer and self._emergency_callback:
            now = time.time()
            if now - self._last_emergency_cb_time >= 0.1:
                self._last_emergency_cb_time = now
                try:
                    self._emergency_callback(distance_m)
                except Exception:
                    pass
