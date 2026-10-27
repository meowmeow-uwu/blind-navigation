"""
ultrasonic_hcsr04.py - Driver cảm biến khoảng cách siêu âm HC-SR04
===================================================================
Hỗ trợ đo khoảng cách thời gian thực trên Raspberry Pi 4 với:
- Tần số đo: 20-40 Hz (bất đồng bộ đa luồng).
- Bộ lọc trung vị (Median Filter) chống gai nhiễu phản xạ.
- Cơ chế Mock Driver tự động khi chạy thử nghiệm trên máy tính phát triển (Mac/PC).
- Tích hợp cờ ngắt an toàn khẩn cấp (Emergency Fail-Safe) khi cự ly < ngưỡng quy định.
"""

import time
import threading
import logging
from typing import Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Kiểm tra thư viện phần cứng RPi.GPIO
try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except (ImportError, RuntimeError):
    HAS_RPI_GPIO = False
    logger.warning("[HC-SR04] RPi.GPIO không khả dụng. Chạy ở chế độ MOCK (Simulation).")


class HCSR04Sensor:
    """
    Driver cảm biến siêu âm HC-SR04 chạy đa luồng độc lập.
    Đo khoảng cách dựa trên thời gian bay của sóng âm (Speed of sound ~ 343 m/s).
    """

    def __init__(self,
                 trig_pin: int = 23,
                 echo_pin: int = 24,
                 min_range_m: float = 0.03,
                 max_range_m: float = 4.0,
                 poll_rate_hz: int = 30,
                 emergency_distance_m: float = 0.8,
                 filter_window: int = 5):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.poll_interval = 1.0 / poll_rate_hz
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

        # Giá trị mô phỏng phục vụ Unit Test / Mock
        self._simulated_distance: float = 2.5
        self._is_simulated = not HAS_RPI_GPIO

        if HAS_RPI_GPIO:
            self._setup_gpio()

    def _setup_gpio(self):
        """Khởi tạo chân GPIO cho Trigger và Echo."""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, GPIO.LOW)
            time.sleep(0.05)  # Chờ cảm biến ổn định
            logger.info(f"[HC-SR04] Khởi tạo GPIO BCM: TRIG={self.trig_pin}, ECHO={self.echo_pin}")
        except Exception as e:
            logger.error(f"[HC-SR04] Lỗi khởi tạo GPIO: {e}")
            self._is_simulated = True

    def _raw_measure(self) -> Optional[float]:
        """Đo trực tiếp 1 xung siêu âm (Hardware)."""
        if self._is_simulated:
            return self._simulated_distance

        try:
            # Phát xung 10 micro-giây
            GPIO.output(self.trig_pin, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(self.trig_pin, GPIO.LOW)

            pulse_start = time.time()
            timeout = pulse_start + 0.04  # Timeout 40ms (~ 6.8m)

            # Chờ chân Echo lên HIGH
            while GPIO.input(self.echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return None

            # Chờ chân Echo xuống LOW
            pulse_end = time.time()
            while GPIO.input(self.echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return None

            pulse_duration = pulse_end - pulse_start
            # Khoảng cách = (thời gian * 343 m/s) / 2
            distance_m = (pulse_duration * 343.0) / 2.0

            if self.min_range_m <= distance_m <= self.max_range_m:
                return distance_m
            return None
        except Exception as e:
            logger.error(f"[HC-SR04] Lỗi đo xung: {e}")
            return None

    def _poll_loop(self):
        """Vòng lặp đọc cảm biến liên tục trên luồng riêng."""
        while self._running:
            start_t = time.perf_counter()
            dist = self._raw_measure()

            if dist is not None:
                with self._lock:
                    self._history.append(dist)
                    # Lọc trung vị (Median Filter) để loại bỏ nhiễu gai
                    sorted_vals = sorted(self._history)
                    median_val = sorted_vals[len(sorted_vals) // 2]
                    self._latest_distance_m = median_val
                    self._latest_timestamp = time.time()
                    self._is_emergency = (median_val < self.emergency_distance_m)

            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.0, self.poll_interval - elapsed)
            time.sleep(sleep_time)

    def start(self):
        """Khởi động luồng đọc nền."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, name="HCSR04-Worker", daemon=True)
            self._thread.start()
            logger.info("[HC-SR04] Đã khởi động luồng đọc siêu âm liên tục.")

    def stop(self):
        """Dừng luồng đọc cảm biến và giải phóng GPIO."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if HAS_RPI_GPIO and not self._is_simulated:
            try:
                GPIO.cleanup([self.trig_pin, self.echo_pin])
            except Exception:
                pass
        logger.info("[HC-SR04] Đã dừng cảm biến siêu âm.")

    def get_latest_reading(self) -> Tuple[float, float, bool]:
        """
        Lấy thông tin đo mới nhất:
        Returns:
            (distance_m, timestamp, is_emergency)
        """
        with self._lock:
            return self._latest_distance_m, self._latest_timestamp, self._is_emergency

    def set_simulated_distance(self, distance_m: float):
        """Dùng cho thử nghiệm / Unit test mô phỏng kịch bản."""
        self._simulated_distance = distance_m
        with self._lock:
            self._latest_distance_m = distance_m
            self._latest_timestamp = time.time()
            self._is_emergency = (distance_m < self.emergency_distance_m)
