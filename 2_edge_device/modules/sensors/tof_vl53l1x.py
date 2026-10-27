from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToFReading:
    distance_m: float | None
    timestamp: float
    valid: bool
    status: str
    monotonic_timestamp: float


class VL53L1XSensor:
    def __init__(
        self,
        device: Any | None = None,
        median_window: int = 5,
        max_distance_m: float = 4.0,
    ) -> None:
        if median_window < 1:
            raise ValueError("median_window must be at least 1")
        if max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive")
        self.device = device
        self.median_window = median_window
        self.max_distance_m = max_distance_m
        self._samples: deque[float] = deque(maxlen=median_window)
        self._started = False
        self._last_read_error_log = 0.0

    def start(self) -> None:
        if self.device is None:
            try:
                import VL53L1X
            except ImportError as exc:
                raise RuntimeError(
                    "Install the VL53L1X Python driver on Raspberry Pi"
                ) from exc
            self.device = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
        self.device.open()
        self.device.start_ranging(3)
        self._started = True
        LOGGER.info(
            "VL53L1X long-range ranging started on I2C bus 1, address 0x29"
        )

    def read(self) -> ToFReading:
        if not self._started or self.device is None:
            raise RuntimeError("VL53L1X sensor has not been started")
        timestamp = time.time()
        monotonic_timestamp = time.monotonic()
        try:
            raw_mm = self.device.get_distance()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if monotonic_timestamp - self._last_read_error_log >= 5:
                LOGGER.warning("VL53L1X read failed: %s", exc)
                self._last_read_error_log = monotonic_timestamp
            return ToFReading(None, timestamp, False, "read_error", monotonic_timestamp)

        if raw_mm is None:
            return ToFReading(None, timestamp, False, "out_of_range", monotonic_timestamp)
        try:
            raw_mm = float(raw_mm)
        except (TypeError, ValueError):
            return ToFReading(None, timestamp, False, "out_of_range", monotonic_timestamp)
        if not math.isfinite(raw_mm) or raw_mm <= 0:
            return ToFReading(None, timestamp, False, "out_of_range", monotonic_timestamp)
        distance_m = raw_mm / 1000.0
        if distance_m > self.max_distance_m:
            return ToFReading(None, timestamp, False, "out_of_range", monotonic_timestamp)

        self._samples.append(distance_m)
        filtered_distance = statistics.median(self._samples)
        return ToFReading(
            filtered_distance,
            timestamp,
            True,
            "ok",
            monotonic_timestamp,
        )

    def close(self) -> None:
        if self._started and self.device is not None:
            try:
                self.device.stop_ranging()
            finally:
                self._started = False
        if self.device is not None and hasattr(self.device, "close"):
            self.device.close()
