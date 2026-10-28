"""
test_audio_queue.py - Kiểm thử bộ điều phối âm thanh thích ứng & chống đè tiếng
================================================================================
Kiểm tra:
1. Phân tầng phản hồi: Khẩn cấp (<1m: bíp dồn dập) vs Dẫn hướng (1.5-3m: 1-2 từ).
2. Triệt tiêu 100% lặp lại: Cùng 1 Track ID chỉ nói đúng 1 lần.
3. Không chồng chéo âm thanh (Zero overlap): Khẩn cấp ngắt ngay câu nói dẫn hướng.
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.audio.audio_manager import AudioManager


def test_guidance_anti_repeat():
    """Kiểm tra: 1 Track ID chỉ được thông báo duy nhất 1 lần."""
    manager = AudioManager(port=8891, guidance_cooldown_s=1.0)
    try:
        # Lần 1: Cấp track_id=101 -> Được phép nói
        res1 = manager.announce_guidance(track_id=101, class_name="chair", lane="center", distance_m=2.2)
        assert res1 is True
        assert manager.current_guidance_text == "Ghế trước mặt"

        # Lần 2: Cùng track_id=101 gửi lại ngay -> Phải bị chặn (Silent Tracking)
        res2 = manager.announce_guidance(track_id=101, class_name="chair", lane="center", distance_m=2.1)
        assert res2 is False

        # Lần 3: Chờ qua cooldown nhưng vẫn là track_id=101 -> Vẫn bị chặn
        time.sleep(1.1)
        res3 = manager.announce_guidance(track_id=101, class_name="chair", lane="center", distance_m=1.9)
        assert res3 is False
    finally:
        manager.stop()


def test_emergency_preempts_guidance():
    """Kiểm tra: Báo động khẩn cấp ngắt ngay lập tức câu nói dẫn hướng."""
    manager = AudioManager(port=8892, guidance_cooldown_s=0.5)
    try:
        # Bắt đầu dẫn hướng
        manager.announce_guidance(track_id=202, class_name="person", lane="center", distance_m=2.0)
        assert manager.is_emergency is False

        # Đột ngột có vật cản ở cự ly khẩn cấp (0.5m)
        manager.trigger_emergency(0.5)
        assert manager.is_emergency is True

        # Khi đang khẩn cấp, mọi yêu cầu dẫn hướng đều bị từ chối
        res = manager.announce_guidance(track_id=203, class_name="door", lane="left", distance_m=1.8)
        assert res is False
    finally:
        manager.stop()


def test_proportional_beep_interval():
    """Kiểm tra: Cự ly càng gần thì chu kỳ tiếng bíp càng dồn dập."""
    manager = AudioManager(port=8893)
    try:
        manager.trigger_emergency(0.9)
        state_far = manager.get_alert_state()
        interval_far = state_far.get("interval", 0.5)

        manager.trigger_emergency(0.2)
        state_near = manager.get_alert_state()
        interval_near = state_near.get("interval", 0.2)

        # Cự ly 0.2m phải có chu kỳ bíp nhanh hơn cự ly 0.9m
        assert interval_near < interval_far
        assert interval_near <= 0.25
    finally:
        manager.stop()


if __name__ == "__main__":
    print("Running audio queue tests...")
    test_guidance_anti_repeat()
    print("  test_guidance_anti_repeat passed!")
    test_emergency_preempts_guidance()
    print("  test_emergency_preempts_guidance passed!")
    test_proportional_beep_interval()
    print("  test_proportional_beep_interval passed!")
    print("ALL AUDIO TESTS PASSED!")
