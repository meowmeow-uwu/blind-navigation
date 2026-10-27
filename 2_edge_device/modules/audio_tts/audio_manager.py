"""
audio_manager.py - Bộ điều phối âm thanh cảnh báo ưu tiên và chống lặp
=====================================================================
Hiện thực hoá:
1. Cảnh báo khẩn cấp tức thì (Emergency Bypass): Dừng lại ngay!
2. Cảnh báo định hướng theo làn (Trái - Giữa - Phải) và cự ly.
3. Cơ chế triệt tiêu 100% hiện tượng nói lặp dựa trên Track ID và Cooldown.
"""

import time
import logging
import threading
from typing import Optional, Dict

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    logger.warning("[AUDIO] pyttsx3 không khả dụng. Chạy ở chế độ MOCK Audio.")


class AudioFeedbackManager:
    """Quản lý phát âm thanh Text-To-Speech phân tầng theo độ khẩn cấp."""

    LANE_VIETNAMESE = {
        "left": "bên trái",
        "center": "trước mặt",
        "right": "bên phải"
    }

    CLASS_VIETNAMESE = {
        "person": "người",
        "chair": "ghế",
        "dining table": "bàn",
        "couch": "ghế dài",
        "bicycle": "xe đạp",
        "motorcycle": "xe máy",
        "car": "ô tô",
        "pole": "cột",
        "stairs": "cầu thang",
        "door": "cửa",
        "potted plant": "chậu cây",
        "vật cản": "vật cản"
    }

    def __init__(self,
                 language: str = "vi",
                 speech_rate: int = 200,
                 emergency_cooldown_s: float = 0.6,
                 tts_cooldown_s: float = 2.0):
        self.language = language
        self.speech_rate = speech_rate
        self.emergency_cooldown_s = emergency_cooldown_s
        self.tts_cooldown_s = tts_cooldown_s

        self.last_emergency_time = 0.0
        self.last_speech_time = 0.0
        self.announced_tracks: Dict[int, float] = {}  # track_id -> timestamp

        self._is_speaking = False
        self._lock = threading.Lock()
        self._tts_engine = None

        if HAS_PYTTSX3:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', self.speech_rate)
            except Exception as e:
                logger.error(f"[AUDIO] Lỗi khởi tạo pyttsx3: {e}")

    def trigger_emergency(self, distance_m: Optional[float] = None):
        """
        Kênh khẩn cấp (Emergency Route): Độ ưu tiên cao nhất,
        phát lệnh 'Dừng lại' lập tức khi khoảng cách < 0.8m!
        """
        now = time.time()
        if now - self.last_emergency_time < self.emergency_cooldown_s:
            return

        self.last_emergency_time = now
        msg = "Dừng lại! Vật cản quá gần!" if distance_m is None else f"Dừng lại! {distance_m:.1f} mét!"
        print(f"\n🚨 [EMERGENCY AUDIO] >>> {msg} <<<\n")
        self._speak_async(msg, is_emergency=True)

    def announce_obstacle(self,
                          track_id: int,
                          class_name: str,
                          distance_m: float,
                          lane: str,
                          is_emergency: bool = False):
        """
        Kênh định hướng (Guidance Route): Đọc ngắn gọn 1-2 từ khoá.
        Ví dụ: 'Cột trước mặt, 1m2' hoặc 'Ghế bên phải'.
        """
        if is_emergency or distance_m < 0.8:
            self.trigger_emergency(distance_m)
            return

        now = time.time()

        # Kiểm tra chống spam theo Track ID
        if track_id in self.announced_tracks:
            last_t = self.announced_tracks[track_id]
            # Chỉ đọc lại nếu đã qua 5 giây
            if now - last_t < 5.0:
                return

        # Kiểm tra cooldown giữa 2 lần phát TTS thông thường
        if now - self.last_speech_time < self.tts_cooldown_s:
            return

        # Định dạng câu nói ngắn gọn tiếng Việt
        obj_name = self.CLASS_VIETNAMESE.get(class_name, class_name)
        lane_name = self.LANE_VIETNAMESE.get(lane, "phía trước")

        text = f"{obj_name} {lane_name}, {distance_m:.1f} mét"
        self.announced_tracks[track_id] = now
        self.last_speech_time = now

        print(f"🔊 [AUDIO GUIDANCE] {text}")
        self._speak_async(text, is_emergency=False)

    def _speak_async(self, text: str, is_emergency: bool = False):
        """Chạy TTS trên luồng riêng để không block luồng xử lý chính."""
        def _worker():
            with self._lock:
                if self._is_speaking and not is_emergency:
                    return
                self._is_speaking = True

            try:
                if self._tts_engine:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
            except Exception as e:
                logger.debug(f"[AUDIO] Lỗi phát âm: {e}")
            finally:
                with self._lock:
                    self._is_speaking = False

        threading.Thread(target=_worker, daemon=True).start()
