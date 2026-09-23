# 2_edge_device/modules/audio_broker.py
import pyttsx3
import threading
import queue
import time

class AsyncAudioBroker:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.queue = queue.Queue(maxsize=3) # Chỉ giữ tối đa 3 câu cảnh báo mới nhất
        self.running = True
        self.rate = rate
        self.volume = volume
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        
        # Khởi chạy luồng nói ngầm
        self.worker_thread = threading.Thread(target=self._process_speech, daemon=True)
        self.worker_thread.start()

    def _process_speech(self):
        # Khởi tạo engine TTS cục bộ bên trong luồng để tránh xung đột COM/ALSA
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        
        while self.running:
            try:
                # Chờ lệnh nói với timeout 0.1s
                text = self.queue.get(timeout=0.1)
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")

    def speak(self, text: str, cooldown: float = 2.0):
        """
        Đưa câu cảnh báo vào hàng đợi.
        - cooldown: Khoảng thời gian (giây) tối thiểu giữa 2 lần nhắc lại cùng 1 câu.
        """
        curr_time = time.time()
        # Chống lặp từ liên tục gây mệt mỏi thính giác
        if text == self.last_spoken_text and (curr_time - self.last_spoken_time) < cooldown:
            return
        
        if not self.queue.full():
            self.queue.put(text)
            self.last_spoken_text = text
            self.last_spoken_time = curr_time

    def stop(self):
        self.running = False
        self.worker_thread.join(timeout=1.0)