from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class AudioBroker:
    """Single-worker speech queue; inference never waits for speech output."""

    def __init__(self, speaker: Callable[[str], None] | None = None) -> None:
        self._speaker = speaker
        self._queue: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="audio-output", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, message: str, priority: int = 0) -> None:
        item = (message, priority)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                pending = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pending = None
            if pending is not None and priority < pending[1]:
                self._queue.put_nowait(pending)
                LOGGER.info("Kept higher-priority pending audio alert")
                return
            self._queue.put_nowait(item)
            LOGGER.info("Replaced pending audio alert with priority %d", priority)

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            LOGGER.warning("Audio worker did not stop within %.1f seconds", timeout)

    def _run(self) -> None:
        speaker = self._speaker or self._speak_with_tts
        while not self._stop.is_set():
            try:
                message, _ = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                speaker(message)
            except Exception:
                LOGGER.exception("Audio output failed")
            finally:
                self._queue.task_done()

    @staticmethod
    def _speak_with_tts(message: str) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
        engine.stop()
