"""
audio_broker.py - Cầu nối tương thích ngược cho Audio Module
============================================================
Export AudioManager sang tên AsyncAudioBroker để tương thích 100% với các script cũ.
"""

from .audio_manager import AudioManager

# Alias đảm bảo không gãy import ở bất kỳ đâu
AsyncAudioBroker = AudioManager
__all__ = ["AudioManager", "AsyncAudioBroker"]
