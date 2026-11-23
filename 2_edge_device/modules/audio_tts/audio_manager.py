"""
modules/audio_tts/audio_manager.py - Bridge tới AudioManager cốt lõi
=====================================================================
"""

from modules.audio.audio_manager import AudioManager

AudioFeedbackManager = AudioManager
__all__ = ["AudioManager", "AudioFeedbackManager"]
