"""
Speech package.

Handles voice input (STT), voice output (TTS),
audio utilities, and confidence validation.
"""

from speech.stt import transcribe_audio
from speech.tts import synthesize_speech
from speech.confidence import (
    validate_stt_result,
    validate_answer_for_voice,
)

__all__ = [
    "transcribe_audio",
    "synthesize_speech",
    "validate_stt_result",
    "validate_answer_for_voice",
]
