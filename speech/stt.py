"""
stt.py

Speech-to-Text module using Whisper (faster-whisper).

RESPONSIBILITIES:
- Convert validated audio into text
- Provide confidence signals
- Perform NO translation
- Perform NO normalization
- Perform NO language guessing beyond Whisper output

DESIGN:
- Deterministic
- Fail-fast
- CPU-safe
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from faster_whisper import WhisperModel

from speech.audio_utils import prepare_audio_for_stt


# ============================================================
# CONFIG
# ============================================================

WHISPER_MODEL_SIZE = "small"     # balanced for accuracy + speed
DEVICE = "cpu"
COMPUTE_TYPE = "int8"            # safe on most systems

# Confidence thresholds
MIN_AVG_LOGPROB = -1.0           # below this → unreliable but usable


# ============================================================
# WHISPER SINGLETON
# ============================================================

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
    return _model


# ============================================================
# PUBLIC API
# ============================================================

def transcribe_audio(
    audio_bytes: bytes,
    filename: str | None = None,
) -> Dict[str, Any]:
    """
    Transcribe audio into text.

    Returns:
    {
        "success": bool,
        "text": str | None,
        "language": str | None,
        "confidence": float,
        "reason": str | None
    }
    """

    # ---------------- AUDIO PREP ----------------
    try:
        waveform, _ = prepare_audio_for_stt(audio_bytes, filename)
    except Exception as e:
        return _fail(f"audio_error: {e}")

    # ---------------- TRANSCRIPTION ----------------
    try:
        model = _get_model()

        segments, info = model.transcribe(
            waveform,
            language=None,        # auto-detect
            beam_size=5,
            vad_filter=True,
            temperature=0.0,
        )

    except Exception as e:
        return _fail(f"stt_failure: {e}")

    # ---------------- COLLECT TEXT ----------------
    texts = []
    logprobs = []

    for seg in segments:
        if seg.text and seg.text.strip():
            texts.append(seg.text.strip())
            if seg.avg_logprob is not None:
                logprobs.append(seg.avg_logprob)

    # Silence / no speech guard
    if not texts:
        return _fail("no_speech_detected")

    full_text = " ".join(texts).strip()

    # ---------------- CONFIDENCE ----------------
    if logprobs:
        avg_logprob = sum(logprobs) / len(logprobs)
    else:
        avg_logprob = -10.0

    confidence = _logprob_to_confidence(avg_logprob)

    language = getattr(info, "language", None)

    if avg_logprob < MIN_AVG_LOGPROB:
        return {
            "success": True,  # 🔥 still usable
            "text": full_text,
            "language": language,
            "confidence": confidence,
            "reason": "low_confidence_transcription",
        }

    return {
        "success": True,
        "text": full_text,
        "language": language,
        "confidence": confidence,
        "reason": None,
    }


def validate_stt_result(stt_result: dict) -> dict:
    """
    Validates STT output for usability (not correctness).

    Returns:
    {
        "is_valid": bool,
        "reason": str | None
    }
    """

    if not stt_result:
        return {"is_valid": False, "reason": "empty_result"}

    if not stt_result.get("success"):
        return {
            "is_valid": False,
            "reason": stt_result.get("reason", "stt_failed"),
        }

    text = (stt_result.get("text") or "").strip()

    if not text:
        return {"is_valid": False, "reason": "no_text"}

    # 🔥 DO NOT reject low confidence
    return {"is_valid": True, "reason": None}


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _logprob_to_confidence(avg_logprob: float) -> float:
    """
    Maps Whisper logprob → [0, 1] confidence.
    """
    if avg_logprob >= -0.3:
        return 0.95
    if avg_logprob >= -0.6:
        return 0.85
    if avg_logprob >= -1.0:
        return 0.7
    if avg_logprob >= -1.5:
        return 0.55
    return 0.35


def _fail(reason: str) -> Dict[str, Any]:
    return {
        "success": False,
        "text": None,
        "language": None,
        "confidence": 0.0,
        "reason": reason,
    }
