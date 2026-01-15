"""
confidence.py

Confidence arbitration for voice-based interaction.

PURPOSE:
- Decide whether speech input is reliable enough to continue
- Decide whether system output is safe to speak aloud
- Prevent low-confidence audio → confident voice output

DESIGN PRINCIPLES:
- Conservative (silence is better than wrong speech)
- Deterministic
- Transparent reasons for failure
- No ML, no heuristics hidden from caller

THIS MODULE DOES NOT:
- Perform STT
- Perform TTS
- Modify text
"""

from __future__ import annotations

from typing import Dict, Optional


# ============================================================
# THRESHOLDS (VOICE SAFETY)
# ============================================================

# ---- STT ----
MIN_STT_CONFIDENCE = 0.65        # below this → ask user to repeat
MIN_TEXT_LENGTH = 3             # too short → unreliable by definition

# ---- ANSWER ----
MIN_ANSWER_CONFIDENCE_FOR_VOICE = 0.55
ABSOLUTE_BLOCK_CONFIDENCE = 0.40   # NEVER speak below this


# ============================================================
# PUBLIC API
# ============================================================

def validate_stt_result(
    stt_result: Dict[str, object],
) -> Dict[str, object]:
    """
    Validate STT output before NLP / RAG.

    Input contract (from stt.py):
    {
        "success": bool,
        "text": str | None,
        "language": str | None,
        "confidence": float,
        "reason": str | None
    }

    Returns:
    {
        "is_valid": bool,
        "reason": str | None
    }
    """

    if not stt_result:
        return _fail("missing_stt_result")

    if not stt_result.get("success"):
        return _fail(stt_result.get("reason") or "stt_failed")

    text = (stt_result.get("text") or "").strip()
    confidence = float(stt_result.get("confidence", 0.0))

    if not text:
        return _fail("empty_transcription")

    if len(text.split()) < MIN_TEXT_LENGTH:
        return _fail("transcription_too_short")

    if confidence < MIN_STT_CONFIDENCE:
        return _fail("low_stt_confidence")

    return {
        "is_valid": True,
        "reason": None,
    }


def validate_answer_for_voice(
    answer_text: Optional[str],
    answer_confidence: float,
) -> Dict[str, object]:
    """
    Decide whether an answer is safe to speak aloud.

    Rules:
    - Never speak very low confidence answers
    - Prefer silence + text over wrong audio
    """

    if not answer_text or not answer_text.strip():
        return _fail("empty_answer")

    if answer_confidence < ABSOLUTE_BLOCK_CONFIDENCE:
        return _fail("confidence_too_low_to_speak")

    if answer_confidence < MIN_ANSWER_CONFIDENCE_FOR_VOICE:
        return {
            "is_valid": False,
            "reason": "text_only_recommended",
        }

    return {
        "is_valid": True,
        "reason": None,
    }


# ============================================================
# INTERNAL
# ============================================================

def _fail(reason: str) -> Dict[str, object]:
    return {
        "is_valid": False,
        "reason": reason,
    }
