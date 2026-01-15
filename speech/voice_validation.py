"""
voice_validation.py

Validates whether an answer is suitable for voice playback.

RESPONSIBILITIES:
- Prevent speaking low-quality or unsafe answers
- Avoid reading nonsense / hallucinations aloud
- UX guard for TTS

DESIGN:
- Conservative
- Deterministic
- No ML
"""

from typing import Dict


# ============================================================
# CONFIG
# ============================================================

MIN_CONFIDENCE_FOR_VOICE = 0.45
MAX_VOICE_CHARS = 800   # speaking very long answers is bad UX


# ============================================================
# PUBLIC API
# ============================================================

def validate_answer_for_voice(
    text: str,
    confidence: float,
) -> Dict[str, object]:
    """
    Returns:
    {
        "is_valid": bool,
        "reason": str | None
    }
    """

    if not text or not text.strip():
        return {
            "is_valid": False,
            "reason": "empty_answer",
        }

    if confidence < MIN_CONFIDENCE_FOR_VOICE:
        return {
            "is_valid": False,
            "reason": "low_confidence_for_voice",
        }

    if len(text) > MAX_VOICE_CHARS:
        return {
            "is_valid": False,
            "reason": "answer_too_long_for_voice",
        }

    # 🚫 Avoid speaking structured junk
    forbidden_markers = [
        "```",
        "<html",
        "{",
        "}",
    ]

    lowered = text.lower()
    if any(m in lowered for m in forbidden_markers):
        return {
            "is_valid": False,
            "reason": "non_speakable_content",
        }

    return {
        "is_valid": True,
        "reason": None,
    }
