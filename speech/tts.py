"""
tts.py

Text-to-Speech module for Agri-RAG.

RESPONSIBILITIES:
- Convert final answer text into speech
- Support multilingual Indian languages
- Deterministic, fast, server-side

DESIGN PRINCIPLES:
- No business logic
- No translation
- No guessing
- Fail fast on invalid input

NOTE:
- This uses gTTS for stability and simplicity
- Can be replaced later with Coqui / Azure without API changes
"""

from __future__ import annotations

import io
from typing import Dict, Optional

from gtts import gTTS


# ============================================================
# SUPPORTED LANGUAGES (gTTS-compatible)
# ============================================================

LANGUAGE_MAP = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",
    "ta": "ta",
    "te": "te",
    "ml": "ml",
    "mr": "mr",
    "bn": "bn",
    "gu": "gu",
    "pa": "pa",
    "ur": "ur",
}


# ============================================================
# PUBLIC API
# ============================================================

def synthesize_speech(
    text: str,
    language: str = "en",
) -> Dict[str, object]:
    """
    Convert text into speech.

    Returns:
    {
        "success": bool,
        "audio_bytes": bytes | None,
        "mime_type": "audio/mpeg",
        "reason": str | None
    }
    """

    # ---------------- BASIC VALIDATION ----------------
    if not text or not text.strip():
        return _fail("empty_text")

    lang_code = LANGUAGE_MAP.get(language, "en")

    try:
        tts = gTTS(
            text=text,
            lang=lang_code,
            slow=False,
        )

        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)

        audio_bytes = buffer.read()

        if not audio_bytes:
            return _fail("tts_generation_failed")

        return {
            "success": True,
            "audio_bytes": audio_bytes,
            "mime_type": "audio/mpeg",
            "reason": None,
        }

    except Exception as e:
        return _fail(f"tts_error: {e}")


# ============================================================
# INTERNAL
# ============================================================

def _fail(reason: str) -> Dict[str, object]:
    return {
        "success": False,
        "audio_bytes": None,
        "mime_type": None,
        "reason": reason,
    }
