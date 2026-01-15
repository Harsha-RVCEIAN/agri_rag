"""
translator.py

Responsible for translating user queries and system answers
between Indian languages and English.

Design principles:
- Deterministic (no creative paraphrasing)
- Safe (never translate if language detection is unreliable)
- LLM-backed (uses existing LLMClient)
- English is the INTERNAL SYSTEM LANGUAGE

IMPORTANT:
- If translation fails → caller must stop the pipeline
- This module NEVER guesses
"""

from typing import Dict

from llm.llm_client import LLMClient


# ============================================================
# CONSTANTS
# ============================================================

INTERNAL_LANGUAGE = "en"   # System language (RAG, embeddings)
MAX_TRANSLATION_TOKENS = 256


# ============================================================
# TRANSLATOR CLASS
# ============================================================

class Translator:
    """
    Translator wrapper around existing LLMClient.

    Uses:
    - Gemini (preferred) OR
    - Local model (fallback)

    Translation rules:
    - Meaning preservation over fluency
    - NO summarization
    - NO explanation
    """

    def __init__(self, provider: str = "gemini"):
        self.client = LLMClient(provider=provider)

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def to_english(self, text: str, lang_info: Dict[str, object]) -> Dict[str, object]:
        """
        Translate user input to English.

        Returns:
        {
            "text": "translated text",
            "success": True | False,
            "source_language": "hi",
        }
        """

        if not self._can_translate(text, lang_info):
            return self._fail(lang_info)

        # Already English → no translation
        if lang_info["language"] == INTERNAL_LANGUAGE:
            return {
                "text": text,
                "success": True,
                "source_language": "en",
            }

        prompt = self._build_to_english_prompt(
            text=text,
            language_name=lang_info["language_name"],
            script=lang_info["script"],
        )

        translated = self._run_translation(prompt)

        if not translated:
            return self._fail(lang_info)

        return {
            "text": translated,
            "success": True,
            "source_language": lang_info["language"],
        }

    def from_english(self, text: str, lang_info: Dict[str, object]) -> Dict[str, object]:
        """
        Translate system answer FROM English to user's language.
        """

        if not self._can_translate(text, lang_info):
            return self._fail(lang_info)

        # User wants English → no translation
        if lang_info["language"] == INTERNAL_LANGUAGE:
            return {
                "text": text,
                "success": True,
                "target_language": "en",
            }

        prompt = self._build_from_english_prompt(
            text=text,
            language_name=lang_info["language_name"],
            script=lang_info["script"],
        )

        translated = self._run_translation(prompt)

        if not translated:
            return self._fail(lang_info)

        return {
            "text": translated,
            "success": True,
            "target_language": lang_info["language"],
        }

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _can_translate(self, text: str, lang_info: Dict[str, object]) -> bool:
        """
        Translation gatekeeper.
        """
        if not text or not text.strip():
            return False

        if not lang_info:
            return False

        if not lang_info.get("is_reliable"):
            return False

        if lang_info.get("language") == "unknown":
            return False

        return True

    def _run_translation(self, prompt: str) -> str | None:
        """
        Executes translation via LLM.
        """

        try:
            result = self.client.generate(
                system_prompt="You are a precise translation engine.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=MAX_TRANSLATION_TOKENS,
            )

            return result.strip() if result else None

        except Exception:
            return None

    def _build_to_english_prompt(self, text: str, language_name: str, script: str) -> str:
        return (
            f"Translate the following text into English.\n"
            f"Language: {language_name}\n"
            f"Script: {script}\n\n"
            f"Rules:\n"
            f"- Preserve exact meaning\n"
            f"- Do NOT summarize\n"
            f"- Do NOT explain\n"
            f"- Output ONLY the translated English text\n\n"
            f"Text:\n{text}"
        )

    def _build_from_english_prompt(self, text: str, language_name: str, script: str) -> str:
        return (
            f"Translate the following English text into {language_name}.\n"
            f"Script: {script}\n\n"
            f"Rules:\n"
            f"- Preserve exact meaning\n"
            f"- Use natural phrasing for native speakers\n"
            f"- Do NOT add information\n"
            f"- Output ONLY the translated text\n\n"
            f"Text:\n{text}"
        )

    def _fail(self, lang_info: Dict[str, object]) -> Dict[str, object]:
        """
        Standard failure response.
        Forces caller to stop pipeline.
        """
        return {
            "text": None,
            "success": False,
            "language": lang_info.get("language") if lang_info else None,
        }
