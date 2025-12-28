import os
import logging
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

import torch
from transformers import pipeline

# Gemini SDK
from google import genai
from google.genai import types


# ---------------- GLOBAL SINGLETONS ----------------
# 🔑 Prevents repeated heavy initialization

_LOCAL_PIPE = None
_LOCAL_TOKENIZER = None
_GEMINI_CLIENT = None


class LLMClient:
    """
    Unified LLM client with strict role separation.

    LOCAL (FLAN-T5):
        - RAG answers ONLY
        - Grounded extraction
        - Deterministic output

    GEMINI:
        - Definitions
        - Safe fallback
        - General agriculture knowledge
    """

    def __init__(
        self,
        local_model: str = "google/flan-t5-base",
        max_input_tokens: int = 1024,
        max_output_tokens: int = 256,
        provider: str = "local",  # "local" | "gemini"
    ):
        self.provider = provider
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        global _LOCAL_PIPE, _LOCAL_TOKENIZER, _GEMINI_CLIENT

        # ---------------- LOCAL MODEL ----------------
        if self.provider == "local":
            if _LOCAL_PIPE is None:
                device = 0 if torch.cuda.is_available() else -1
                _LOCAL_PIPE = pipeline(
                    task="text2text-generation",
                    model=local_model,
                    device=device,
                )
                _LOCAL_TOKENIZER = _LOCAL_PIPE.tokenizer

            self.pipe = _LOCAL_PIPE
            self.tokenizer = _LOCAL_TOKENIZER

        # ---------------- GEMINI ----------------
        elif self.provider == "gemini":
            if _GEMINI_CLIENT is None:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY is missing")
                _GEMINI_CLIENT = genai.Client(api_key=api_key)

            self.gemini_client = _GEMINI_CLIENT

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # -------------------------------------------------
    # UTILS
    # -------------------------------------------------

    def _truncate(self, text: str) -> str:
        if not self.tokenizer:
            return text
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _dedupe_repetition(self, text: str) -> str:
        """
        Hard stop for FLAN-T5 repetition loops.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen = set()
        clean = []

        for s in sentences:
            key = s.strip().lower()
            if len(key) < 8:
                continue
            if key in seen:
                continue
            seen.add(key)
            clean.append(s.strip())

        return " ".join(clean)

    # -------------------------------------------------
    # GENERATE
    # -------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using selected provider.
        NEVER raises to API layer.
        """

        # ================= LOCAL (FLAN-T5) =================
        if self.provider == "local":
            try:
                prompt = (
                    "Answer strictly using the provided context.\n"
                    "Do not repeat sentences.\n\n"
                    f"{system_prompt}\n\n"
                    f"{user_prompt}"
                )

                prompt = self._truncate(prompt)

                output = self.pipe(
                    prompt,
                    max_new_tokens=min(
                        max_tokens or self.max_output_tokens,
                        self.max_output_tokens,
                    ),
                    do_sample=False,
                    num_beams=4,
                    repetition_penalty=1.25,
                )

                text = output[0]["generated_text"].strip()
                return self._dedupe_repetition(text)

            except Exception:
                logging.exception("❌ Local LLM failed")
                return ""

        # ================= GEMINI =================
        if self.provider == "gemini":
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=f"{system_prompt}\n\n{user_prompt}",
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens or self.max_output_tokens,
                    ),
                    timeout=10,  # 🔑 HARD TIMEOUT (major latency win)
                )

                text = (response.text or "").strip()

                # Safety: reject garbage / empty answers
                if len(text.split()) < 6:
                    return ""

                return text

            except Exception:
                logging.exception("❌ Gemini failed")
                return ""

        return ""
