from dotenv import load_dotenv
load_dotenv(override=True)

import os
import logging
import re
from typing import Optional

import torch
from transformers import pipeline

from google import genai
from google.genai import types
from google.genai.errors import ClientError


# ============================================================
# GLOBAL SINGLETONS (CRITICAL FOR PERFORMANCE & STABILITY)
# ============================================================

_LOCAL_PIPE = None
_LOCAL_TOKENIZER = None
_GEMINI_CLIENT = None


# ============================================================
# HARD LIMITS (DO NOT CHANGE LIGHTLY)
# ============================================================

GEMINI_FALLBACK_MAX_TOKENS = 200     # 🔒 fixed size per call
GEMINI_DEFINITION_MAX_TOKENS = 300
GEMINI_MAX_TURNS = 1                # 🔒 IMPORTANT: prevents quota burn


# ============================================================
# CORE SYSTEM INSTRUCTION (GENERIC + SAFE)
# ============================================================

CORE_SYSTEM_PROMPT = (
    "SYSTEM ROLE: Agricultural Information Assistant.\n\n"
    "RULES:\n"
    "1. Answer clearly and concisely.\n"
    "2. Provide a short and complete summary donot give half answers,  not long explanations.\n"
    "3. If listing points, include all key points briefly.\n"
    "4. Do NOT repeat ideas or add filler text.\n"
    "5. Stop when the answer is logically complete.\n"
    "6. Never stop at mid-sentence or mid-thought.\n"
    "7. if it is extending token limit , then pick till its previous sentence and display it. \n"
)


class LLMClient:
    """
    Unified LLM client with strict role separation.

    LOCAL (FLAN-T5):
        - RAG answers ONLY

    GEMINI:
        - Definitions
        - Fallback answers
        - General agriculture knowledge
    """

    def __init__(
        self,
        local_model: str = "google/flan-t5-base",
        max_input_tokens: int = 1024,
        max_output_tokens: int = 4096,
        provider: str = "local",
    ):
        self.provider = provider
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        global _LOCAL_PIPE, _LOCAL_TOKENIZER, _GEMINI_CLIENT

        # ---------------- LOCAL MODEL ----------------
        if self.provider == "local":
            if _LOCAL_PIPE is None:
                device = 0 if torch.cuda.is_available() else -1
                logging.info(f"🔧 Initializing FLAN-T5 on device={device}")

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
                    raise RuntimeError("❌ GEMINI_API_KEY is missing")

                logging.info("🔑 Initializing Gemini client")
                _GEMINI_CLIENT = genai.Client(api_key=api_key)

            self.gemini_client = _GEMINI_CLIENT

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ============================================================
    # UTILS
    # ============================================================

    def _truncate(self, text: str) -> str:
        if not hasattr(self, "tokenizer") or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _dedupe_repetition(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen = set()
        clean = []
        for s in sentences:
            k = s.strip().lower()
            if len(k) < 8 or k in seen:
                continue
            seen.add(k)
            clean.append(s.strip())
        return " ".join(clean)

    # ============================================================
    # GENERATE
    # ============================================================

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:

        # ================= LOCAL (FLAN-T5) =================
        if self.provider == "local":
            try:
                prompt = (
                    "Answer strictly using the provided context.\n\n"
                    f"{system_prompt}\n\n{user_prompt}"
                )

                prompt = self._truncate(prompt)

                out = self.pipe(
                    prompt,
                    max_new_tokens=min(
                        max_tokens or self.max_output_tokens,
                        self.max_output_tokens,
                    ),
                    do_sample=False,
                    num_beams=4,
                    repetition_penalty=1.25,
                )

                text = self._dedupe_repetition(out[0]["generated_text"].strip())
                return text or "Answer could not be generated from the documents."

            except Exception:
                logging.exception("❌ Local LLM failed")
                return "Answer could not be generated from the documents."

        # ================= GEMINI =================
        if self.provider == "gemini":
            try:
                token_cap = max_tokens or GEMINI_FALLBACK_MAX_TOKENS

                response = self.gemini_client.models.generate_content(
                    model="models/gemini-flash-latest",
                    contents=(
                        CORE_SYSTEM_PROMPT
                        + "\n\n"
                        + system_prompt
                        + "\n\n"
                        + user_prompt
                    ),
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=token_cap,
                    ),
                )

                text = (response.text or "").strip()
                text = self._dedupe_repetition(text)

                return text or "Answer not available at the moment."

            except ClientError as e:
                # 🔒 QUOTA / RATE LIMIT HANDLING
                if "RESOURCE_EXHAUSTED" in str(e):
                    logging.warning("⚠️ Gemini quota exhausted — fallback suppressed")
                    return (
                        "I’m temporarily unable to fetch additional information. "
                        "Please try again later or rely on available documents."
                    )

                logging.exception("❌ Gemini client error")
                return "Answer service is temporarily unavailable."

            except Exception:
                logging.exception("❌ Gemini failed")
                return "Answer service is temporarily unavailable."

        return "Unable to generate answer."
