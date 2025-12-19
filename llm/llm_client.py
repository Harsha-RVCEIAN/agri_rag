import os
import logging
import re
from typing import Optional

import torch
from transformers import pipeline

# ✅ NEW Gemini SDK (REQUIRED)
from google import genai
from google.genai import types


class LLMClient:
    """
    Unified LLM client (STRICT ROLE SEPARATION).

    LOCAL (FLAN-T5):
        - Summarization
        - Grounded extraction
        - RAG answers ONLY

    GEMINI:
        - Fallback
        - General agricultural knowledge
        - Definitions (e.g., organic farming)
    """

    # -------------------------------------------------
    # INIT
    # -------------------------------------------------

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

        self.pipe = None
        self.tokenizer = None
        self.gemini_client = None

        # ---------------- LOCAL MODEL ----------------
        if self.provider == "local":
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                task="text2text-generation",
                model=local_model,
                device=device,
            )
            self.tokenizer = self.pipe.tokenizer

        # ---------------- GEMINI ----------------
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")

            self.gemini_client = genai.Client(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # -------------------------------------------------
    # UTILS
    # -------------------------------------------------

    def _truncate(self, text: str) -> str:
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _dedupe_repetition(self, text: str) -> str:
        """
        HARD FIX for FLAN-T5 looping garbage.
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
                # 🔑 FLAN needs explicit instruction
                prompt = (
                    "Summarize the following agricultural information clearly.\n"
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
                    repetition_penalty=1.25,
                    num_beams=4,
                )

                text = output[0]["generated_text"].strip()
                return self._dedupe_repetition(text)

            except Exception as e:
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
                )

                text = response.text.strip()

                # HARD SAFETY: Gemini must answer properly
                if len(text.split()) < 6:
                    return ""

                return text

            except Exception as e:
                logging.exception("❌ Gemini failed")
                return ""

        return ""
