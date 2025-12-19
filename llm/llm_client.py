import os
import logging
import re
from typing import Optional

import torch
from transformers import pipeline

# Gemini (optional)
try:
    import google.generativeai as genai
except ImportError:
    genai = None


class LLMClient:
    """
    Unified LLM client.

    Local model is used ONLY for:
    - summarization
    - grounded extraction

    Gemini is used for:
    - fallback general knowledge
    """

    def __init__(
        self,
        local_model: str = "google/flan-t5-base",
        max_input_tokens: int = 1024,
        max_output_tokens: int = 200,
        provider: str = "local"   # "local" or "gemini"
    ):
        self.provider = provider
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        self.pipe = None
        self.tokenizer = None

        # ---------------- LOCAL MODEL ----------------
        if self.provider == "local":
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                task="text2text-generation",
                model=local_model,
                device=device
            )
            self.tokenizer = self.pipe.tokenizer

        # ---------------- GEMINI ----------------
        self.gemini_model = None
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or not genai:
                logging.warning("Gemini disabled (missing key or package)")
                self.provider = "disabled"
            else:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-pro")

    # -------------------------------------------------
    # UTILS
    # -------------------------------------------------

    def _truncate(self, text: str) -> str:
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_tokens
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _dedupe_repetition(self, text: str) -> str:
        """
        Removes repeated sentence loops produced by T5.
        This is CRITICAL.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen = set()
        clean = []

        for s in sentences:
            key = s.strip().lower()
            if len(key) < 10:
                continue
            if key in seen:
                continue
            seen.add(key)
            clean.append(s.strip())

        return " ".join(clean)

    # -------------------------------------------------
    # GENERATION
    # -------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:

        # ---------- LOCAL (FLAN-T5) ----------
        if self.provider == "local":
            try:
                # 🔑 CRITICAL: T5 needs explicit summarization instruction
                prompt = (
                    "Summarize the following agricultural information clearly "
                    "without repeating sentences.\n\n"
                    f"{system_prompt}\n\n"
                    f"{user_prompt}"
                )

                prompt = self._truncate(prompt)

                output = self.pipe(
                    prompt,
                    max_new_tokens=min(
                        max_tokens or self.max_output_tokens,
                        self.max_output_tokens
                    ),
                    do_sample=False,
                    repetition_penalty=1.2,   # 🔑 critical
                    num_beams=4               # 🔑 critical
                )

                text = output[0]["generated_text"].strip()
                return self._dedupe_repetition(text)

            except Exception:
                logging.exception("Local LLM failed")
                return ""

        # ---------- GEMINI ----------
        if self.provider == "gemini" and self.gemini_model:
            try:
                response = self.gemini_model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens or self.max_output_tokens
                    }
                )
                return response.text.strip()
            except Exception:
                logging.exception("Gemini failed")
                return ""

        return ""
