import os
import logging
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
    Unified LLM client:
    - Local: FLAN-T5-Base (default, always works)
    - External: Gemini Pro (OPTIONAL fallback)
    """

    def __init__(
        self,
        local_model: str = "google/flan-t5-base",
        max_input_tokens: int = 1024,
        max_output_tokens: int = 256,
        provider: str = "local"   # "local" or "gemini"
    ):
        self.provider = provider
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        # ---------------- LOCAL (FLAN) ----------------
        self.pipe = None
        self.tokenizer = None

        if self.provider == "local":
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                task="text2text-generation",
                model=local_model,
                device=device
            )
            self.tokenizer = self.pipe.tokenizer

        # ---------------- GEMINI (OPTIONAL) ----------------
        self.gemini_model = None

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key or not genai:
                logging.warning(
                    "Gemini fallback requested but GEMINI_API_KEY not set. "
                    "Fallback will be unavailable."
                )
                self.provider = "disabled"
            else:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-pro")

    # -------------------------------------------------
    # LOCAL UTILS
    # -------------------------------------------------

    def _truncate_prompt(self, text: str) -> str:
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_tokens
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"{system_prompt}\n\n{user_prompt}".strip()

    # -------------------------------------------------
    # MAIN GENERATION
    # -------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response using selected provider.
        """

        # ---------- LOCAL (FLAN) ----------
        if self.provider == "local":
            try:
                prompt = self._build_prompt(system_prompt, user_prompt)
                prompt = self._truncate_prompt(prompt)

                out = self.pipe(
                    prompt,
                    max_new_tokens=min(
                        max_tokens or self.max_output_tokens,
                        self.max_output_tokens
                    ),
                    do_sample=False
                )
                return out[0]["generated_text"].strip()

            except Exception:
                logging.exception("Local LLM generation failed")
                return "Not found in the provided documents."

        # ---------- GEMINI (FALLBACK) ----------
        if self.provider == "gemini" and self.gemini_model:
            try:
                full_prompt = (
                    f"{system_prompt}\n\n"
                    f"{user_prompt}\n\n"
                    "Note: This answer is generated using a general AI model."
                )

                response = self.gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens or self.max_output_tokens
                    }
                )
                return response.text.strip()

            except Exception:
                logging.exception("Gemini fallback failed")
                return "Unable to generate fallback response at this time."

        # ---------- FALLBACK DISABLED ----------
        return "Not found in the provided documents."
