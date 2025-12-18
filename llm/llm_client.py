from transformers import pipeline
import torch
import logging


class LLMClient:
    """
    Optimized LLM client using HuggingFace pipeline.
    - CPU friendly
    - Fast (seconds)
    - RAG-safe
    - Keeps existing project interface
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_input_tokens: int = 1024,
        max_output_tokens: int = 256
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        # ---- device selection ----
        # pipeline uses device index: -1 = CPU, >=0 = GPU
        device = 0 if torch.cuda.is_available() else -1

        # ---- CORRECT pipeline for FLAN-T5 ----
        self.pipe = pipeline(
            task="text2text-generation",
            model=self.model_name,
            device=device
        )

        self.tokenizer = self.pipe.tokenizer

    # ---------- INTERNAL UTILS ----------

    def _truncate_prompt(self, text: str) -> str:
        """
        Truncate input safely for encoder-decoder models.
        """
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_input_tokens
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        FLAN-T5 expects instruction-style prompts, not chat tokens.
        """
        return f"""
{system_prompt}

{user_prompt}
""".strip()

    # ---------- MAIN GENERATION ----------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,   # kept for compatibility (ignored)
        max_tokens: int = 400       # kept for compatibility (overridden)
    ) -> str:
        """
        Generate response from LLM.
        Returns raw assistant text.
        """

        prompt = self._build_prompt(system_prompt, user_prompt)
        prompt = self._truncate_prompt(prompt)

        try:
            output = self.pipe(
                prompt,
                max_new_tokens=min(max_tokens, self.max_output_tokens),
                do_sample=False
            )

            return output[0]["generated_text"].strip()

        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA OOM during LLM generation")
            return "Not found in the provided documents."

        except Exception:
            logging.exception("LLM generation failed")
            return "Not found in the provided documents."
