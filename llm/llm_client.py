from transformers import pipeline
import torch
import logging


class LLMClient:
    """
    Local LLM client using HuggingFace transformers.
    Responsibilities:
    - Accept final prompt
    - Call model safely
    - Return raw text
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_input_tokens: int = 4096
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        # ---- device & dtype safety ----
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=dtype,
            device_map=device_map
        )

        # tokenizer reference for length checks
        self.tokenizer = self.pipe.tokenizer

    # ---------- INTERNAL UTILS ----------

    def _truncate_prompt(self, prompt: str) -> str:
        """
        Hard truncate prompt to model context window.
        Prevents silent context loss.
        """
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        if len(tokens) <= self.max_input_tokens:
            return prompt

        # keep the last part (context + question is more important than headers)
        truncated_tokens = tokens[-self.max_input_tokens :]
        return self.tokenizer.decode(truncated_tokens)

    # ---------- MAIN GENERATION ----------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 400
    ) -> str:
        """
        Generate response from LLM.
        Returns raw assistant text.
        """

        prompt = (
            "<s>[SYSTEM]\n"
            f"{system_prompt}\n\n"
            "[USER]\n"
            f"{user_prompt}\n\n"
            "[ASSISTANT]\n"
        )

        prompt = self._truncate_prompt(prompt)

        try:
            output = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            return output[0]["generated_text"].strip()

        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA OOM during LLM generation")
            return "Not found in the provided documents."

        except Exception as e:
            logging.exception("LLM generation failed")
            return "Not found in the provided documents."
