"""
Inference module for FactorioLLM.
Loads the fine-tuned LoRA model via Unsloth and generates
Draftsman Python scripts based on user requests.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Generator, List, Dict, Any, Optional

# Default system prompt with clear domain rules matching factorio-draftsman
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Factorio blueprint generator AI. Write Python code using the factorio-draftsman library "
    "to construct the exact requested blueprint.\n"
    "Requirements:\n"
    "- Write compact, algorithmic code. Never hardcode repeating entities line-by-line.\n"
    "- High-level macro functions from `draftsman_helpers` are pre-imported and available to make code concise:\n"
    "    * add_belt_line(bp, start=(x, y), length=N, direction='east'|'west'|'north'|'south', belt_type='transport-belt'|'fast-transport-belt')\n"
    "    * add_underground_pair(bp, start=(x1, y1), end=(x2, y2), direction='east', belt_type='underground-belt'|'fast-underground-belt')\n"
    "    * add_entity_row(bp, entity_name, start=(x, y), count=N, step=(dx, dy), direction=dir)\n"
    "    * add_power_poles(bp, start=(x, y), count=N, step=(dx, dy), pole_type='small-electric-pole'|'medium-electric-pole')\n"
    "- You can freely combine these macros with standard Python loops and factorio-draftsman entities.\n"
    "- Ensure valid orientations and non-overlapping coordinates.\n"
    "- The script must always conclude with: print(bp.to_string())"
)


def extract_python_code(raw_text: str) -> str:
    """
    Extracts pure Python code from LLM output.
    Handles markdown blocks (```python ... ```) or raw python code.
    """
    text = raw_text.strip()

    # Match ```python ... ``` or ``` ... ```
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        # Take the largest code block if multiple exist
        best_code = max(matches, key=len).strip()
        return best_code

    # If no fences found, check if text looks like python code
    lines = text.splitlines()
    code_lines = []
    started = False

    for line in lines:
        stripped = line.strip()
        if not started:
            # Look for common python start lines
            if (
                stripped.startswith("from ")
                or stripped.startswith("import ")
                or stripped.startswith("#")
                or stripped.startswith("bp = ")
                or stripped.startswith("def ")
            ):
                started = True
                code_lines.append(line)
        else:
            # Stop if we hit conversation delimiters or EOS markers
            if "<|eot_id|>" in line or "<|start_header_id|>" in line:
                break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    # Fallback to returning the stripped text
    return text


def is_code_complete(code_text: str) -> tuple[bool, str]:
    """
    Checks whether the extracted Python code is syntactically complete and finishes
    with print(bp.to_string()).
    Returns (is_complete, reason).
    """
    import ast
    code = code_text.strip()
    if not code:
        return False, "Code is empty."

    # Check for blueprint object creation
    if "Blueprint(" not in code:
        return False, "Script does not instantiate Blueprint()."

    # Check for blueprint string output
    if "to_string()" not in code:
        return False, "Script does not call print(bp.to_string())."

    # Check Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        msg = str(e)
        if "unexpected EOF" in msg or "was never closed" in msg or "unterminated" in msg:
            return False, f"Incomplete syntax (code truncated mid-statement: {e.msg} at line {e.lineno})"
        return False, f"Syntax error: {e.msg} at line {e.lineno}"

    return True, "Complete"


def is_stuck_in_repetition(text: str, consecutive_threshold: int = 8) -> bool:
    """
    Detects autoregressive repetition degeneration by checking if identical non-empty
    lines repeat consecutively more than consecutive_threshold times.
    Completely domain-agnostic.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < consecutive_threshold:
        return False

    last_slice = lines[-consecutive_threshold:]
    return len(set(last_slice)) == 1


class FactorioInference:
    """
    Wraps FastLanguageModel for fast 4-bit LoRA inference.
    """

    def __init__(
        self,
        model_path: str = "factorio_lora_model",
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Loads model and tokenizer into memory if not already loaded."""
        if self._is_loaded:
            return

        import torch
        if getattr(torch.utils, "_pytree", None) and not hasattr(torch.utils._pytree, "register_constant"):
            torch.utils._pytree.register_constant = lambda x: None
        from unsloth import FastLanguageModel

        print(f"[*] Loading model from '{self.model_path}'...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        # Enable 2x faster inference in Unsloth
        FastLanguageModel.for_inference(self.model)
        self._is_loaded = True
        print("[*] Model loaded successfully and configured for inference.")

    def format_messages(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Formats conversation history into Llama-3 chat template.
        Ensures the system prompt is present at the beginning.
        """
        messages = []
        # Ensure system message is first
        has_system = any(m.get("role") == "system" for m in conversation_history)
        if not has_system:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.extend(conversation_history)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        conversation_history: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.12,
        max_new_tokens: int = 2560,
    ) -> str:
        """Synchronously generates the assistant's reply."""
        self.load_model()
        formatted_prompt = self.format_messages(conversation_history)

        max_input_length = max(self.max_seq_length - max_new_tokens, 512)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 1e-4,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off the input tokens to get only generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return raw_output.strip()

    def generate_stream(
        self,
        conversation_history: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.12,
        max_new_tokens: int = 2560,
        max_total_tokens: int = 4096,
        auto_continue: bool = True,
    ) -> Generator[str, None, None]:
        """
        Streams generated tokens one by one.
        If generation ends before the script is complete (e.g. hits max_new_tokens mid-script),
        automatically continues generating up to max_total_tokens to give the model
        enough time and tokens to finish cleanly.
        """
        self.load_model()
        from transformers import TextIteratorStreamer

        formatted_prompt = self.format_messages(conversation_history)

        max_input_length = max(self.max_seq_length - max_new_tokens, 512)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 1e-4,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_generated_text = ""
        total_tokens_approx = 0

        for new_text in streamer:
            full_generated_text += new_text
            total_tokens_approx += 1
            yield new_text

        thread.join()

        # If auto_continue is enabled, check if the script is complete.
        # If incomplete and not stuck in a repetition loop, continue generating.
        continuation_rounds = 0
        max_continuation_rounds = 3

        while auto_continue and continuation_rounds < max_continuation_rounds:
            code = extract_python_code(full_generated_text)
            comp, _ = is_code_complete(code)
            if comp:
                break

            if total_tokens_approx >= max_total_tokens:
                break

            if is_stuck_in_repetition(full_generated_text):
                break

            continuation_rounds += 1
            tokens_remaining = max_total_tokens - total_tokens_approx
            chunk_tokens = min(1024, max(tokens_remaining, 256))

            cont_prompt = formatted_prompt + full_generated_text
            cont_max_input = max(self.max_seq_length - chunk_tokens, 512)
            cont_inputs = self.tokenizer(
                cont_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cont_max_input,
            ).to(self.model.device)

            cont_streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            cont_kwargs = dict(
                **cont_inputs,
                streamer=cont_streamer,
                max_new_tokens=chunk_tokens,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 1e-4,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            cont_thread = threading.Thread(target=self.model.generate, kwargs=cont_kwargs)
            cont_thread.start()

            got_any = False
            for new_text in cont_streamer:
                got_any = True
                full_generated_text += new_text
                total_tokens_approx += 1
                yield new_text

            cont_thread.join()

            if not got_any:
                break
