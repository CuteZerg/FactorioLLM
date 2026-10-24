"""
Inference module for FactorioLLM.
Loads the fine-tuned LoRA model via Unsloth and generates
Draftsman Python scripts based on user requests.
"""

from __future__ import annotations

import re
import threading
from typing import Callable, Generator, List, Dict

# Default system prompt with clear domain rules matching factorio-draftsman.
# Must stay in sync with the training dataset (export_dataset.py uses this exact text).
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Factorio 2.0 blueprint generator. Write Python code using the factorio-draftsman library "
    "that builds exactly the requested blueprint.\n"
    "Rules:\n"
    "- Coordinates are integer tiles: (x, y) is the top-left tile of an entity; x grows east, y grows south.\n"
    "- Place single entities with: bp.entities.append('name', tile_position=(x, y), direction=Direction.EAST)\n"
    "- Macros from `draftsman_helpers` (directions are 'north'|'east'|'south'|'west'):\n"
    "    * add_belt_line(bp, start=(x, y), length=N, direction='east', belt_type='transport-belt')\n"
    "    * add_underground_pair(bp, start=(x1, y1), end=(x2, y2), direction='east', belt_type='underground-belt')\n"
    "    * add_entity_row(bp, 'entity-name', start=(x, y), count=N, step=(dx, dy), direction=None, **kwargs)\n"
    "    * add_power_poles(bp, start=(x, y), count=N, step=(dx, dy), pole_type='small-electric-pole')\n"
    "- Use loops and macros for repeating structures; entities must not overlap.\n"
    "- The script must end with: print(bp.to_string())"
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


def _shorten_text(text: str, keep_lines: int, marker: str) -> str:
    """Keeps the first and last `keep_lines` lines of a text, replacing the middle with a marker line."""
    lines = text.splitlines()
    if len(lines) <= 2 * keep_lines + 1:
        return text
    omitted = len(lines) - 2 * keep_lines
    return "\n".join(lines[:keep_lines] + [marker.format(n=omitted)] + lines[-keep_lines:])


def fit_history_to_budget(
    messages: List[Dict[str, str]],
    count_tokens: Callable[[List[Dict[str, str]]], int],
    max_tokens: int,
) -> List[Dict[str, str]]:
    """
    Shrinks a conversation so that its formatted prompt fits into `max_tokens`.
    Old assistant code is shortened first (it is the bulk of retry prompts), then the middle of
    long user messages (tracebacks). The system prompt and the head and tail of every user turn
    are always kept, so the model never loses the actual request. Raises ValueError if it cannot fit.
    """
    msgs = [dict(m) for m in messages]
    if count_tokens(msgs) <= max_tokens:
        return msgs

    for role, marker in (("assistant", "# ... ({n} lines omitted) ..."), ("user", "... ({n} lines omitted) ...")):
        for keep in (40, 20, 10, 5, 2):
            for m in msgs:
                if m["role"] == role:
                    m["content"] = _shorten_text(m["content"], keep, marker)
            if count_tokens(msgs) <= max_tokens:
                return msgs

    raise ValueError(f"Prompt does not fit into {max_tokens} tokens even after shortening the history.")


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
        stream_timeout: float = 300.0,
    ):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.system_prompt = system_prompt
        self.stream_timeout = stream_timeout
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

    def _with_system_prompt(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if any(m.get("role") == "system" for m in conversation_history):
            return list(conversation_history)
        return [{"role": "system", "content": self.system_prompt}] + list(conversation_history)

    def format_messages(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Formats conversation history into the model's chat template.
        Ensures the system prompt is present at the beginning.
        """
        return self.tokenizer.apply_chat_template(
            self._with_system_prompt(conversation_history),
            tokenize=False,
            add_generation_prompt=True,
        )

    def _count_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        return len(self.tokenizer(self.format_messages(messages), add_special_tokens=False)["input_ids"])

    def _prepare_inputs(self, conversation_history: List[Dict[str, str]], max_new_tokens: int):
        """Tokenizes the conversation, shrinking old turns (never truncating) to leave room for generation."""
        history = fit_history_to_budget(
            self._with_system_prompt(conversation_history),
            self._count_prompt_tokens,
            self.max_seq_length - max_new_tokens,
        )
        # The chat template already contains the BOS token: do not add a second one.
        return self.tokenizer(
            self.format_messages(history), return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)

    def _sampling_kwargs(self, temperature: float, top_p: float, repetition_penalty: float) -> Dict:
        return dict(
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 1e-4,
            pad_token_id=self.tokenizer.eos_token_id,
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
        inputs = self._prepare_inputs(conversation_history, max_new_tokens)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **self._sampling_kwargs(temperature, top_p, repetition_penalty),
            )

        # Slice off the input tokens to get only generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return raw_output.strip()

    def _stream_once(self, input_ids, max_new_tokens: int, sampling: Dict, state: Dict) -> Generator[str, None, None]:
        """
        Runs model.generate in a worker thread and yields decoded text chunks.
        Stores the full output ids in state["output_ids"]. Worker exceptions (e.g. CUDA OOM) are
        re-raised here instead of leaving the streamer blocked forever.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        stop_flag = threading.Event()

        class _StopOnFlag(StoppingCriteria):
            def __call__(self, *args, **kwargs) -> bool:
                return stop_flag.is_set()

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            timeout=self.stream_timeout,
        )

        def _run() -> None:
            try:
                with torch.no_grad():
                    state["output_ids"] = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=StoppingCriteriaList([_StopOnFlag()]),
                        **sampling,
                    )
            except Exception as e:  # surfaced to the caller below
                state["error"] = e
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for new_text in streamer:
                yield new_text
        finally:
            stop_flag.set()  # stops generation if the consumer quit early or the streamer timed out
            thread.join()
        if "error" in state:
            raise RuntimeError(f"Generation failed: {state['error']}") from state["error"]

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
        Streams generated text chunks.
        If generation hits max_new_tokens before the script is complete, it continues from the exact
        token ids generated so far (up to max_total_tokens new tokens and the model context length).
        A generation that ended with EOS is never continued.
        """
        self.load_model()
        sampling = self._sampling_kwargs(temperature, top_p, repetition_penalty)
        input_ids = self._prepare_inputs(conversation_history, max_new_tokens)["input_ids"]
        prompt_length = input_ids.shape[1]

        full_generated_text = ""
        chunk_tokens = max_new_tokens
        max_continuation_rounds = 3

        for continuation_round in range(max_continuation_rounds + 1):
            state: Dict = {}
            for new_text in self._stream_once(input_ids, chunk_tokens, sampling, state):
                full_generated_text += new_text
                yield new_text

            output_ids = state["output_ids"]
            new_in_chunk = output_ids.shape[1] - input_ids.shape[1]
            hit_token_limit = new_in_chunk >= chunk_tokens and output_ids[0, -1].item() != self.tokenizer.eos_token_id
            if not auto_continue or not hit_token_limit or continuation_round == max_continuation_rounds:
                break
            if is_code_complete(extract_python_code(full_generated_text))[0] or is_stuck_in_repetition(full_generated_text):
                break

            generated_total = output_ids.shape[1] - prompt_length
            chunk_tokens = min(1024, max_total_tokens - generated_total, self.max_seq_length - output_ids.shape[1])
            if chunk_tokens <= 0:
                break
            input_ids = output_ids
