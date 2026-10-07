"""
Agentic self-healing execution loop for FactorioLLM.
Coordinates between LLM inference and the Docker sandbox:
1. Generates Python Draftsman code from prompt.
2. Runs code in isolated Docker container.
3. If error occurs, feeds traceback back to LLM to self-heal.
4. Returns verified Factorio blueprint string (0e...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Dict, Any, Optional

from factoriollm.inference import FactorioInference, extract_python_code, is_code_complete
from factoriollm.sandbox import DockerSandbox, ExecutionResult


@dataclass
class AgentStepEvent:
    """Represents a step in the agentic workflow for UI streaming."""
    event_type: str  # 'info', 'token', 'code', 'sandbox_start', 'sandbox_result', 'retry', 'success', 'failure'
    attempt: int
    message: str = ""
    code: str = ""
    execution_result: Optional[ExecutionResult] = None
    blueprint_string: Optional[str] = None


@dataclass
class AgentResult:
    """Final result of the agentic generation process."""
    success: bool
    blueprint_string: Optional[str] = None
    code: Optional[str] = None
    attempts_used: int = 0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_history: List[ExecutionResult] = field(default_factory=list)


class AgenticBlueprintGenerator:
    """
    Coordinates inference and sandbox execution with automatic error recovery.
    """

    def __init__(
        self,
        inference: Optional[FactorioInference] = None,
        sandbox: Optional[DockerSandbox] = None,
        max_retries: int = 3,
    ):
        self.inference = inference or FactorioInference()
        self.sandbox = sandbox or DockerSandbox()
        self.max_retries = max_retries

    def generate_blueprint_stream(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        temperature: float = 0.2,
    ) -> Generator[AgentStepEvent, None, AgentResult]:
        """
        Streaming generator that yields AgentStepEvent updates during the loop.
        """
        retries_limit = max_retries if max_retries is not None else self.max_retries

        # Ensure sandbox docker image is ready
        yield AgentStepEvent(
            event_type="info",
            attempt=1,
            message="Verifying Docker sandbox readiness...",
        )
        self.sandbox.ensure_image()

        history: List[Dict[str, str]] = [
            {"role": "user", "content": prompt}
        ]
        exec_history: List[ExecutionResult] = []

        for attempt in range(1, retries_limit + 1):
            yield AgentStepEvent(
                event_type="info",
                attempt=attempt,
                message=f"Attempt {attempt}/{retries_limit}: Generating Python script with LLM...",
            )

            # Generate code using inference model
            raw_generation = ""
            for token in self.inference.generate_stream(history, temperature=temperature):
                raw_generation += token
                yield AgentStepEvent(
                    event_type="token",
                    attempt=attempt,
                    message=token,
                )

            # Extract pure Python code
            code = extract_python_code(raw_generation)
            is_comp, comp_reason = is_code_complete(code)

            if not is_comp:
                # Script is incomplete, stalled, or truncated
                yield AgentStepEvent(
                    event_type="code",
                    attempt=attempt,
                    code=code,
                    message=f"Generation stopped before script completed: {comp_reason}",
                )

                exec_result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    error_message=f"Generation incomplete: {comp_reason}",
                )
                exec_history.append(exec_result)

                if attempt < retries_limit:
                    retry_prompt = (
                        f"Generation failed: The script was incomplete and truncated ({comp_reason}).\n"
                        f"This occurred because repetitive entity placements exceeded the output token budget.\n\n"
                        f"Please rewrite the entire Python script from scratch using compact, algorithmic code (such as 'for' loops and mathematical expressions) for all repeating structures.\n"
                        f"Requirements:\n"
                        f"- Output only the complete standalone Python script from imports to print(bp.to_string()).\n"
                        f"- Do not output continuation fragments; start directly with the imports.\n"
                        f"- Keep the script concise and algorithmic."
                    )

                    # Sanitize history to prevent continuation hallucinations (like "'y': 7.0}, direction=4))")
                    history = [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": "# [Note: Previous generation was truncated due to output token limit]",
                        },
                        {"role": "user", "content": retry_prompt},
                    ]

                    yield AgentStepEvent(
                        event_type="retry",
                        attempt=attempt,
                        code=code,
                        execution_result=exec_result,
                        message=f"Attempt {attempt} incomplete ({comp_reason}). Requesting loop-optimized rewrite from scratch (attempt {attempt + 1})...",
                    )
                    continue
                else:
                    yield AgentStepEvent(
                        event_type="failure",
                        attempt=attempt,
                        code=code,
                        execution_result=exec_result,
                        message=f"Retries exhausted ({retries_limit}). Script remained incomplete.",
                    )
                    return AgentResult(
                        success=False,
                        code=code,
                        attempts_used=retries_limit,
                        conversation_history=history,
                        error_message=exec_result.error_message,
                        execution_history=exec_history,
                    )

            # Code is complete and syntactically valid! Proceed to Docker sandbox.
            yield AgentStepEvent(
                event_type="code",
                attempt=attempt,
                code=code,
                message=f"Extracted verified Python code ({len(code.splitlines())} lines).",
            )

            # Run in sandbox
            yield AgentStepEvent(
                event_type="sandbox_start",
                attempt=attempt,
                code=code,
                message=f"Executing script inside Docker sandbox (timeout {self.sandbox.timeout}s)...",
            )

            exec_result = self.sandbox.execute(code)
            exec_history.append(exec_result)

            yield AgentStepEvent(
                event_type="sandbox_result",
                attempt=attempt,
                code=code,
                execution_result=exec_result,
                message="Sandbox execution finished.",
            )

            if exec_result.success:
                yield AgentStepEvent(
                    event_type="success",
                    attempt=attempt,
                    code=code,
                    blueprint_string=exec_result.blueprint_string,
                    execution_result=exec_result,
                    message=f"Success! Working Factorio blueprint compiled in {exec_result.execution_time:.2f}s.",
                )
                return AgentResult(
                    success=True,
                    blueprint_string=exec_result.blueprint_string,
                    code=code,
                    attempts_used=attempt,
                    conversation_history=history,
                    execution_history=exec_history,
                )

            # If failed in sandbox and attempts remain: self-heal
            if attempt < retries_limit:
                feedback = exec_result.get_feedback_for_model()
                retry_prompt = (
                    f"Executing your Python script in the Factorio sandbox failed with the following error:\n"
                    f"{feedback}\n\n"
                    f"Please analyze the error traceback carefully. Fix the issue in the Python code using the factorio-draftsman library.\n"
                    f"Important instructions:\n"
                    f"- Write compact, algorithmic code using Python for-loops instead of repeating lines.\n"
                    f"- Ensure correct entity class names, valid positions, and proper coordinates.\n"
                    f"- The script MUST conclude with print(bp.to_string()). Output only the corrected complete Python script."
                )

                # Keep a bounded sliding window: original prompt + last failed code + last error
                history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": retry_prompt},
                ]

                yield AgentStepEvent(
                    event_type="retry",
                    attempt=attempt,
                    code=code,
                    execution_result=exec_result,
                    message=f"Execution error encountered. Traceback forwarded to model for self-healing (attempt {attempt + 1})...",
                )
            else:
                yield AgentStepEvent(
                    event_type="failure",
                    attempt=attempt,
                    code=code,
                    execution_result=exec_result,
                    message=f"Retries exhausted ({retries_limit}). Unable to automatically recover from errors.",
                )

        return AgentResult(
            success=False,
            code=code,
            attempts_used=retries_limit,
            conversation_history=history,
            error_message=exec_history[-1].error_message if exec_history else "Unknown error",
            execution_history=exec_history,
        )

    def generate_blueprint(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        temperature: float = 0.2,
    ) -> AgentResult:
        """Synchronous helper that runs the generator loop to completion."""
        gen = self.generate_blueprint_stream(prompt, max_retries, temperature)
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value
        return result
