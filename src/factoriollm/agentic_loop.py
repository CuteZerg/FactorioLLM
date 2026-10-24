"""
Agentic self-healing execution loop for FactorioLLM.
Coordinates between LLM inference and the Docker sandbox:
1. Generates Python Draftsman code from prompt.
2. Runs code in isolated Docker container.
3. Validates the resulting blueprint (validator.py): errors are things that never work in game.
4. On an execution or validation error, feeds the traceback / issue list back to the LLM to self-heal.
5. Returns verified Factorio blueprint string (0e...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Dict, Any, Optional

from factoriollm.inference import FactorioInference, extract_python_code, is_code_complete
from factoriollm.sandbox import DockerSandbox, ExecutionResult
from factoriollm.validator import Issue, ValidationReport, validate_string


@dataclass
class AgentStepEvent:
    """Represents a step in the agentic workflow for UI streaming."""
    event_type: str  # 'info', 'token', 'code', 'sandbox_start', 'sandbox_result', 'retry', 'success', 'failure'
    attempt: int
    message: str = ""
    code: str = ""
    execution_result: Optional[ExecutionResult] = None
    blueprint_string: Optional[str] = None
    validation: Optional[ValidationReport] = None


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
    validation: Optional[ValidationReport] = None


class AgenticBlueprintGenerator:
    """
    Coordinates inference and sandbox execution with automatic error recovery.
    """

    def __init__(
        self,
        inference: Optional[FactorioInference] = None,
        sandbox: Optional[DockerSandbox] = None,
        max_retries: int = 3,
        validate: bool = True,
    ):
        self.inference = inference or FactorioInference()
        self.sandbox = sandbox or DockerSandbox()
        self.max_retries = max_retries
        self.validate = validate

    @staticmethod
    def _validate(blueprint_string: str) -> ValidationReport:
        try:
            return validate_string(blueprint_string)
        except Exception as e:  # undecodable / unknown entities: report instead of crashing the loop
            return ValidationReport(errors=[Issue("unparseable", f"Blueprint cannot be analysed: {e}", (0, 0))])

    def generate_blueprint_stream(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        temperature: float = 0.2,
        sandbox_timeout: Optional[float] = None,
    ) -> Generator[AgentStepEvent, None, AgentResult]:
        """
        Streaming generator that yields AgentStepEvent updates during the loop.
        """
        retries_limit = max_retries if max_retries is not None else self.max_retries
        timeout = sandbox_timeout if sandbox_timeout is not None else self.sandbox.timeout

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
                message=f"Executing script inside Docker sandbox (timeout {timeout}s)...",
            )

            exec_result = self.sandbox.execute(code, timeout=timeout)
            exec_history.append(exec_result)

            yield AgentStepEvent(
                event_type="sandbox_result",
                attempt=attempt,
                code=code,
                execution_result=exec_result,
                message="Sandbox execution finished.",
            )

            validation = None
            if exec_result.success and self.validate:
                validation = self._validate(exec_result.blueprint_string)
                if not validation.ok:
                    exec_result = ExecutionResult(
                        success=False,
                        blueprint_string=exec_result.blueprint_string,
                        # no stdout: it holds the blueprint string, which is noise in the retry prompt
                        error_message="The blueprint compiles but cannot work in game:\n" + validation.feedback(),
                        execution_time=exec_result.execution_time,
                    )
                    exec_history[-1] = exec_result

            if exec_result.success:
                n_warn = len(validation.warnings) if validation else 0
                yield AgentStepEvent(
                    event_type="success",
                    attempt=attempt,
                    code=code,
                    blueprint_string=exec_result.blueprint_string,
                    execution_result=exec_result,
                    validation=validation,
                    message=f"Success! Working Factorio blueprint compiled in {exec_result.execution_time:.2f}s"
                    + (f" ({n_warn} validator warnings)." if n_warn else "."),
                )
                return AgentResult(
                    success=True,
                    blueprint_string=exec_result.blueprint_string,
                    code=code,
                    attempts_used=attempt,
                    conversation_history=history,
                    execution_history=exec_history,
                    validation=validation,
                )

            # If failed in sandbox / validation and attempts remain: self-heal
            if attempt < retries_limit:
                feedback = exec_result.get_feedback_for_model()
                if validation is not None:
                    retry_prompt = (
                        f"Your script runs, but the blueprint it builds cannot work in Factorio:\n"
                        f"{feedback}\n\n"
                        f"Fix these issues (positions are integer tiles of the entity's top-left corner). "
                        f"Output only the corrected complete Python script ending with print(bp.to_string())."
                    )
                else:
                    retry_prompt = (
                        f"Executing your Python script in the Factorio sandbox failed with the following error:\n"
                        f"{feedback}\n\n"
                        f"Please analyze the error traceback carefully. Fix the issue in the Python code using the factorio-draftsman library.\n"
                        f"Important instructions:\n"
                        f"- Write compact, algorithmic code using Python for-loops instead of repeating lines.\n"
                        f"- Ensure correct entity names, valid integer tile positions and directions.\n"
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
        sandbox_timeout: Optional[float] = None,
    ) -> AgentResult:
        """Synchronous helper that runs the generator loop to completion."""
        gen = self.generate_blueprint_stream(prompt, max_retries, temperature, sandbox_timeout)
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value
        return result
