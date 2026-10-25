"""
Sandbox execution module for FactorioLLM.
Executes generated factorio-draftsman Python scripts in an isolated,
resource-limited Docker container with no network access.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


# Regular expression to extract Factorio blueprint string (starts with 0e)
BLUEPRINT_REGEX = re.compile(r"(0e[a-zA-Z0-9+/=]{20,})")

RESULT_MARKER = "__FACTORIOLLM_RESULT__"


@dataclass
class ExecutionResult:
    """Result of running a blueprint generator script inside the sandbox."""
    success: bool
    blueprint_string: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    execution_time: float = 0.0

    def get_feedback_for_model(self) -> str:
        """
        Formats error information into a concise feedback prompt for the LLM
        to trigger self-healing.
        """
        if self.success:
            return ""

        parts = []
        if self.error_message:
            parts.append(f"Error summary: {self.error_message}")
        if self.stderr.strip():
            # Truncate stderr if it's excessively long
            err_lines = self.stderr.strip().splitlines()
            if len(err_lines) > 25:
                err_text = "\n".join(err_lines[-25:])
            else:
                err_text = self.stderr.strip()
            parts.append(f"Traceback / Stderr:\n```\n{err_text}\n```")
        elif self.stdout.strip():
            parts.append(f"Stdout:\n```\n{self.stdout.strip()[:1000]}\n```")

        return "\n\n".join(parts)


# Dynamic compatibility shim to resolve synthetic entity class names from dataset
# (e.g., StoneFurnace -> Furnace, SmallLamp -> Lamp, MediumElectricPole -> ElectricPole)
COMPATIBILITY_SHIM = """
import re
import draftsman.entity as _d_ent

def _compat_getattr(name):
    if name in _d_ent.__dict__:
        return _d_ent.__dict__[name]
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\\1-\\2', name)
    kebab = re.sub(r'([a-z0-9])([A-Z])', r'\\1-\\2', s1).lower()
    try:
        return _d_ent.get_entity_class(kebab)
    except Exception:
        raise AttributeError(f"module 'draftsman.entity' has no attribute '{name}'")

_d_ent.__getattr__ = _compat_getattr
"""


def get_helpers_shim() -> str:
    """Returns Python code string injecting draftsman_helpers module and functions into sandbox."""
    helpers_file = Path(__file__).resolve().parent / "helpers.py"
    if helpers_file.exists():
        content = helpers_file.read_text(encoding="utf-8")
        return f"""
import sys as _sys, types as _types
_dh_mod = _types.ModuleType("draftsman_helpers")
exec({repr(content)}, _dh_mod.__dict__)
_sys.modules["draftsman_helpers"] = _dh_mod

import builtins as _builtins
for _k in ("add_belt_line", "add_underground_pair", "add_entity_row", "add_power_poles", "resolve_direction"):
    if hasattr(_dh_mod, _k):
        setattr(_builtins, _k, getattr(_dh_mod, _k))

# Pasted poles never wire themselves to each other in game: wire every exported blueprint.
if hasattr(_dh_mod, "connect_power"):
    from draftsman.blueprintable import Blueprint as _Blueprint
    _original_to_string = _Blueprint.to_string

    def _wired_to_string(self, *args, **kwargs):
        try:
            _dh_mod.connect_power(self)
        except Exception as _exc:  # never break an otherwise valid script
            print("[connect_power failed: %s]" % _exc, file=_sys.stderr)
        return _original_to_string(self, *args, **kwargs)

    _Blueprint.to_string = _wired_to_string
"""
    return ""


# Runs inside the container: executes each script in a forked child (fresh state, own timeout)
# and prints one JSON result line per script.
_BATCH_RUNNER = """
import contextlib as _ctx, io as _io, json as _json, os as _os, signal as _signal, sys as _sys, traceback as _tb

def _run_one(_code, _timeout):
    _r, _w = _os.pipe()
    _pid = _os.fork()
    if _pid == 0:
        _os.close(_r)
        _signal.alarm(_timeout)
        _out, _err = _io.StringIO(), None
        try:
            with _ctx.redirect_stdout(_out):
                exec(compile(_code, "script.py", "exec"), {"__name__": "__main__"})
        except BaseException:
            _err = _tb.format_exc(limit=8)
        with _os.fdopen(_w, "w") as _f:
            _f.write(_json.dumps({"stdout": _out.getvalue()[-200000:], "error": _err}))
        _os._exit(0)
    _os.close(_w)
    with _os.fdopen(_r) as _f:
        _data = _f.read()
    _os.waitpid(_pid, 0)
    if not _data:
        return {"stdout": "", "error": "Killed: per-script time or memory limit exceeded."}
    return _json.loads(_data)

for _i, _code in enumerate(_SCRIPTS):
    _res = _run_one(_code, _TIMEOUT)
    _res["i"] = _i
    _sys.__stdout__.write(_MARKER + _json.dumps(_res) + "\\n")
    _sys.__stdout__.flush()
"""


def validate_blueprint_string(bp_string: str) -> Optional[str]:
    """Returns an error message if the string is not a decodable blueprint, else None (size-limited decode)."""
    from factoriollm.blueprint_ir import BlueprintIRError, decode_blueprint_string

    try:
        data = decode_blueprint_string(bp_string)
    except BlueprintIRError as e:
        return str(e)
    if "blueprint" not in data:
        return "Decoded string is not a single blueprint."
    return None


class DockerSandbox:
    """
    Executes Python scripts inside an isolated Docker container with strict constraints:
    - Network: none (no internet or local LAN access)
    - Memory limit: 512MB (no swap)
    - CPU quota: 1 core
    - PID limit: 64 processes
    - Dropped capabilities: ALL, no-new-privileges, read-only root filesystem
    - Captured output capped; the container is killed on timeout
    """

    def __init__(
        self,
        image_name: str = "factoriollm-sandbox:latest",
        dockerfile_path: Optional[str | Path] = None,
        timeout: float = 12.0,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 64,
        max_output_bytes: int = 1_000_000,
    ):
        self.image_name = image_name
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.max_output_bytes = max_output_bytes
        # Snapshot of helpers.py: editing it during a long eval must not change what the eval runs.
        self.preamble = COMPATIBILITY_SHIM + "\n" + get_helpers_shim() + "\n"

        if dockerfile_path is None:
            # Default to docker/Dockerfile.sandbox relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.dockerfile_path = base_dir / "docker" / "Dockerfile.sandbox"
        else:
            self.dockerfile_path = Path(dockerfile_path)

    def is_docker_available(self) -> bool:
        """Check if the Docker CLI and daemon are available."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return res.returncode == 0
        except FileNotFoundError:
            return False

    def is_image_available(self) -> bool:
        """Check if the sandbox image is already built locally."""
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return res.returncode == 0
        except FileNotFoundError:
            return False

    def build_image(self) -> bool:
        """Build the sandbox Docker image from the Dockerfile."""
        if not self.dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {self.dockerfile_path}")

        print(f"[*] Building sandbox Docker image '{self.image_name}'...")
        cmd = [
            "docker", "build",
            "-t", self.image_name,
            "-f", str(self.dockerfile_path),
            str(self.dockerfile_path.parent.parent),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] Docker build failed:\n{res.stderr}")
            return False
        print(f"[*] Sandbox Docker image '{self.image_name}' built successfully.")
        return True

    def ensure_image(self) -> None:
        """Ensure the sandbox Docker image exists, building it if needed."""
        if not self.is_docker_available():
            raise RuntimeError("Docker is not installed or the Docker daemon is not running.")
        if not self.is_image_available():
            if not self.build_image():
                raise RuntimeError(f"Failed to build Docker sandbox image '{self.image_name}'.")

    def _docker_cmd(self, container_name: str) -> List[str]:
        return [
            "docker", "run",
            "--rm",
            "-i",
            "--name", container_name,
            "--network", "none",
            f"--memory={self.memory_limit}",
            f"--memory-swap={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--user", "1000:1000",
            self.image_name,
            "python", "-",
        ]

    def _preamble(self) -> str:
        return self.preamble

    def _run_container(self, script: str, timeout: float, max_output: int) -> tuple[Optional[int], str, str, bool]:
        """Runs a script in a fresh container. Returns (returncode, stdout, stderr, timed_out)."""
        container_name = f"factoriollm-{uuid.uuid4().hex[:12]}"
        proc = subprocess.Popen(
            self._docker_cmd(container_name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def _drain(stream, sink: bytearray) -> None:
            # Keep reading so the container never blocks on a full pipe, but store at most max_output bytes.
            while chunk := stream.read(65536):
                if len(sink) < max_output:
                    sink.extend(chunk[: max_output - len(sink)])

        def _feed() -> None:
            try:
                proc.stdin.write(script.encode("utf-8"))
                proc.stdin.close()
            except OSError:
                pass

        out_buf, err_buf = bytearray(), bytearray()
        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, out_buf), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err_buf), daemon=True),
            threading.Thread(target=_feed, daemon=True),
        ]
        for t in threads:
            t.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Killing the docker CLI does not stop the container: kill it by name.
            subprocess.run(["docker", "kill", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            proc.kill()
            proc.wait()
        for t in threads:
            t.join(timeout=5)
        proc.stdout.close()
        proc.stderr.close()

        decode = lambda b: b.decode("utf-8", errors="replace")
        return (None if timed_out else proc.returncode), decode(out_buf), decode(err_buf), timed_out

    @staticmethod
    def _result_from_output(stdout: str, stderr: str, exec_time: float) -> ExecutionResult:
        match = BLUEPRINT_REGEX.search(stdout)
        if not match:
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message="Script exited successfully (code 0), but no valid Factorio blueprint string (starting with '0e...') was found in output. Make sure the script calls 'print(bp.to_string())'.",
                execution_time=exec_time,
            )

        bp_string = match.group(1).strip()
        validation_error = validate_blueprint_string(bp_string)
        if validation_error:
            return ExecutionResult(
                success=False,
                blueprint_string=bp_string,
                stdout=stdout,
                stderr=stderr,
                error_message=f"Blueprint validation failed: {validation_error}",
                execution_time=exec_time,
            )
        return ExecutionResult(success=True, blueprint_string=bp_string, stdout=stdout, stderr=stderr, execution_time=exec_time)

    def execute(self, code: str, timeout: Optional[float] = None) -> ExecutionResult:
        """
        Executes the provided Python script string inside the Docker container.
        Pipes code via stdin directly to 'python -', avoiding any disk I/O on host.
        """
        self.ensure_image()
        timeout = float(timeout if timeout is not None else self.timeout)

        # Compile the generated code under its own filename so traceback line numbers match the model's code.
        script_to_run = self._preamble() + f"\nexec(compile({code!r}, 'script.py', 'exec'), {{'__name__': '__main__'}})\n"

        start_time = time.perf_counter()
        try:
            returncode, stdout, stderr, timed_out = self._run_container(script_to_run, timeout, self.max_output_bytes)
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"Subprocess error: {str(e)}",
                execution_time=time.perf_counter() - start_time,
            )
        exec_time = time.perf_counter() - start_time

        if timed_out:
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message=f"TimeoutExpired: Execution exceeded limit of {timeout}s (possible infinite loop).",
                execution_time=exec_time,
            )

        if returncode != 0:
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message=f"Script failed with exit code {returncode}.",
                execution_time=exec_time,
            )

        return self._result_from_output(stdout, stderr, exec_time)

    def execute_batch(self, scripts: Sequence[str], per_script_timeout: int = 10) -> List[ExecutionResult]:
        """
        Executes many scripts in ONE container (each in a forked child with its own timeout).
        Used for verifying datasets without paying the container start-up cost per script.
        """
        if not scripts:
            return []
        self.ensure_image()

        runner = (
            self._preamble()
            + f"\n_SCRIPTS = {json.dumps(list(scripts))!r}\n"
            + "import json as _j\n_SCRIPTS = _j.loads(_SCRIPTS)\n"
            + f"_TIMEOUT = {int(per_script_timeout)}\n_MARKER = {RESULT_MARKER!r}\n"
            + _BATCH_RUNNER
        )
        total_timeout = 30 + per_script_timeout * len(scripts)
        start_time = time.perf_counter()
        _, stdout, stderr, timed_out = self._run_container(runner, total_timeout, max_output=300_000 * len(scripts))
        elapsed = time.perf_counter() - start_time

        results: List[Optional[ExecutionResult]] = [None] * len(scripts)
        for line in stdout.splitlines():
            if not line.startswith(RESULT_MARKER):
                continue
            payload = json.loads(line[len(RESULT_MARKER):])
            i = payload["i"]
            if payload.get("error"):
                results[i] = ExecutionResult(success=False, stdout=payload.get("stdout", ""), stderr=payload["error"],
                                             error_message="Script raised an exception.")
            else:
                results[i] = self._result_from_output(payload.get("stdout", ""), "", 0.0)

        missing_reason = "Batch container timed out." if timed_out else f"Batch container failed: {stderr[-500:]}"
        per_script = elapsed / len(scripts)
        out: List[ExecutionResult] = []
        for r in results:
            r = r or ExecutionResult(success=False, error_message=missing_reason)
            r.execution_time = per_script
            out.append(r)
        return out


if __name__ == "__main__":
    # Self-test when run directly
    sandbox = DockerSandbox()
    test_code = """
from draftsman.blueprintable import Blueprint

bp = Blueprint()
bp.entities.append("wooden-chest", tile_position=(0, 0))
print(bp.to_string())
"""
    print("[*] Testing DockerSandbox with a simple wooden chest...")
    res = sandbox.execute(test_code)
    print(f"Success: {res.success}")
    print(f"Time: {res.execution_time:.3f}s")
    if res.success:
        print(f"Blueprint: {res.blueprint_string}")
    else:
        print(f"Error: {res.error_message}")
        print(f"Stderr: {res.stderr}")
