"""
Sandbox execution module for FactorioLLM.
Executes generated factorio-draftsman Python scripts in an isolated,
resource-limited Docker container with no network access.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Regular expression to extract Factorio blueprint string (starts with 0e)
BLUEPRINT_REGEX = re.compile(r"(0e[a-zA-Z0-9+/=]{20,})")


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
"""
    return ""


class DockerSandbox:
    """
    Executes Python scripts inside an isolated Docker container with strict constraints:
    - Network: none (no internet or local LAN access)
    - Memory limit: 512MB
    - CPU quota: 1 core
    - PID limit: 64 processes
    - Dropped capabilities: ALL
    """

    def __init__(
        self,
        image_name: str = "factoriollm-sandbox:latest",
        dockerfile_path: Optional[str | Path] = None,
        timeout: float = 12.0,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 64,
    ):
        self.image_name = image_name
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit

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

    def execute(self, code: str) -> ExecutionResult:
        """
        Executes the provided Python script string inside the Docker container.
        Pipes code via stdin directly to 'python -', avoiding any disk I/O on host.
        """
        self.ensure_image()

        docker_cmd = [
            "docker", "run",
            "--rm",
            "-i",
            "--network", "none",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            self.image_name,
            "python", "-",
        ]

        # Prepend compatibility shim and high-level helpers
        script_to_run = COMPATIBILITY_SHIM + "\n" + get_helpers_shim() + "\n" + code

        start_time = time.perf_counter()
        try:
            proc = subprocess.run(
                docker_cmd,
                input=script_to_run,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            exec_time = time.perf_counter() - start_time
        except subprocess.TimeoutExpired as e:
            exec_time = time.perf_counter() - start_time
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message=f"TimeoutExpired: Execution exceeded limit of {self.timeout}s (possible infinite loop).",
                execution_time=exec_time,
            )
        except Exception as e:
            exec_time = time.perf_counter() - start_time
            return ExecutionResult(
                success=False,
                error_message=f"Subprocess error: {str(e)}",
                execution_time=exec_time,
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Check return code
        if proc.returncode != 0:
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message=f"Script failed with exit code {proc.returncode}.",
                execution_time=exec_time,
            )

        # Look for blueprint string in stdout
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

        # Validate blueprint string locally using Draftsman
        try:
            from draftsman.blueprintable import Blueprint
            try:
                Blueprint.from_string(bp_string)
            except AttributeError:
                Blueprint(bp_string)
        except Exception as e:
            return ExecutionResult(
                success=False,
                blueprint_string=bp_string,
                stdout=stdout,
                stderr=stderr,
                error_message=f"Draftsman Blueprint validation failed: {str(e)}",
                execution_time=exec_time,
            )

        return ExecutionResult(
            success=True,
            blueprint_string=bp_string,
            stdout=stdout,
            stderr=stderr,
            execution_time=exec_time,
        )


if __name__ == "__main__":
    # Self-test when run directly
    sandbox = DockerSandbox()
    test_code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import Container

bp = Blueprint()
bp.entities.append(Container("wooden-chest", tile_position=(0, 0)))
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
