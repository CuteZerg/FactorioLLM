"""
Unattended overnight pipeline: runs evaluation / training steps one after another and records
progress in outputs/pipeline/status.json (read by tools/train_monitor.py).

    uv run python tools/overnight.py            # run in the foreground
    uv run python tools/overnight.py --detach   # run via Windows Task Scheduler, independent of this terminal
    uv run python tools/overnight.py --abort    # stop after the current step (training stops gracefully)

Steps (pilot -> full run):
  1. agentic eval of adapters/pilot-qwen7b (3 attempts)
  2. training on the full dataset, 2 epochs  -> adapters/full-qwen7b-2ep
  3. single-attempt eval of the new adapter (comparable with the pilot report)
  4. agentic eval of the new adapter (3 attempts)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "outputs" / "pipeline"
STATUS_FILE = PIPELINE_DIR / "status.json"
ABORT_FILE = PIPELINE_DIR / "ABORT"
TASK_NAME = "FactorioLLM-overnight"

FULL_RUN = "full-qwen7b-2ep"
EVAL_ARGS = ["--set", "all", "--limit", "50", "--max-new-tokens", "2048"]

STEPS = [
    {"name": "eval-pilot-retries3", "kind": "eval",
     "cmd": ["-m", "factoriollm.eval_suite", "--adapter", "adapters/pilot-qwen7b", *EVAL_ARGS, "--retries", "3"]},
    {"name": f"train-{FULL_RUN}", "kind": "train", "run": FULL_RUN,
     "cmd": ["src/factoriollm/train_model.py", "--run-name", FULL_RUN, "--epochs", "2"]},
    {"name": f"eval-{FULL_RUN}", "kind": "eval", "needs_adapter": FULL_RUN,
     "cmd": ["-m", "factoriollm.eval_suite", "--adapter", f"adapters/{FULL_RUN}", *EVAL_ARGS]},
    {"name": f"eval-{FULL_RUN}-retries3", "kind": "eval", "needs_adapter": FULL_RUN,
     "cmd": ["-m", "factoriollm.eval_suite", "--adapter", f"adapters/{FULL_RUN}", *EVAL_ARGS, "--retries", "3"]},
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_status(status: dict) -> None:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATUS_FILE)


def run_pipeline() -> int:
    ABORT_FILE.unlink(missing_ok=True)
    status = {"pid": os.getpid(), "started": now(), "state": "running", "current": None, "steps": []}
    write_status(status)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    for i, step in enumerate(STEPS, 1):
        if ABORT_FILE.exists():
            status["state"] = "aborted"
            break
        entry = {"name": step["name"], "kind": step["kind"], "run": step.get("run"), "started": now(), "state": "running"}
        adapter = step.get("needs_adapter")
        if adapter and not (ROOT / "adapters" / adapter / "adapter_model.safetensors").exists():
            entry.update(state="skipped", reason=f"adapters/{adapter} not found", finished=now())
            status["steps"].append(entry)
            write_status(status)
            continue

        log_path = PIPELINE_DIR / f"{i:02d}-{step['name']}.log"
        entry["log"] = str(log_path.relative_to(ROOT))
        status["current"] = entry
        status["steps"].append(entry)
        write_status(status)

        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, *step["cmd"]], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry["pid"] = proc.pid
            write_status(status)
            code = proc.wait()

        entry.update(finished=now(), exit_code=code, state="done" if code == 0 else "failed")
        status["current"] = None
        write_status(status)
        if code != 0 and step["kind"] == "train":
            status["state"] = "failed"
            write_status(status)
            return code

    if status["state"] == "running":
        status["state"] = "done"
    status["finished"] = now()
    write_status(status)
    return 0


def detach() -> None:
    """Registers and starts a one-off Task Scheduler job so the pipeline survives closing the terminal / IDE."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    launcher = PIPELINE_DIR / "run_overnight.cmd"
    launcher.write_text(
        "@echo off\r\n"
        f'cd /d "{ROOT}"\r\n'
        f'"{sys.executable}" tools\\overnight.py >> "{PIPELINE_DIR / "overnight.out"}" 2>&1\r\n',
        encoding="utf-8",
    )
    register_and_run(TASK_NAME, launcher)
    print(f"[*] Started scheduled task '{TASK_NAME}'. Status: {STATUS_FILE}")


def register_and_run(task_name: str, launcher: Path) -> None:
    subprocess.run(["schtasks", "/Create", "/TN", task_name, "/TR", f'"{launcher}"', "/SC", "ONCE",
                    "/ST", "23:59", "/F"], check=True, capture_output=True)
    subprocess.run(["schtasks", "/Run", "/TN", task_name], check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Unattended FactorioLLM pipeline")
    parser.add_argument("--detach", action="store_true", help="Run via Windows Task Scheduler")
    parser.add_argument("--abort", action="store_true", help="Stop after the current step; training stops gracefully")
    args = parser.parse_args()
    if args.detach:
        detach()
    elif args.abort:
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        ABORT_FILE.touch()
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8")) if STATUS_FILE.exists() else {}
        current = status.get("current") or {}
        if current.get("kind") == "train":
            (ROOT / "outputs" / current["run"] / "STOP").touch()
        print("[*] Abort requested")
    else:
        sys.exit(run_pipeline())


if __name__ == "__main__":
    main()
