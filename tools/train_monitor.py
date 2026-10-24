"""
GPU watchdog + Telegram bot for unattended runs.

Watchdog (works without Telegram): if the GPU stays at or above --max-temp for --hot-minutes, the
current training run is stopped gracefully (STOP file -> checkpoint -> exit; resume with --resume).
If it is still running and above --kill-temp after --kill-after-minutes, the process is killed.

Telegram (optional, TELEGRAM_BOT_TOKEN in .env; PROXY_URL is used if set):
  /start  binds the bot to your chat (first chat only, or TELEGRAM_CHAT_ID from .env)
  /status GPU + pipeline + training progress      /log   last loss / eval lines
  /stop   stop the pipeline (training saves a checkpoint first)
Periodic summaries every --summary-hours, alerts on step changes, failures and overheating.

    uv run python tools/train_monitor.py            # foreground
    uv run python tools/train_monitor.py --detach   # via Task Scheduler, independent of the terminal
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "outputs" / "pipeline"
STATUS_FILE = PIPELINE_DIR / "status.json"
CHAT_FILE = PIPELINE_DIR / "telegram_chat_id"
MONITOR_LOG = PIPELINE_DIR / "monitor.log"
TASK_NAME = "FactorioLLM-monitor"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- System state -------------------------------------------------------------------------

def gpu_stats() -> Optional[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        t, p, pl, mu, mt, u = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"temp": int(float(t)), "power": float(p), "power_limit": float(pl),
                "mem_used": int(float(mu)), "mem_total": int(float(mt)), "util": int(float(u))}
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return None


def pipeline_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def tail_text(path: Path, n_bytes: int = 200_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n_bytes))
            return f.read().decode("utf-8", errors="replace").replace("\r", "\n")
    except FileNotFoundError:
        return ""


def training_progress(run: str) -> dict:
    text = tail_text(ROOT / "outputs" / run / "training.log")
    info: dict = {}
    bars = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([\d:]+),\s*([\d.]+)s/it\]", text)
    if bars:
        step, total, elapsed, eta, sec = bars[-1]
        info.update(step=int(step), total=int(total), elapsed=elapsed, eta=eta, sec_per_it=float(sec))
    losses = re.findall(r"\{'loss': '([\d.e-]+)'.*?'epoch': '([\d.]+)'\}", text)
    if losses:
        info["loss"], info["epoch"] = losses[-1]
    evals = re.findall(r"'eval_loss': '([\d.e-]+)'.*?'epoch': '([\d.]+)'", text)
    if evals:
        info["evals"] = evals[-3:]
    if re.search(r"Traceback|OutOfMemoryError|CUDA out of memory", text):
        info["error"] = True
    return info


def eval_progress(log_path: Path) -> str:
    cases = re.findall(r"\[(\d+)/(\d+)\]", tail_text(log_path, 20_000))
    return f"{cases[-1][0]}/{cases[-1][1]} cases" if cases else "starting"


def status_text() -> str:
    lines = [f"🖥 {datetime.now():%d.%m %H:%M}"]
    g = gpu_stats()
    if g:
        lines.append(f"GPU {g['temp']}°C, {g['power']:.0f}/{g['power_limit']:.0f} W, util {g['util']}%, "
                     f"VRAM {g['mem_used'] / 1024:.1f}/{g['mem_total'] / 1024:.1f} GB")
    st = pipeline_status()
    if not st:
        lines.append("Pipeline: not started")
        return "\n".join(lines)
    alive = pid_alive(st.get("pid"))
    lines.append(f"Pipeline: {st.get('state')}{'' if alive or st.get('state') != 'running' else ' (process NOT running!)'}")
    for s in st.get("steps", []):
        mark = {"done": "✅", "failed": "❌", "running": "⏳", "skipped": "⏭"}.get(s["state"], "•")
        lines.append(f"{mark} {s['name']}" + (f" (exit {s['exit_code']})" if s.get("exit_code") not in (None, 0) else ""))
    cur = st.get("current")
    if cur and cur.get("kind") == "train":
        p = training_progress(cur["run"])
        if "step" in p:
            lines.append(f"Train: step {p['step']}/{p['total']} ({100 * p['step'] / p['total']:.0f}%), "
                         f"elapsed {p['elapsed']}, ETA {p['eta']}, {p['sec_per_it']:.1f} s/it")
        if "loss" in p:
            lines.append(f"Loss {p['loss']} @ epoch {p['epoch']}")
        if p.get("evals"):
            lines.append("Eval loss: " + ", ".join(f"{l} (ep {e})" for l, e in p["evals"]))
    elif cur and cur.get("kind") == "eval":
        lines.append(f"Eval: {eval_progress(ROOT / cur['log'])}")
    return "\n".join(lines)


def last_log_lines() -> str:
    st = pipeline_status()
    cur = st.get("current") or (st.get("steps") or [{}])[-1]
    if cur.get("kind") == "train":
        text = tail_text(ROOT / "outputs" / cur["run"] / "training.log")
        picked = re.findall(r"\{'(?:loss|eval_loss)'.*?\}", text)[-6:]
    else:
        text = tail_text(ROOT / cur.get("log", "missing"), 20_000)
        picked = [l for l in text.splitlines() if l.strip()][-8:]
    return "\n".join(picked)[-3500:] or "(empty)"


def request_stop(reason: str) -> None:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    (PIPELINE_DIR / "ABORT").touch()
    cur = pipeline_status().get("current") or {}
    if cur.get("kind") == "train":
        (ROOT / "outputs" / cur["run"] / "STOP").touch()
    log(f"stop requested: {reason}")


# --- Telegram -----------------------------------------------------------------------------

class Telegram:
    def __init__(self, token: str, chat_id: Optional[str]):
        proxy = os.getenv("PROXY_URL")
        if proxy and proxy.startswith("socks5://"):
            proxy = proxy.replace("socks5://", "socks5h://", 1)
        self.base = f"https://api.telegram.org/bot{token}"
        self.client = httpx.Client(proxy=proxy or None, timeout=httpx.Timeout(40.0, connect=15.0))
        self.chat_id = chat_id or (CHAT_FILE.read_text().strip() if CHAT_FILE.exists() else None)
        self.offset = 0

    def send(self, text: str) -> None:
        if not self.chat_id:
            return
        try:
            self.client.post(f"{self.base}/sendMessage", json={"chat_id": self.chat_id, "text": text[:4000]})
        except Exception as e:
            log(f"telegram send failed: {type(e).__name__}")  # never log the URL: it contains the token

    def poll(self, handler) -> None:
        while True:
            try:
                r = self.client.get(f"{self.base}/getUpdates", params={"timeout": 30, "offset": self.offset})
                for upd in r.json().get("result", []):
                    self.offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    chat = str((msg.get("chat") or {}).get("id", ""))
                    text = (msg.get("text") or "").strip()
                    if not chat or not text:
                        continue
                    if self.chat_id is None and text.startswith("/start"):
                        self.chat_id = chat
                        CHAT_FILE.write_text(chat)
                        log(f"bound to chat {chat}")
                    if chat == self.chat_id:
                        handler(text.split()[0].split("@")[0].lower())
            except Exception as e:
                log(f"telegram poll failed: {type(e).__name__}")
                time.sleep(15)


# --- Main loop ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FactorioLLM GPU watchdog and Telegram bot")
    parser.add_argument("--max-temp", type=int, default=85)
    parser.add_argument("--hot-minutes", type=float, default=5)
    parser.add_argument("--kill-temp", type=int, default=90)
    parser.add_argument("--kill-after-minutes", type=float, default=10)
    parser.add_argument("--summary-hours", type=float, default=2)
    parser.add_argument("--interval", type=int, default=30, help="Seconds between GPU checks")
    parser.add_argument("--detach", action="store_true", help="Run via Windows Task Scheduler")
    args = parser.parse_args()

    if args.detach:
        sys.path.insert(0, str(ROOT / "tools"))
        from overnight import register_and_run
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        launcher = PIPELINE_DIR / "run_monitor.cmd"
        launcher.write_text(f'@echo off\r\ncd /d "{ROOT}"\r\n"{sys.executable}" tools\\train_monitor.py '
                            f'--max-temp {args.max_temp} --summary-hours {args.summary_hours} '
                            f'>> "{PIPELINE_DIR / "monitor.out"}" 2>&1\r\n', encoding="utf-8")
        register_and_run(TASK_NAME, launcher)
        print(f"[*] Started scheduled task '{TASK_NAME}'. Log: {MONITOR_LOG}")
        return

    load_dotenv(ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg = Telegram(token, os.getenv("TELEGRAM_CHAT_ID")) if token else None
    log(f"monitor started (telegram={'on' if tg else 'off'}, max_temp={args.max_temp})")

    def handle(cmd: str) -> None:
        if cmd in ("/start", "/status"):
            tg.send(status_text())
        elif cmd == "/log":
            tg.send(last_log_lines())
        elif cmd == "/stop":
            request_stop("telegram /stop")
            tg.send("🛑 Stop requested: training saves a checkpoint and the pipeline ends after the current step.")
        else:
            tg.send("/status — state, /log — last lines, /stop — stop the pipeline")

    if tg:
        threading.Thread(target=tg.poll, args=(handle,), daemon=True).start()
        tg.send("🤖 Monitor started\n" + status_text())

    hot = deque()
    stop_sent_at: Optional[float] = None
    last_summary = time.time()
    last_steps: dict = {}
    first_loop = True
    dead_reported = False
    while True:
        g = gpu_stats()
        st = pipeline_status()
        cur = st.get("current") or {}

        # step changes and failures
        for s in st.get("steps", []):
            prev = last_steps.get(s["name"])
            if prev != s["state"]:
                last_steps[s["name"]] = s["state"]
                if not first_loop:
                    icon = {"done": "✅", "failed": "❌", "running": "⏳", "skipped": "⏭"}.get(s["state"], "•")
                    msg = f"{icon} {s['name']}: {s['state']}"
                    log(msg)
                    if tg:
                        tg.send(msg + "\n" + status_text())
        if st.get("state") == "running" and st.get("pid") and not pid_alive(st["pid"]):
            if not dead_reported:
                dead_reported = True
                log("pipeline process is not running")
                if tg:
                    tg.send("⚠️ Pipeline process died unexpectedly\n" + status_text())

        # watchdog
        if g:
            hot.append((time.time(), g["temp"]))
            while hot and hot[0][0] < time.time() - args.hot_minutes * 60:
                hot.popleft()
            window_full = hot and hot[0][0] <= time.time() - (args.hot_minutes * 60 - args.interval * 2)
            if window_full and min(t for _, t in hot) >= args.max_temp and stop_sent_at is None:
                request_stop(f"GPU >= {args.max_temp}°C for {args.hot_minutes} min")
                stop_sent_at = time.time()
                if tg:
                    tg.send(f"🔥 GPU {g['temp']}°C for {args.hot_minutes} min: stopping training (checkpoint is saved).")
            if (stop_sent_at and time.time() - stop_sent_at > args.kill_after_minutes * 60
                    and g["temp"] >= args.kill_temp and cur.get("pid") and pid_alive(cur["pid"])):
                subprocess.run(["taskkill", "/PID", str(cur["pid"]), "/T", "/F"], capture_output=True)
                log(f"killed pid {cur['pid']} at {g['temp']}°C")
                if tg:
                    tg.send(f"☠️ GPU still {g['temp']}°C: training process killed.")

        if tg and time.time() - last_summary >= args.summary_hours * 3600:
            last_summary = time.time()
            tg.send(status_text())
        first_loop = False
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
