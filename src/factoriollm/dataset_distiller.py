"""
Dataset Distiller & Teacher-Guided Refactorer for FactorioLLM.
Uses an advanced Teacher LLM (Gemini 3.8 Flash) to refactor unrolled repetitive
Draftsman scripts into concise, DRY, idiomatic Python code with loops and macros.

Every refactored script is strictly verified in the Draftsman sandbox for 100%
mathematical entity equivalence before being saved into the training dataset.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
import httpx
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from dotenv import load_dotenv

# Ensure environment variables (API key and proxy) are loaded
load_dotenv()
proxy = os.getenv("PROXY_URL")
if proxy:
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["ALL_PROXY"] = proxy

from google import genai
from google.genai import types

from factoriollm.dataset_lifter import (
    execute_script_to_blueprint,
    verify_code_equivalence,
    lift_blueprint_to_code,
)
from factoriollm.inference import DEFAULT_SYSTEM_PROMPT


TEACHER_SYSTEM_PROMPT = """You are an expert Factorio Draftsman Python Code Refactorer.
Your task is to refactor low-level factorio-draftsman scripts into clean, algorithmic, DRY (Don't Repeat Yourself) Python code.

Architecture & High-level Macros available from draftsman_helpers:
from draftsman.blueprintable import Blueprint
from draftsman_helpers import add_belt_line, add_underground_pair, add_entity_row, add_power_poles

Macro Signatures:
- add_belt_line(bp, start=(x, y), length=N, direction=dir, belt_type=name)
- add_underground_pair(bp, start=(x1, y1), end=(x2, y2), direction=dir, belt_type=name)
- add_entity_row(bp, entity_name, start=(x, y), count=N, step=(dx, dy), direction=dir, **kwargs)
- add_power_poles(bp, start=(x, y), count=N, step=(dx, dy), pole_type=name)

Core Refactoring Rules:
1. DRY Principle: NEVER output long sequences of repetitive copy-pasted lines with shifted coordinates.
   - If underground pairs repeat at regular intervals, group them into a loop:
     for y in (6.0, 9.0, 12.0):
         add_underground_pair(bp, start=(5.0, y), end=(5.0, y - 2.0), direction=0)
   - If belt runs vary, store clean parameter tuples and iterate:
     belt_runs = [((3.0, 32.0), 11, 0), ((3.0, 19.0), 3, 0), ...]
     for start_pos, length, direction in belt_runs:
         add_belt_line(bp, start=start_pos, length=length, direction=direction)
   - For entity rows, use add_entity_row(...) or standard Python for loops.
2. 100% Mathematical Precision: The refactored code MUST build the EXACT SAME blueprint entities (identical entity prototypes, identical coordinates, identical directions and recipes) as the original script.
3. Keep code compact, readable, and idiomatic.
4. Output ONLY valid Python code inside a ```python ``` markdown block. Always end with: print(bp.to_string())
"""


def create_gemini_client() -> genai.Client:
    """Initializes the Gemini API client with custom connection pool and optimized proxy routing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing from .env")

    proxy_addr = os.getenv("PROXY_URL")
    if proxy_addr and proxy_addr.startswith("socks5://"):
        proxy_addr = proxy_addr.replace("socks5://", "socks5h://")

    custom_http_client = httpx.Client(
        proxy=proxy_addr if proxy_addr else None,
        timeout=httpx.Timeout(connect=25.0, read=90.0, write=30.0, pool=30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
    )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(httpx_client=custom_http_client),
    )


def needs_distillation(code: str) -> bool:
    """Determines whether a code snippet contains unrolled repetitions that need refactoring."""
    macro_calls = sum(code.count(m) for m in ["add_belt_line", "add_underground_pair", "add_entity_row", "add_power_poles"])
    appends = code.count(".append(")
    return macro_calls > 4 or appends > 15


def refactor_script_with_teacher(
    client: genai.Client,
    orig_code: str,
    prompt_text: str,
    model_name: str = "gemini-3.8-flash",
    max_retries: int = 3,
) -> Optional[str]:
    """Calls Teacher LLM without thinking tokens for maximum distillation speed with backoff retries."""
    user_msg = f"Refactor this Factorio Draftsman script for request '{prompt_text}':\n\n```python\n{orig_code}\n```"
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=TEACHER_SYSTEM_PROMPT,
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            full_text = (resp.text or "").strip()
            m = re.search(r"```(?:python)?\s*\n(.*?)```", full_text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            if "from draftsman" in full_text:
                return full_text
            return None
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                wait_sec = (attempt + 1) * 3
                time.sleep(wait_sec)
                continue
            elif "disconnect" in err_str or "timeout" in err_str or "connection" in err_str:
                time.sleep(1)
                continue
            else:
                return None
    return None


def process_single_record(
    client: genai.Client,
    record: Dict[str, Any],
    idx: int,
) -> Tuple[int, Dict[str, Any], str]:
    """
    Processes, refactors, and mathematically verifies a single dataset record.
    Returns (idx, record, status).
    status: 'compact' | 'refactored' | 'fallback_unverified' | 'fallback_error'
    """
    messages = record.get("messages", [])
    if len(messages) < 3:
        return idx, record, "compact"

    messages[0]["content"] = DEFAULT_SYSTEM_PROMPT
    prompt_text = messages[1].get("content", "")
    orig_code = messages[2].get("content", "")

    if not needs_distillation(orig_code):
        return idx, record, "compact"

    refactored_code = refactor_script_with_teacher(client, orig_code, prompt_text)
    if refactored_code:
        # Verify 100% equivalence in Draftsman
        try:
            if verify_code_equivalence(orig_code, refactored_code):
                messages[2]["content"] = refactored_code
                return idx, record, "refactored"
            else:
                return idx, record, "fallback_unverified"
        except Exception:
            return idx, record, "fallback_unverified"

    # If refactoring fails, keep safe original
    return idx, record, "fallback_error"


def distill_dataset(
    input_path: str = "data/dataset_v2.jsonl",
    output_path: str = "data/dataset_v3.jsonl",
    limit: Optional[int] = None,
    max_workers: int = 4,
    save_interval: int = 10,
) -> None:
    """
    Batched, multithreaded distillation pipeline with strictly ordered in-place writing
    and automatic checkpoint resuming.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    client = create_gemini_client()

    # Determine already processed records for resuming
    processed_count = 0
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for _ in f:
                processed_count += 1
    if processed_count > 0:
        print(f"[*] Resuming from record {processed_count}...")

    # Read remaining input lines
    lines_to_process: List[Tuple[int, str]] = []
    with open(in_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < processed_count:
                continue
            if limit is not None and len(lines_to_process) >= limit:
                break
            line_str = line.strip()
            if line_str:
                lines_to_process.append((idx, line_str))

    total = len(lines_to_process)
    if total == 0:
        print("[*] All records already distilled!")
        return

    print(f"[*] Starting distillation of {total} records with {max_workers} parallel workers...")
    print(f"[*] Writing to {output_path} (resumed at record index {processed_count})...")
    
    out_handle = open(out_file, "a", encoding="utf-8")
    start_time = time.time()
    
    # State for strictly in-order streaming write
    next_write_idx = processed_count
    pending_results: Dict[int, Tuple[Dict[str, Any], str]] = {}
    
    stats = {
        "refactored": 0,
        "compact": 0,
        "fallback_unverified": 0,
        "fallback_error": 0,
    }
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_single_record,
                    client,
                    json.loads(line_str),
                    idx,
                )
                for idx, line_str in lines_to_process
            ]

            completed = 0
            for future in as_completed(futures):
                idx, rec, status = future.result()
                stats[status] = stats.get(status, 0) + 1
                pending_results[idx] = (rec, status)
                completed += 1

                # Flush all consecutive completed records in exact order
                while next_write_idx in pending_results:
                    write_rec, _ = pending_results.pop(next_write_idx)
                    out_handle.write(json.dumps(write_rec, ensure_ascii=False) + "\n")
                    next_write_idx += 1

                if completed % save_interval == 0 or completed == total:
                    out_handle.flush()
                    elapsed = time.time() - start_time
                    speed = completed / max(elapsed, 1e-4)
                    eta = (total - completed) / max(speed, 1e-4)
                    ref = stats["refactored"]
                    cmp = stats["compact"]
                    fb = stats["fallback_unverified"] + stats["fallback_error"]
                    print(
                        f"[{completed}/{total}] "
                        f"Refactored: {ref} | AlreadyCompact: {cmp} | Fallback: {fb} | "
                        f"Speed: {speed:.1f} rec/s | ETA: {eta/60:.1f}m"
                    )
    finally:
        # Flush any remaining results
        while next_write_idx in pending_results:
            write_rec, _ = pending_results.pop(next_write_idx)
            out_handle.write(json.dumps(write_rec, ensure_ascii=False) + "\n")
            next_write_idx += 1
        out_handle.flush()
        out_handle.close()

    print(f"[*] Dataset distillation finished! Total records written: {next_write_idx}")
    print(f"[*] Final Stats: Refactored={stats['refactored']}, Compact={stats['compact']}, Fallback={stats['fallback_unverified'] + stats['fallback_error']}")


def main():
    parser = argparse.ArgumentParser(description="Teacher-Guided Dataset Distiller for FactorioLLM")
    parser.add_argument("--input", default="data/dataset_v2.jsonl", help="Input dataset path")
    parser.add_argument("--output", default="data/dataset_v3.jsonl", help="Output dataset path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--interval", type=int, default=10, help="Log progress interval")
    args = parser.parse_args()

    distill_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_workers=args.workers,
        save_interval=args.interval,
    )


if __name__ == "__main__":
    main()
