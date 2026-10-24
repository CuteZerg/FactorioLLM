"""
Export (v4 pipeline, stage 3): canonical scripts + grounded prompts -> chat-format JSONL.

- The system prompt is inference.DEFAULT_SYSTEM_PROMPT, so training and inference match.
- Optional rotation augmentation: every (prompt, blueprint) pair gets one deterministic rotation
  (0/90/180/270), direction words in the prompt are rewritten accordingly, and the rotated code
  is re-lifted and verified. Each prompt still maps to exactly one answer.
- Train/eval split is done by blueprint key, so prompts of one blueprint never leak across splits.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from factoriollm.blueprint_ir import IRBlueprint, IREntity, rotated
from factoriollm.canonical_builder import make_token_counter
from factoriollm.database import get_pool
from factoriollm.dataset_lifter import lift_and_verify_locally
from factoriollm.inference import DEFAULT_SYSTEM_PROMPT
from factoriollm.prompt_generator import count_conflicts, normalize_prompt, prompt_language

DATA_DIR = Path("data")

# Clockwise order of compass words; rotating by k quarter-turns shifts each word by k.
_EN_DIRECTIONS = ["north", "east", "south", "west"]
_RU_STEMS = [  # noun and adjective stems: север/северн, восток/восточн, юг/южн, запад/западн
    ("северн", "восточн", "южн", "западн"),
    ("север", "восток", "юг", "запад"),
]
_RU_PATTERN = re.compile(r"(?<![а-яё])(" + "|".join(s for stems in _RU_STEMS for s in stems) + ")", re.IGNORECASE)


def _stable_fraction(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def rotate_direction_words(prompt: str, k: int) -> str:
    """Rewrites compass directions in an EN/RU prompt for a clockwise rotation by k quarter-turns."""
    if k % 4 == 0:
        return prompt

    def _en(m: re.Match) -> str:
        word = m.group(0)
        new = _EN_DIRECTIONS[(_EN_DIRECTIONS.index(word.lower()) + k) % 4]
        return new.capitalize() if word[0].isupper() else new

    def _ru(m: re.Match) -> str:
        word = m.group(0)
        stems = next(s for s in _RU_STEMS if word.lower() in s)
        new = stems[(stems.index(word.lower()) + k) % 4]
        return new.capitalize() if word[0].isupper() else new

    out = re.sub(r"\b(north|east|south|west)\b", _en, prompt, flags=re.IGNORECASE)
    # One pass over all stems (adjective stems first) so no word is rotated twice.
    return _RU_PATTERN.sub(_ru, out)


def ir_from_json(data: List[list]) -> IRBlueprint:
    return IRBlueprint([IREntity(n, x, y, d, tuple(sorted(extras.items()))) for n, x, y, d, extras in data])


def build_record(prompt: str, code: str) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
        ]
    }


async def export_dataset_to_jsonl(
    output_file: str = "dataset_v4.jsonl",
    eval_file: str = "dataset_v4_eval.jsonl",
    eval_fraction: float = 0.03,
    rotate: bool = True,
    max_seq_length: int = 4096,
) -> Dict[str, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.blueprint_key, c.canonical_code, c.ir, c.facts, p.prompt
            FROM canonical_blueprints c JOIN generated_prompts p ON p.canonical_id = c.id
            WHERE c.status = 'ok'
            ORDER BY c.id, p.id
            """
        )
    await pool.close()

    count_tokens = make_token_counter()
    system_tokens = count_tokens(DEFAULT_SYSTEM_PROMPT)
    stats: Counter = Counter()
    lengths: List[int] = []
    rotated_cache: Dict[Tuple[int, int], str] = {}
    splits = {"train": [], "eval": []}

    for r in rows:
        split = "eval" if _stable_fraction(r["blueprint_key"]) < eval_fraction else "train"
        prompt, code = normalize_prompt(r["prompt"]), r["canonical_code"]
        if count_conflicts(prompt, json.loads(r["facts"])):
            stats["dropped_count_conflict"] += 1
            continue
        lang = prompt_language(prompt)
        k = int(_stable_fraction(f"{r['id']}|{prompt}") * 4) % 4 if rotate else 0
        if k:
            key = (r["id"], k)
            if key not in rotated_cache:
                rotated_cache[key] = lift_and_verify_locally(rotated(ir_from_json(json.loads(r["ir"])), k)) or ""
            if rotated_cache[key]:
                prompt, code = rotate_direction_words(prompt, k), rotated_cache[key]
                stats["rotated"] += 1
            else:
                stats["rotation_failed"] += 1

        n_tokens = system_tokens + count_tokens(prompt) + count_tokens(code) + 30  # + chat template overhead
        if n_tokens > max_seq_length:
            stats["too_long"] += 1
            continue
        lengths.append(n_tokens)
        splits[split].append(build_record(prompt, code))
        stats[f"{split}_{lang}"] += 1

    for split, name in (("train", output_file), ("eval", eval_file)):
        with open(DATA_DIR / name, "w", encoding="utf-8") as f:
            for rec in splits[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[*] {split}: {len(splits[split])} records -> {DATA_DIR / name}")

    if lengths:
        q = sorted(lengths)
        print(f"[*] Tokens per example: p50={q[len(q) // 2]}, p90={q[int(len(q) * 0.9)]}, max={q[-1]}, mean={statistics.mean(q):.0f}")
    print(f"[*] Stats: {dict(stats)}")
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Export the v4 training dataset from PostgreSQL")
    parser.add_argument("--output", default="dataset_v4.jsonl")
    parser.add_argument("--eval-output", default="dataset_v4_eval.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    parser.add_argument("--no-rotate", action="store_true", help="Disable rotation augmentation")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()
    asyncio.run(export_dataset_to_jsonl(args.output, args.eval_output, args.eval_fraction, not args.no_rotate, args.max_seq_length))


if __name__ == "__main__":
    main()
