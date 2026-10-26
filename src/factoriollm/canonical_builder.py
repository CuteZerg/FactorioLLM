"""
Canonical dataset builder (v4 pipeline, stage 1 — local, no LLM calls).

For every cached blueprint (and every child of a blueprint book) in the `blueprints` table:
    raw JSON -> tile-grid IR -> policy filters -> macro-lifted code -> token budget -> dedupe
    -> execution in the Docker sandbox (batch mode) -> exact IR equivalence check -> facts
and stores one canonical script per blueprint in `canonical_blueprints`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from factoriollm.blueprint_ir import (
    BlueprintIRError,
    IRBlueprint,
    decode_blueprint_string,
    extract_facts,
    iter_blueprints,
    parse_blueprint_dict,
)
from factoriollm.database import get_pool
from factoriollm.dataset_lifter import ir_equivalent, lift_ir_to_code, run_trusted_script
from factoriollm.sandbox import DockerSandbox

TOKENIZER_PATH = Path("factorio_lora_model/tokenizer.json")

# Children taken from one blueprint book, by the book's favorites: popular books are where players
# keep their canonical designs, so they are not truncated.
BOOK_CHILD_LIMITS = ((100, None), (10, 50), (0, 25))
MIN_VERSION = (0, 15)  # older strings are converted by the IR (renames, 8 -> 16 directions)


def book_child_limit(favorites: int) -> Optional[int]:
    return next(limit for threshold, limit in BOOK_CHILD_LIMITS if (favorites or 0) >= threshold)


@dataclass
class Candidate:
    blueprint_key: str
    book_index: str
    label: str
    title: str
    ir: IRBlueprint
    code: str
    code_hash: str
    n_tokens: int
    status: str = "pending"
    error: Optional[str] = None


def make_token_counter() -> Callable[[str], int]:
    """Token counter matching the current model's tokenizer; falls back to a character heuristic."""
    if TOKENIZER_PATH.exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(TOKENIZER_PATH))
        tok.no_truncation()
        return lambda text: len(tok.encode(text).ids)
    return lambda text: len(text) // 3


def ir_to_json(ir: IRBlueprint) -> List[list]:
    return [[e.name, e.x, e.y, e.direction, dict(e.extras)] for e in ir.entities]


def collect_candidates(
    rows: List[Any],
    count_tokens: Callable[[str], int],
    min_entities: int,
    max_entities: int,
    max_code_tokens: int,
    stats: Counter,
) -> List[Candidate]:
    """Parses, filters, lifts and dedupes all blueprints of the given DB rows."""
    candidates: List[Candidate] = []
    seen_hashes: Dict[str, Candidate] = {}

    for row in rows:
        try:
            data = decode_blueprint_string(row["blueprint_string"])
        except BlueprintIRError:
            stats["decode_error"] += 1
            continue

        limit = book_child_limit(row["favorites"])
        for n, (book_index, bp, version) in enumerate(iter_blueprints(data)):
            if limit is not None and n >= limit:
                stats["book_cap"] += 1
                break
            if version != (0, 0) and version < MIN_VERSION:  # (0, 0): string without a version field
                stats["old_version"] += 1
                continue
            n_entities = len(bp.get("entities") or [])
            if n_entities < min_entities or n_entities > max_entities:
                stats["entity_count"] += 1
                continue
            try:
                ir = parse_blueprint_dict(bp, version=version, book_path=book_index)
            except BlueprintIRError as e:
                msg = str(e)
                stats["not_allowed" if "not allowed" in msg else "unknown_entity" if "unknown entity" in msg
                      else "legacy_recipe" if "legacy recipe" in msg else "quality" if "quality" in msg
                      else "unknown_recipe" if "unknown recipe" in msg else "not_vanilla_recipe" if "base game" in msg
                      else "circuit_logic" if "circuit-wired" in msg
                      else "label_signal" if "label signal" in msg
                      else "modules_or_requests" if "module" in msg or "request" in msg else "invalid_geometry"] += 1
                continue

            code = lift_ir_to_code(ir)
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            cand = Candidate(
                blueprint_key=row["key"],
                book_index=book_index,
                label=ir.label,
                title=" ".join(str(row["title"] or "").split())[:200],
                ir=ir,
                code=code,
                code_hash=code_hash,
                n_tokens=count_tokens(code),
            )
            if code_hash in seen_hashes:
                cand.status, cand.error = "duplicate", f"same code as {seen_hashes[code_hash].blueprint_key}/{seen_hashes[code_hash].book_index}"
            elif cand.n_tokens > max_code_tokens:
                cand.status, cand.error = "too_long", f"{cand.n_tokens} tokens > {max_code_tokens}"
            else:
                seen_hashes[code_hash] = cand
            stats[cand.status] += 1
            candidates.append(cand)
    return candidates


def verify_candidates(candidates: List[Candidate], mode: str, batch_size: int, workers: int) -> None:
    """Executes pending candidates and marks them ok / mismatch / exec_error (in place)."""
    pending = [c for c in candidates if c.status == "pending"]
    if mode == "local":
        for c in pending:
            try:
                ok = ir_equivalent(c.ir, run_trusted_script(c.code))
                c.status, c.error = ("ok", None) if ok else ("mismatch", "rebuilt blueprint differs")
            except Exception as e:
                c.status, c.error = "exec_error", f"{type(e).__name__}: {e}"[:500]
        return

    sandbox = DockerSandbox(memory_limit="1g")
    sandbox.ensure_image()
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    def _run(batch: List[Candidate]) -> None:
        results = sandbox.execute_batch([c.code for c in batch], per_script_timeout=20)
        for c, r in zip(batch, results):
            if not r.success:
                c.status, c.error = "exec_error", (r.stderr or r.error_message or "")[-500:]
            elif ir_equivalent(c.ir, r.blueprint_string):
                c.status, c.error = "ok", None
            else:
                c.status, c.error = "mismatch", "rebuilt blueprint differs"

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_run, batches):
            done += 1
            print(f"    verified batch {done}/{len(batches)}")


async def _store(pool, candidates: List[Candidate]) -> None:
    records = [
        (
            c.blueprint_key, c.book_index, c.label, c.title, c.code, c.code_hash,
            json.dumps(extract_facts(c.ir)), json.dumps(ir_to_json(c.ir)), c.n_tokens, c.status, c.error,
        )
        for c in candidates
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO canonical_blueprints
                (blueprint_key, book_index, label, title, canonical_code, code_hash, facts, ir, n_tokens, status, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10, $11)
            ON CONFLICT (blueprint_key, book_index) DO UPDATE SET
                label = EXCLUDED.label, title = EXCLUDED.title, canonical_code = EXCLUDED.canonical_code,
                code_hash = EXCLUDED.code_hash, facts = EXCLUDED.facts, ir = EXCLUDED.ir,
                n_tokens = EXCLUDED.n_tokens, status = EXCLUDED.status, error = EXCLUDED.error,
                updated_at = CURRENT_TIMESTAMP
            """,
            records,
        )


async def build_canonical(
    min_favorites: int = 2,
    min_entities: int = 4,
    max_entities: int = 500,
    max_code_tokens: int = 3600,
    limit: Optional[int] = None,
    verify: str = "sandbox",
    batch_size: int = 150,
    workers: int = 4,
    rebuild: bool = False,
    chunk_rows: int = 500,
    dry_run: bool = False,
) -> Counter:
    pool = await get_pool()
    async with pool.acquire() as conn:
        done_keys = set() if rebuild else {r["blueprint_key"] for r in await conn.fetch("SELECT DISTINCT blueprint_key FROM canonical_blueprints")}
        rows = await conn.fetch(
            "SELECT key, title, favorites, blueprint_string FROM blueprints WHERE favorites >= $1 ORDER BY favorites DESC, key",
            min_favorites,
        )
    rows = [r for r in rows if r["key"] not in done_keys]
    if limit is not None:
        rows = rows[:limit]
    print(f"[*] {len(rows)} blueprint rows to process (favorites >= {min_favorites}, {len(done_keys)} keys already done)")

    stats: Counter = Counter()
    final: Counter = Counter()
    count_tokens = make_token_counter()
    for chunk_start in range(0, len(rows), chunk_rows):
        chunk = rows[chunk_start:chunk_start + chunk_rows]
        print(f"[*] Rows {chunk_start + 1}-{chunk_start + len(chunk)} of {len(rows)}")
        candidates = await asyncio.to_thread(
            collect_candidates, chunk, count_tokens, min_entities, max_entities, max_code_tokens, stats
        )
        if dry_run:  # statistics only: no sandbox, nothing stored
            final.update(c.status for c in candidates)
            continue
        # Codes stored by earlier chunks / runs count as duplicates too.
        # A rebuilt row matching its own earlier version is not a duplicate.
        async with pool.acquire() as conn:
            stored: Dict[str, set] = {}
            for r in await conn.fetch("SELECT code_hash, blueprint_key, book_index FROM canonical_blueprints WHERE status = 'ok'"):
                stored.setdefault(r["code_hash"], set()).add((r["blueprint_key"], r["book_index"]))
        for c in candidates:
            if c.status == "pending" and stored.get(c.code_hash, set()) - {(c.blueprint_key, c.book_index)}:
                c.status, c.error = "duplicate", "same code stored earlier"

        print(f"    verifying {sum(c.status == 'pending' for c in candidates)} candidates ({verify})...")
        await asyncio.to_thread(verify_candidates, candidates, verify, batch_size, workers)
        await _store(pool, candidates)
        final.update(c.status for c in candidates)
        print(f"    stored: {dict(Counter(c.status for c in candidates))}")
    await pool.close()

    print("\n[*] Canonical build summary")
    print(f"    Skipped before lifting: {dict((k, v) for k, v in stats.items() if k not in ('pending', 'duplicate', 'too_long'))}")
    print(f"    {'Would store' if dry_run else 'Stored'} by status: {dict(final)}")
    return final


def main():
    parser = argparse.ArgumentParser(description="Build canonical tile-grid scripts from cached blueprints")
    parser.add_argument("--min-favorites", type=int, default=2)
    parser.add_argument("--max-entities", type=int, default=500)
    parser.add_argument("--max-code-tokens", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true", help="Only parse, lift and count; no sandbox, no DB writes")
    parser.add_argument("--limit", type=int, default=None, help="Max blueprint rows (books count as one row)")
    parser.add_argument("--verify", choices=["sandbox", "local"], default="sandbox")
    parser.add_argument("--workers", type=int, default=4, help="Parallel sandbox containers")
    parser.add_argument("--rebuild", action="store_true", help="Reprocess keys that are already stored")
    args = parser.parse_args()

    asyncio.run(build_canonical(
        min_favorites=args.min_favorites,
        max_entities=args.max_entities,
        max_code_tokens=args.max_code_tokens,
        dry_run=args.dry_run,
        limit=args.limit,
        verify=args.verify,
        workers=args.workers,
        rebuild=args.rebuild,
    ))


if __name__ == "__main__":
    main()
