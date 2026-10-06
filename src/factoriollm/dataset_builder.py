"""Dataset builder: orchestrates scraper -> decompiler -> ai_refactor -> JSONL output."""

import asyncio
import concurrent.futures
import json
import logging
import warnings
from pathlib import Path
from typing import Any

# Suppress noisy draftsman warnings during batch processing
warnings.filterwarnings("ignore", module="draftsman")
warnings.filterwarnings("ignore", message="Unknown signal")
warnings.filterwarnings("ignore", message="'label' exceeds")

from tqdm import tqdm
from factoriollm.database import get_pool

from factoriollm.ai_refactor import refactor_blueprint_code
from factoriollm.decompiler import decompile_and_normalize
from factoriollm.scraper import scrape_and_filter, load_cached_blueprints

# Paths
DATA_DIR = Path("data")
ERRORS_LOG = DATA_DIR / "errors.log"

# System prompt for the fine-tuning dataset
SYSTEM_PROMPT = (
    "You are a Factorio blueprint generation AI. "
    "Write Python code using the factorio-draftsman library to create "
    "the requested blueprint. The code must end with print(bp.to_string()) "
    "to output the final blueprint string."
)

# Configure error logger
logging.basicConfig(
    filename=str(ERRORS_LOG),
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dataset_builder")


async def _get_processed_keys(pool) -> set[str]:
    """Load the set of already-processed blueprint keys from DB."""
    async with pool.acquire() as conn:
        records = await conn.fetch("SELECT key FROM processing_tasks")
        return {r["key"] for r in records}

async def _mark_processed(pool, key: str, status: str, error_message: str | None = None) -> None:
    """Mark a key as processed in DB."""
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO processing_tasks (key, status, error_message)
            VALUES ($1, $2, $3)
            ON CONFLICT (key) DO UPDATE SET
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                updated_at = CURRENT_TIMESTAMP
        """, str(key), str(status), str(error_message) if error_message else None)

async def _append_to_dataset(pool, key: str, entry: dict[str, Any]) -> None:
    """Append a single ShareGPT-format entry to the database."""
    messages = entry.get("messages", [])
    if len(messages) < 3: return
    user_prompt = messages[1].get("content", "")
    refactored_code = messages[2].get("content", "")
    
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO dataset_entries (blueprint_key, user_prompt, refactored_code)
            VALUES ($1, $2, $3)
        """, str(key), str(user_prompt), str(refactored_code))


def _build_dataset_entry(user_prompt: str, assistant_code: str) -> dict[str, Any]:
    """Build one ShareGPT/OpenAI Chat format entry."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_code},
        ]
    }


async def process_single_blueprint(
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Process a single blueprint through the full pipeline:
    blueprintString -> decompile -> ai_refactor -> dataset entries.

    Returns a list of 0-3 dataset entries (one per generated prompt).
    """
    key = data.get("_key", "unknown")
    title = data.get("title", "Untitled")
    blueprint_string = data.get("blueprintString", "")

    if not blueprint_string:
        logger.warning("Key %s (%s): no blueprintString", key, title)
        return []

    # Step 1: Decompile to flat Python code
    try:
        flat_code = decompile_and_normalize(blueprint_string)
    except Exception as e:
        logger.warning("Key %s (%s): decompiler error: %s", key, title, e)
        return []

    # Skip trivially empty blueprints (no entities to refactor)
    if flat_code.strip().count("\n") < 5:
        logger.warning("Key %s (%s): decompiled code too short, skipping", key, title)
        return []

    # Step 2: AI refactor (generates loops + 3 synthetic prompts)
    try:
        result = await refactor_blueprint_code(flat_code)
    except Exception as e:
        logger.warning("Key %s (%s): ai_refactor error: %s", key, title, e)
        return []

    if result is None:
        logger.warning("Key %s (%s): ai_refactor returned None", key, title)
        return []

    # Validate result structure
    refactored_code = result.get("refactored_code", "")
    user_prompts = result.get("user_prompts", [])

    if not refactored_code or not user_prompts:
        logger.warning("Key %s (%s): invalid ai_refactor result structure", key, title)
        return []

    # Step 3: Build dataset entries (one per prompt)
    entries: list[dict[str, Any]] = []
    for prompt in user_prompts:
        if isinstance(prompt, str) and prompt.strip():
            entry = _build_dataset_entry(prompt.strip(), refactored_code)
            entries.append(entry)

    return entries


async def build_dataset(
    blueprints: list[dict[str, Any]],
    limit: int | None = None,
) -> None:
    """Run the full dataset building pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = await get_pool()
    
    processed_keys = await _get_processed_keys(pool)
    to_process = [bp for bp in blueprints if bp.get("_key") not in processed_keys]

    if limit is not None:
        to_process = to_process[:limit]

    print(f"\\n[*] Dataset builder starting")
    print(f"    Total blueprints available:   {len(blueprints)}")
    print(f"    Already processed:            {len(processed_keys)}")
    print(f"    To process this run:          {len(to_process)}")

    total_entries = 0
    successful = 0
    failed = 0

    max_workers = 50
    semaphore = asyncio.Semaphore(max_workers)

    async def _process_with_semaphore(bp):
        async with semaphore:
            try:
                entries = await process_single_blueprint(bp)
                return bp, entries, None
            except Exception as e:
                return bp, [], e

    tasks = [_process_with_semaphore(bp) for bp in to_process]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Building dataset"):
        bp_data, entries, exc = await coro
        key = bp_data.get("_key", "unknown")
        title = bp_data.get("title", "Untitled")
        safe_title = title.encode("ascii", "ignore").decode("ascii")

        if exc:
            logger.warning("Blueprint %s generated an exception: %s", key, exc)
            await _mark_processed(pool, key, 'failed', str(exc))
            failed += 1
            tqdm.write(f"  [SKIP] [{successful}/{len(to_process)}] \"{safe_title}\" -> error")
            continue

        if entries:
            for entry in entries:
                await _append_to_dataset(pool, key, entry)
            total_entries += len(entries)
            successful += 1
            tqdm.write(
                f"  [OK] [{successful}/{len(to_process)}] "
                f"\"{safe_title}\" -> {len(entries)} entries"
            )
            await _mark_processed(pool, key, 'success')
        else:
            failed += 1
            tqdm.write(f"  [SKIP] [{successful}/{len(to_process)}] \"{safe_title}\" -> skipped")
            await _mark_processed(pool, key, 'failed', 'No valid entries generated')

    print(f"\\n[*] Dataset building complete:")
    print(f"    Blueprints processed:  {successful + failed}")
    print(f"    Successful:            {successful}")
    print(f"    Failed/Skipped:        {failed}")
    print(f"    Total dataset entries:  {total_entries}")


async def run_full_pipeline(
    scrape_batch: int | None = None,
    build_limit: int | None = None,
    use_cache_only: bool = False,
) -> None:
    """
    End-to-end pipeline: scrape -> filter -> decompile -> refactor -> dataset.

    Args:
        scrape_batch: How many new blueprints to download (None = all).
        build_limit: How many blueprints to process through AI refactoring.
        use_cache_only: If True, skip scraping and use only cached data.
    """
    # Step 1: Get blueprints
    if use_cache_only:
        blueprints = await load_cached_blueprints()
    else:
        blueprints = await scrape_and_filter(batch_size=scrape_batch)

    if not blueprints:
        print("[!] No blueprints to process. Exiting.")
        return

    # Step 2: Build the dataset
    await build_dataset(blueprints, limit=build_limit)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FactorioLLM Dataset Builder")
    parser.add_argument(
        "--scrape", type=int, default=None,
        help="Number of blueprints to scrape (default: all)",
    )
    parser.add_argument(
        "--build", type=int, default=None,
        help="Number of blueprints to process through AI refactoring (default: all)",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Skip scraping, use only previously cached blueprints",
    )
    args = parser.parse_args()

    asyncio.run(
        run_full_pipeline(
            scrape_batch=args.scrape,
            build_limit=args.build,
            use_cache_only=args.cache_only,
        )
    )
