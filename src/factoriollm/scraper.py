"""Async scraper for downloading Factorio blueprints from factorioprints.com (Firebase REST API)."""

import asyncio
import base64
import json
import os
import re
import zlib
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

FIREBASE_BASE_URL = "https://facorio-blueprints.firebaseio.com/blueprints"
PROXY_URL = os.getenv("PROXY_URL")

# Filtering constants
MIN_FAVORITES = 5
MAX_ENTITIES = 300
MIN_VERSION = 1.0

# Rate limiting
REQUEST_DELAY_SEC = 0.3
MAX_CONCURRENT_REQUESTS = 10

# Data paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CACHE_FILE = RAW_DIR / "blueprints_cache.jsonl"
KEYS_FILE = RAW_DIR / "all_keys.json"


def _get_http_client() -> httpx.AsyncClient:
    """Create an async HTTP client with optional proxy support."""
    transport_kwargs: dict[str, Any] = {}
    if PROXY_URL:
        transport_kwargs["proxy"] = PROXY_URL
    return httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS),
        **transport_kwargs,
    )


def _parse_version_from_tags(tags: list[str]) -> float | None:
    """Extract the numeric version from tags like '/version/0,14/' or '/version/1,1/'."""
    for tag in tags:
        match = re.search(r"/version/(\d+),(\d+)/", tag)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major + minor / 100.0
    return None


def _count_entities(blueprint_string: str) -> int | None:
    """Decode a blueprint string and count entities without full draftsman parsing."""
    try:
        # Strip the version byte prefix (e.g. '0')
        raw = base64.b64decode(blueprint_string[1:])
        data = json.loads(zlib.decompress(raw))
        # Handle both single blueprints and blueprint books
        if "blueprint" in data:
            entities = data["blueprint"].get("entities", [])
            return len(entities)
        elif "blueprint_book" in data:
            # Skip blueprint books entirely — too complex for single-blueprint dataset
            return None
    except Exception:
        return None


def filter_blueprint(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a blueprint passes all quality filters.

    Returns (passed: bool, reason: str).
    """
    # Must have a blueprint string
    blueprint_string = data.get("blueprintString")
    if not blueprint_string:
        return False, "no blueprintString"

    # Minimum favorites threshold
    favorites = data.get("numberOfFavorites", 0)
    if favorites < MIN_FAVORITES:
        return False, f"low favorites ({favorites} < {MIN_FAVORITES})"

    # Tags-based filtering
    tags = data.get("tags", [])

    # Only vanilla blueprints
    if any("/mods/modded/" in tag for tag in tags):
        return False, "modded blueprint"

    # Version filter: >= 1.0
    version = _parse_version_from_tags(tags)
    if version is not None and version < MIN_VERSION:
        return False, f"old version ({version})"

    # Entity count filter
    entity_count = _count_entities(blueprint_string)
    if entity_count is None:
        return False, "cannot decode / blueprint book"
    if entity_count > MAX_ENTITIES:
        return False, f"too many entities ({entity_count} > {MAX_ENTITIES})"
    if entity_count == 0:
        return False, "empty blueprint (0 entities)"

    return True, "ok"


async def fetch_all_keys(client: httpx.AsyncClient) -> list[str]:
    """Fetch all blueprint keys from Firebase (shallow query)."""
    url = f"{FIREBASE_BASE_URL}.json?shallow=true"
    print("[*] Fetching all blueprint keys (shallow)...")
    resp = await client.get(url)
    resp.raise_for_status()
    keys_dict: dict[str, bool] = resp.json()
    keys = list(keys_dict.keys())
    print(f"[*] Found {len(keys)} total blueprints on factorioprints.com")
    return keys


async def fetch_blueprint(
    client: httpx.AsyncClient,
    key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None]:
    """Fetch a single blueprint by key with rate limiting."""
    async with semaphore:
        url = f"{FIREBASE_BASE_URL}/{key}.json"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            await asyncio.sleep(REQUEST_DELAY_SEC)
            return key, resp.json()
        except Exception as e:
            print(f"[!] Error fetching {key}: {e}")
            return key, None


def _load_cached_keys() -> set[str]:
    """Load keys that have already been cached to disk."""
    cached: set[str] = set()
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    cached.add(record.get("_key", ""))
                except json.JSONDecodeError:
                    continue
    return cached


def _save_to_cache(key: str, data: dict[str, Any]) -> None:
    """Append a single blueprint record to the cache file."""
    record = {"_key": key, **data}
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def scrape_and_filter(
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Main scraping pipeline: fetch keys, download blueprints, filter, and cache.

    Args:
        batch_size: If set, limit the number of blueprints to process.

    Returns:
        List of filtered blueprint data dicts.
    """
    # Ensure data directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with _get_http_client() as client:
        # Step 1: Get all keys (or load from cache)
        if KEYS_FILE.exists():
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                all_keys = json.load(f)
            print(f"[*] Loaded {len(all_keys)} keys from cache")
        else:
            all_keys = await fetch_all_keys(client)
            with open(KEYS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_keys, f)

        # Step 2: Determine which keys still need downloading
        cached_keys = _load_cached_keys()
        keys_to_fetch = [k for k in all_keys if k not in cached_keys]
        print(f"[*] Already cached: {len(cached_keys)}, remaining: {len(keys_to_fetch)}")

        if batch_size is not None:
            keys_to_fetch = keys_to_fetch[:batch_size]
            print(f"[*] Batch mode: processing {batch_size} blueprints")

        # Step 3: Download with concurrency control
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [fetch_blueprint(client, k, semaphore) for k in keys_to_fetch]

        stats = {"downloaded": 0, "filtered_in": 0, "filtered_out": 0, "errors": 0}
        results: list[dict[str, Any]] = []

        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Downloading blueprints",
        ):
            key, data = await coro
            if data is None:
                stats["errors"] += 1
                continue

            stats["downloaded"] += 1
            _save_to_cache(key, data)

            passed, reason = filter_blueprint(data)
            if passed:
                stats["filtered_in"] += 1
                data["_key"] = key
                results.append(data)
            else:
                stats["filtered_out"] += 1

    # Also include previously cached blueprints that pass filtering
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    key = record.get("_key", "")
                    if key in cached_keys:
                        passed, _ = filter_blueprint(record)
                        if passed and not any(r.get("_key") == key for r in results):
                            results.append(record)
                except json.JSONDecodeError:
                    continue

    print(f"\n[*] Scraping complete:")
    print(f"    Downloaded:   {stats['downloaded']}")
    print(f"    Passed filter: {stats['filtered_in']}")
    print(f"    Filtered out:  {stats['filtered_out']}")
    print(f"    Errors:        {stats['errors']}")
    print(f"    Total usable:  {len(results)}")

    return results


def load_cached_blueprints() -> list[dict[str, Any]]:
    """Load and filter all previously cached blueprints without network access."""
    results: list[dict[str, Any]] = []
    if not CACHE_FILE.exists():
        return results

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                passed, _ = filter_blueprint(record)
                if passed:
                    results.append(record)
            except json.JSONDecodeError:
                continue

    print(f"[*] Loaded {len(results)} usable blueprints from cache")
    return results


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    blueprints = asyncio.run(scrape_and_filter(batch_size=20))
    print(f"\n[*] Sample titles:")
    for bp in blueprints[:5]:
        title = bp.get("title", "Untitled")
        favs = bp.get("numberOfFavorites", 0)
        print(f"  - {title} ({favs} favs)")
