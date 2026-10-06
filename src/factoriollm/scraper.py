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
from factoriollm.database import get_pool

load_dotenv(override=True)

FIREBASE_BASE_URL = "https://facorio-blueprints.firebaseio.com/blueprints"
PROXY_URL = os.getenv("PROXY_URL")

# Filtering constants
MIN_FAVORITES = 3
MAX_ENTITIES = 500
MIN_VERSION = 1.0

# Rate limiting
REQUEST_DELAY_SEC = 0.3
MAX_CONCURRENT_REQUESTS = 10

# Data paths


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

    # Filter out Space Age / 2.0 / Expansion content
    title = data.get("title", "").lower()
    space_age_keywords = [
        "space age", "fulgora", "vulcanus", "gleba", "aquilo", 
        "quality", "legendary", "epic", "rare", "uncommon"
    ]
    if any(kw in title for kw in space_age_keywords) or any("space" in t.lower() or "2.0" in t.lower() for t in tags):
        return False, "space age content"

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


async def _load_cached_keys(pool) -> set[str]:
    """Load keys that have already been cached to Postgres."""
    async with pool.acquire() as conn:
        records = await conn.fetch("SELECT key FROM blueprints")
        return {r["key"] for r in records}


async def _save_to_cache(pool, key: str, data: dict[str, Any]) -> None:
    """Insert a single blueprint record to Postgres."""
    title = data.get("title", "")
    bp_string = data.get("blueprintString", "")
    favorites = data.get("favorites", data.get("numberOfFavorites", 0))
    if isinstance(favorites, dict):
        favorites = len(favorites)
    else:
        try:
            favorites = int(favorites)
        except (ValueError, TypeError):
            favorites = 0

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO blueprints (key, title, blueprint_string, favorites, raw_data)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (key) DO UPDATE SET
                title = EXCLUDED.title,
                favorites = EXCLUDED.favorites
        """, str(key), str(title), str(bp_string), favorites, json.dumps(data))


async def scrape_and_filter(
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Main scraping pipeline: fetch keys, download blueprints, filter, and cache.
    """
    pool = await get_pool()
    
    async with _get_http_client() as client:
        # Step 1: Get all keys
        all_keys = await fetch_all_keys(client)

        # Step 2: Determine which keys still need downloading
        cached_keys = await _load_cached_keys(pool)
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
            await _save_to_cache(pool, key, data)

            passed, reason = filter_blueprint(data)
            if passed:
                stats["filtered_in"] += 1
                data["_key"] = key
                results.append(data)
            else:
                stats["filtered_out"] += 1

    # Also include previously cached blueprints that pass filtering
    result_keys = {r.get("_key") for r in results}
    async with pool.acquire() as conn:
        records = await conn.fetch("SELECT key, raw_data FROM blueprints")
        for r in records:
            try:
                record = json.loads(r["raw_data"]) if isinstance(r["raw_data"], str) else r["raw_data"]
                key = r["key"]
                if key in cached_keys:
                    record["_key"] = key
                    passed, _ = filter_blueprint(record)
                    if passed and key not in result_keys:
                        results.append(record)
                        result_keys.add(key)
            except (json.JSONDecodeError, TypeError):
                continue

    print(f"\\n[*] Scraping complete:")
    print(f"    Downloaded:   {stats['downloaded']}")
    print(f"    Passed filter: {stats['filtered_in']}")
    print(f"    Filtered out:  {stats['filtered_out']}")
    print(f"    Errors:        {stats['errors']}")
    print(f"    Total usable:  {len(results)}")

    return results


async def load_cached_blueprints() -> list[dict[str, Any]]:
    """Load and filter all previously cached blueprints from Postgres."""
    pool = await get_pool()
    results: list[dict[str, Any]] = []
    
    async with pool.acquire() as conn:
        records = await conn.fetch("SELECT key, raw_data FROM blueprints")
        for r in records:
            try:
                record = json.loads(r["raw_data"])
                record["_key"] = r["key"]
                passed, _ = filter_blueprint(record)
                if passed:
                    results.append(record)
            except json.JSONDecodeError:
                continue
                
    print(f"[*] Loaded {len(results)} usable blueprints from DB cache")
    return results


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    blueprints = asyncio.run(scrape_and_filter(batch_size=20))
    print(f"\n[*] Sample titles:")
    for bp in blueprints[:5]:
        title = bp.get("title", "Untitled")
        favs = bp.get("numberOfFavorites", 0)
        print(f"  - {title} ({favs} favs)")
