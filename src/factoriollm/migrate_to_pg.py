import asyncio
import json
import os
from pathlib import Path
from tqdm import tqdm
from factoriollm.database import get_pool

DATA_DIR = Path("data")
RAW_BLUEPRINTS_FILE = DATA_DIR / "raw" / "blueprints_cache.jsonl"
PROCESSED_KEYS_FILE = DATA_DIR / "processed_keys.txt"
DATASET_FILE = DATA_DIR / "dataset.jsonl"

async def migrate():
    print("[*] Connecting to database...")
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        print("[*] Migrating raw blueprints...")
        if RAW_BLUEPRINTS_FILE.exists():
            with open(RAW_BLUEPRINTS_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in tqdm(lines, desc="Blueprints"):
                if not line.strip():
                    continue
                bp = json.loads(line)
                key = bp.get("_key") or bp.get("id")
                if not key:
                    continue
                title = bp.get("title", "")
                bp_string = bp.get("blueprintString", "")
                favorites = bp.get("favorites", 0)
                if isinstance(favorites, dict):
                    favorites = len(favorites)
                else:
                    try:
                        favorites = int(favorites)
                    except (ValueError, TypeError):
                        favorites = 0
                
                await conn.execute("""
                    INSERT INTO blueprints (key, title, blueprint_string, favorites, raw_data)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (key) DO NOTHING
                """, str(key), str(title), str(bp_string), int(favorites), line)
        else:
            print("  [!] raw_blueprints.json not found.")
            
        print("[*] Migrating processed keys...")
        if PROCESSED_KEYS_FILE.exists():
            with open(PROCESSED_KEYS_FILE, "r", encoding="utf-8") as f:
                keys = [line.strip() for line in f if line.strip()]
            
            for key in tqdm(keys, desc="Processing tasks"):
                await conn.execute("""
                    INSERT INTO processing_tasks (key, status)
                    VALUES ($1, 'success')
                    ON CONFLICT (key) DO NOTHING
                """, str(key))
        else:
            print("  [!] processed_keys.txt not found.")
                
        print("[*] Migrating dataset entries...")
        if DATASET_FILE.exists():
            print("  Clearing existing dataset_entries to prevent duplicates...")
            await conn.execute("TRUNCATE TABLE dataset_entries")
            
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in tqdm(lines, desc="Dataset entries"):
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("messages", [])
                if len(messages) >= 3:
                    prompt = messages[1].get("content", "")
                    code = messages[2].get("content", "")
                    await conn.execute("""
                        INSERT INTO dataset_entries (user_prompt, refactored_code)
                        VALUES ($1, $2)
                    """, str(prompt), str(code))
        else:
            print("  [!] dataset.jsonl not found.")

        print("[*] Migration complete!")

if __name__ == "__main__":
    asyncio.run(migrate())
