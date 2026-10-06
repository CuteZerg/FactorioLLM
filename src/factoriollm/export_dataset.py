import asyncio
import json
from pathlib import Path
from factoriollm.database import get_pool

DATA_DIR = Path("data")

async def export_dataset_to_jsonl(output_file="dataset.jsonl", batch_size=1000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / output_file
    
    pool = await get_pool()
    total_exported = 0
    
    print(f"[*] Exporting dataset from PostgreSQL to {out_path}...")
    
    async with pool.acquire() as conn:
        # Use a server-side cursor to handle millions of rows without RAM issues
        async with conn.transaction():
            cursor = await conn.cursor("SELECT user_prompt, refactored_code FROM dataset_entries")
            
            with open(out_path, "w", encoding="utf-8") as f:
                while True:
                    records = await cursor.fetch(batch_size)
                    if not records:
                        break
                        
                    for r in records:
                        # Convert to ShareGPT / OpenAI chat format
                        entry = {
                            "messages": [
                                {"role": "system", "content": "You are a Factorio blueprint generation AI. Write Python code using the factorio-draftsman library to create the requested blueprint. The code must end with print(bp.to_string()) to output the final blueprint string."},
                                {"role": "user", "content": r["user_prompt"]},
                                {"role": "assistant", "content": r["refactored_code"]}
                            ]
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        total_exported += 1
                        
                    print(f"    Exported {total_exported} records...")

    print(f"[*] Done! Exported {total_exported} total records to {out_path}")

if __name__ == "__main__":
    asyncio.run(export_dataset_to_jsonl())
