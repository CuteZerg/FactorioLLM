import asyncio
from factoriollm.database import get_connection

async def init_db():
    print("[*] Connecting to database...")
    try:
        conn = await get_connection()
    except Exception as e:
        print(f"[!] Error connecting to database: {e}")
        print("Please ensure your local PostgreSQL server is running and DATABASE_URL in .env is correct.")
        return

    print("[*] Creating tables...")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS blueprints (
            key VARCHAR(255) PRIMARY KEY,
            title TEXT,
            blueprint_string TEXT,
            favorites INTEGER,
            created_at TIMESTAMP,
            raw_data JSONB
        );
        
        CREATE TABLE IF NOT EXISTS processing_tasks (
            key VARCHAR(255) PRIMARY KEY REFERENCES blueprints(key),
            status VARCHAR(50) DEFAULT 'pending',
            error_message TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS dataset_entries (
            id SERIAL PRIMARY KEY,
            blueprint_key VARCHAR(255) REFERENCES blueprints(key),
            user_prompt TEXT,
            refactored_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- v4 pipeline: one canonical (tile-grid, macro-lifted, verified) script per blueprint / book child
        CREATE TABLE IF NOT EXISTS canonical_blueprints (
            id SERIAL PRIMARY KEY,
            blueprint_key VARCHAR(255) NOT NULL REFERENCES blueprints(key),
            book_index TEXT NOT NULL DEFAULT '',
            label TEXT,
            title TEXT,
            canonical_code TEXT,
            code_hash CHAR(64),
            facts JSONB,
            ir JSONB,
            n_tokens INTEGER,
            status VARCHAR(32) NOT NULL,
            error TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (blueprint_key, book_index)
        );
        CREATE INDEX IF NOT EXISTS canonical_blueprints_status_idx ON canonical_blueprints (status);
        CREATE INDEX IF NOT EXISTS canonical_blueprints_hash_idx ON canonical_blueprints (code_hash);

        CREATE TABLE IF NOT EXISTS generated_prompts (
            id SERIAL PRIMARY KEY,
            canonical_id INTEGER NOT NULL REFERENCES canonical_blueprints(id) ON DELETE CASCADE,
            lang VARCHAR(8) NOT NULL,
            prompt TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS generated_prompts_canonical_idx ON generated_prompts (canonical_id);
    """)
    
    print("[*] Database initialized successfully.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())
