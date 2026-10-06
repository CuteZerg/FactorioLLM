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
    """)
    
    print("[*] Database initialized successfully.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())
