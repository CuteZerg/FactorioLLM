import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

# URL базы данных по умолчанию
# Для локальной БД без пароля часто используют postgresql://postgres@localhost:5432/factoriollm
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/factoriollm"
)

async def get_pool() -> asyncpg.Pool:
    """Создает и возвращает пул соединений к PostgreSQL."""
    return await asyncpg.create_pool(DATABASE_URL)

async def get_connection() -> asyncpg.Connection:
    """Возвращает одиночное соединение (для простых скриптов)."""
    return await asyncpg.connect(DATABASE_URL)
