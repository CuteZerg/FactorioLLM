import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()


def _database_url() -> str:
    """Возвращает DATABASE_URL из окружения/.env (без зашитых дефолтных паролей)."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set: add it to .env (e.g. postgresql://user:password@localhost:5432/factoriollm)")
    return url


async def get_pool() -> asyncpg.Pool:
    """Создает и возвращает пул соединений к PostgreSQL."""
    return await asyncpg.create_pool(_database_url())

async def get_connection() -> asyncpg.Connection:
    """Возвращает одиночное соединение (для простых скриптов)."""
    return await asyncpg.connect(_database_url())
