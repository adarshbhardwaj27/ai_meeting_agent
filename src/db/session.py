"""
Async SQLite engine and session factory.
"""

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base

# Default path: data/meetmind.db (relative to CWD — usually repo root)
_default_url = "sqlite+aiosqlite:///./data/meetmind.db"
DATABASE_URL = os.getenv("DATABASE_URL", _default_url)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables and the data/ directory if needed."""
    # For SQLite, try to create the parent directory (works for relative paths
    # and paths the app user owns). In Docker the directory is pre-created via
    # the Dockerfile, so a PermissionError here just means it already exists
    # via a volume mount — safe to ignore.
    if DATABASE_URL.startswith("sqlite"):
        db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass  # Directory provided by volume mount — no action needed

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an AsyncSession per request."""
    async with AsyncSessionLocal() as session:
        yield session
