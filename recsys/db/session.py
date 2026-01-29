from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy import exc
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def build_sessionmaker(
    database_url: str,
    *,
    engine: Optional[AsyncEngine] = None,
    expire_on_commit: bool = False,
    pool_pre_ping: bool = True,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    if engine is None:
        engine = create_async_engine(database_url, pool_pre_ping=pool_pre_ping)

    session = async_sessionmaker(engine, expire_on_commit=expire_on_commit)
    return engine, session


@asynccontextmanager
async def session_scope(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with sessionmaker() as session:
        try:
            yield session
        except exc.SQLAlchemyError:
            await session.rollback()
            raise
        finally:
            # close() обычно не обязателен (async with закроет), но пусть будет явно
            await session.close()
