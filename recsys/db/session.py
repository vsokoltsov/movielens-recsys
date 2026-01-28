from fastapi import Request
from typing import cast
from collections.abc import AsyncGenerator
from sqlalchemy import exc
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    sessionmaker = cast(async_sessionmaker[AsyncSession], request.app.state.sessionmaker)
    async with sessionmaker() as session:
        try:
            yield session
        except exc.SQLAlchemyError:
            await session.rollback()
            raise
        finally:
            await session.close()

