from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.db.repositories.ratings import RatingsRepository
from recsys.db.repositories.movies import MoviesRepository
from recsys.context import RequestContext, set_request_ctx
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

async def init_request_context(
    session: AsyncSession = Depends(get_db_session),
) -> None:
    ctx = RequestContext(
        ratings=RatingsRepository(session=session),
        movies=MoviesRepository(session=session),
    )
    set_request_ctx(ctx)


def get_recommender(request: Request):
    return request.app.state.recommender

def get_sessionmaker(request: Request):
    return request.app.state.sessionmaker
