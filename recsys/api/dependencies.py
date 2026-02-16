from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.repository.protocols import RatingsRepositoryProtocol
from recsys.repository.protocols import MoviesRepositoryProtocol
from recsys.repository.db.ratings import RatingsDBRepository
from recsys.repository.db.movies import MoviesDBRepository
from recsys.repository.csv.ratings import RatingsCSVRepository
from recsys.repository.csv.movies import MoviesCSVRepository
from recsys.modeling.dataset import Dataset
from recsys.context import RequestContext, set_request_ctx
from recsys.aggregates import Source
from fastapi import Request
from typing import cast, Optional
from collections.abc import AsyncGenerator
from sqlalchemy import exc
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
)
from recsys.config import get_settings


async def get_db_session(
    request: Request,
) -> AsyncGenerator[Optional[AsyncSession], None]:
    if getattr(request.app.state, "sessionmaker", None) is None:
        yield None
        return
    sessionmaker = cast(
        async_sessionmaker[AsyncSession], request.app.state.sessionmaker
    )
    async with sessionmaker() as session:
        try:
            yield session
        except exc.SQLAlchemyError:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_request_context(
    session: Optional[AsyncSession] = Depends(get_db_session),
) -> None:
    settings = get_settings()
    ratings_repo: RatingsRepositoryProtocol
    movies_repo: MoviesRepositoryProtocol

    if settings.SOURCE == Source.CSV:
        dataset = Dataset(dataset_path=settings.MOVIELENS_PATH)
        ratings_repo = RatingsCSVRepository(df=dataset.ratings)
        movies_repo = MoviesCSVRepository(df=dataset.movies)
    elif settings.SOURCE == Source.DB:
        if session is None:
            raise ValueError("database session was not initialized")
        ratings_repo = RatingsDBRepository(session=session)
        movies_repo = MoviesDBRepository(session=session)

    ctx = RequestContext(
        ratings=ratings_repo,
        movies=movies_repo,
    )
    set_request_ctx(ctx)


def get_recommender(request: Request):
    return request.app.state.recommender


def get_sessionmaker(request: Request):
    return request.app.state.sessionmaker
