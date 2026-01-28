from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.db.repositories.ratings import RatingsRepository
from recsys.db.repositories.movies import MoviesRepository
from recsys.context import RequestContext, set_request_ctx
from recsys.db.session import get_db_session
from fastapi import Request


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