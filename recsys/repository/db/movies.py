from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.db.models import Movie
from recsys.aggregates import Movie as MovieAggregate
from recsys.repository.protocols import MoviesRepositoryProtocol


@dataclass
class MoviesDBRepository(MoviesRepositoryProtocol):
    session: AsyncSession

    async def fetch_movies_by_ids(
        self, movie_ids: Sequence[int]
    ) -> list[MovieAggregate]:
        if not movie_ids:
            return []
        stmt = select(Movie).where(Movie.movie_id.in_([int(x) for x in movie_ids]))
        res = await self.session.execute(stmt)
        orm_movies = list(res.scalars().all())
        return [
            MovieAggregate(id=int(m.movie_id), title=str(m.title), genre=str(m.genres) or "")
            for m in orm_movies
        ]
