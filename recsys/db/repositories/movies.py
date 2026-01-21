from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recsys.db.models import Movie

@dataclass
class MoviesRepository:
    session: AsyncSession

    async def fetch_movies_by_ids(self, movie_ids: Sequence[int]) -> list[Movie]:
        if not movie_ids:
            return []
        stmt = select(Movie).where(Movie.movie_id.in_([int(x) for x in movie_ids]))
        res = await self.session.execute(stmt)
        return list(res.scalars().all())