from typing import Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from recsys.db.models import Rating
from sqlalchemy import select


class RatingsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def fetch_ratings_df(
        self,
        min_rating: int,
        user_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        stmt = select(
            Rating.user_id,
            Rating.movie_id,
            Rating.rating,
            Rating.timestamp,
        ).where(Rating.rating >= min_rating)

        if user_id is not None:
            stmt = stmt.where(Rating.user_id == user_id)

        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        rows = result.all()

        return pd.DataFrame(
            rows,
            columns=["user_id", "movie_id", "rating", "timestamp"],
        )

    async def fetch_user_seen_movie_ids(
        self,
        user_id: int,
        min_rating: int,
    ) -> set[int]:
        query = text("""
            SELECT movie_id
            FROM ratings
            WHERE user_id = :user_id
              AND rating >= :min_rating
        """)

        result = await self.session.execute(
            query,
            {
                "user_id": user_id,
                "min_rating": min_rating,
            },
        )

        return {row[0] for row in result.fetchall()}