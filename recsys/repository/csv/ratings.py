from typing import Optional
import pandas as pd
from recsys.repository.protocols import RatingsRepositoryProtocol


class RatingsCSVRepository(RatingsRepositoryProtocol):
    """Ratings from a pandas DataFrame (e.g. loaded from CSV). Same interface as DB version."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        self._df = df.copy()

    async def fetch_ratings_df(
        self,
        min_rating: int,
        user_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        out = self._df[self._df["rating"] >= min_rating]
        if user_id is not None:
            out = out[out["user_id"] == user_id]
        if limit is not None:
            out = out.head(limit)
        return out[["user_id", "movie_id", "rating", "timestamp"]].reset_index(
            drop=True
        )

    async def fetch_user_seen_movie_ids(
        self,
        user_id: int,
        min_rating: int,
    ) -> set[int]:
        mask = (self._df["user_id"] == user_id) & (self._df["rating"] >= min_rating)
        return set(self._df.loc[mask, "movie_id"].unique())
