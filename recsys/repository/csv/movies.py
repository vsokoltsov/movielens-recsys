from __future__ import annotations

from typing import Sequence
import pandas as pd

from recsys.aggregates import Movie
from recsys.repository.protocols import MoviesRepositoryProtocol


class MoviesCSVRepository(MoviesRepositoryProtocol):
    """Movies from a pandas DataFrame. Returns aggregate Movie (protocol-compliant)."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with columns: movie_id, title, genres
        """
        self._df = df.copy()

    async def fetch_movies_by_ids(self, movie_ids: Sequence[int]) -> list[Movie]:
        if not movie_ids:
            return []
        ids = [int(x) for x in movie_ids]
        out = self._df[self._df["movie_id"].isin(ids)]
        order = {mid: i for i, mid in enumerate(ids)}
        out = out.sort_values("movie_id", key=lambda s: s.map(order))
        out = out.drop_duplicates(subset=["movie_id"], keep="first")
        return [
            Movie(id=int(r.movie_id), title=str(r.title), genre=str(r.genres))
            for r in out.itertuples(index=False)
        ]
