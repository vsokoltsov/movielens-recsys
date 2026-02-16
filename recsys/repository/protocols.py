import pandas as pd
from typing import Protocol, Optional, Sequence
from recsys.aggregates import Movie


class RatingsRepositoryProtocol(Protocol):
    async def fetch_ratings_df(
        self,
        min_rating: int,
        user_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame: ...

    async def fetch_user_seen_movie_ids(
        self,
        user_id: int,
        min_rating: int,
    ) -> set[int]: ...


class MoviesRepositoryProtocol(Protocol):
    async def fetch_movies_by_ids(self, movie_ids: Sequence[int]) -> list[Movie]: ...
