from __future__ import annotations

import os
from typing import List, Any, cast, AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from recsys.aggregates import ModelType, Movie
from recsys.utils import read_from_csv
from recsys.config import MOVIELENS_PATH, MODELS
from recsys.api.config import get_settings
from recsys.recommender import Recommender


class RecsResponse(BaseModel):
    movies: List[Movie]

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    settings = get_settings()
    app.state.movies = read_from_csv(
        path=os.path.join(MOVIELENS_PATH, "movies.dat"),
        columns=["movie_id", "title", "genres"],
    )
    app.state.ratings = read_from_csv(
        path=os.path.join(MOVIELENS_PATH, "ratings.dat"),
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    app.state.users = read_from_csv(
        path=os.path.join(MOVIELENS_PATH, "users.dat"),
        columns=["user_id", "gender", "age", "occupation", "zip"],
    )
    app.state.recommender = Recommender(
        model_type=settings.MODEL_TYPE,
        source=settings.SOURCE,
        movies=app.state.movies,
        rating_threshold=settings.RATING_THRESHOLD,
        model_path=os.path.join(MODELS, settings.MODEL_NAME)
    )
    yield

app = FastAPI(
    title="MovieLens Recommender API",
    version="1.0.0",
    lifespan=lifespan
)

def get_recommender() -> Recommender:
    return cast(Recommender, app.state.recommender)

@app.get(
    "/users/{id}/recommendations",
    response_model=RecsResponse,
)
def get_recommendations(
    id: int = Path(..., ge=1),
    k: int = Query(10, ge=1, le=100),
    recommender: Recommender = Depends(get_recommender)
):
    try:
        movies: List[Movie] = recommender.recommend(user_id=id, n_items=k)
        return RecsResponse(movies=movies)
    except Exception as e:
        # e.g., neural_mf not implemented
        raise HTTPException(status_code=501, detail=e)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recsys.api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
