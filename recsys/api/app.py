from __future__ import annotations

import os
import pandas as pd
from typing import List, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Path, Query
from pydantic import BaseModel

from recsys.aggregates import Movie
from recsys.utils import read_from_bytes, read_from_csv
from recsys.config import get_settings
from recsys.aggregates import Source, Setup
from recsys.recommender import Recommender
from recsys.storage import (
    GCPStorageClient,
    GCPModelStorage,
    LocalStorage,
    StorageProtocol,
)
from recsys.api.dependencies import get_recommender
from recsys.api.dependencies import init_request_context
from recsys.context import get_request_ctx
from recsys.db.session import build_sessionmaker


class RecsResponse(BaseModel):
    movies: List[Movie]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    settings = get_settings()
    storage: StorageProtocol
    if settings.SOURCE == Source.DB:
        engine, session = build_sessionmaker(
            database_url=settings.DATABASE_URL,
            expire_on_commit=False,
            pool_pre_ping=True,
        )
        app.state.engine = engine
        app.state.sessionmaker = session

    if settings.SETUP == Setup.LOCAL:
        storage = LocalStorage(model_path=settings.MODELS_PATH)

        if settings.SOURCE == Source.CSV:
            users_df = read_from_csv(
                path=os.path.join(settings.MOVIELENS_PATH, "users.dat"),
                columns=["user_id", "gender", "age", "occupation", "zip"],
            )
            movies_df = read_from_csv(
                path=os.path.join(settings.MOVIELENS_PATH, "movies.dat"),
                columns=["movie_id", "title", "genres"],
            )
            ratings_df = read_from_csv(
                path=os.path.join(settings.MOVIELENS_PATH, "ratings.dat"),
                columns=["user_id", "movie_id", "rating", "timestamp"],
            )
            app.state.movies = movies_df
            app.state.ratings = ratings_df
            app.state.users = users_df

    if settings.SETUP == Setup.CLOUD:
        gcs_client = GCPStorageClient()
        model_bucket = settings.MODEL_BUCKET

        movies_df = pd.DataFrame()
        if model_bucket:
            storage = GCPModelStorage(bucket_name=model_bucket)
            app.state.storage = storage

        if settings.SOURCE == Source.CSV:
            raw_bucket = settings.RAW_BUCKET
            if not raw_bucket:
                raise ValueError("'RAW_BUCKET' variable is not set")

            users_df = read_from_bytes(
                bts=gcs_client.read_bytes(bucket=raw_bucket, obj="ml-1m/users.dat"),
                columns=["user_id", "gender", "age", "occupation", "zip"],
            )
            movies_df = read_from_bytes(
                bts=gcs_client.read_bytes(bucket=raw_bucket, obj="ml-1m/movies.dat"),
                columns=["movie_id", "title", "genres"],
            )
            ratings_df = read_from_bytes(
                bts=gcs_client.read_bytes(bucket=raw_bucket, obj="ml-1m/ratings.dat"),
                columns=["user_id", "movie_id", "rating", "timestamp"],
            )
            app.state.movies = movies_df
            app.state.ratings = ratings_df
            app.state.users = users_df

    app.state.recommender = Recommender(
        model_type=settings.MODEL_TYPE,
        source=settings.SOURCE,
        model_path=settings.MODELS_PATH,
        rating_threshold=settings.RATING_THRESHOLD,
    )
    await app.state.recommender.preload(storage=storage)
    yield
    await engine.dispose()


app = FastAPI(
    title="MovieLens Recommender API",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(init_request_context)],
)


@app.get(
    "/users/{id}/recommendations",
    response_model=RecsResponse,
)
async def get_recommendations(
    id: int = Path(..., ge=1),
    k: int = Query(10, ge=1, le=100),
    recommender: Recommender = Depends(get_recommender),
    ctx=Depends(get_request_ctx),
):
    movies = await recommender.recommend(
        ctx=ctx,
        user_id=id,
        n_items=k,
    )
    return RecsResponse(movies=movies)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "recsys.api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
