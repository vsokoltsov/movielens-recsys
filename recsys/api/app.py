from __future__ import annotations

import os
import pandas as pd
from typing import List, Any, cast, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from recsys.aggregates import ModelType, Movie
from recsys.utils import read_from_csv, read_from_bytes
from recsys.config import MOVIELENS_PATH, MODELS, DATABASE_URL
from recsys.api.config import get_settings, Source
from recsys.recommender import Recommender
from recsys.gcp import GCPStorageClient, GCPModelStorage
from recsys.db.repositories.ratings import RatingsRepository
from recsys.db.repositories.movies import MoviesRepository
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from recsys.api.dependencies import init_request_context
from recsys.context import get_request_ctx



class RecsResponse(BaseModel):
    movies: List[Movie]

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    load_dotenv()

    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    app.state.engine = engine
    app.state.sessionmaker = SessionLocal

    model_path = os.path.join(MODELS, settings.MODEL_NAME)
    gcs_client = GCPStorageClient()
    model_bucket = os.environ.get('MODEL_BUCKET')
    
    storage = None
    movies_df = pd.DataFrame()
    if model_bucket:
        storage = GCPModelStorage(bucket_name=model_bucket)
        app.state.storage = storage

    if settings.SOURCE == Source.CSV:
        raw_bucket = os.environ.get('RAW_BUCKET')
        if not raw_bucket:
            raise ValueError("'RAW_BUCKET' variable is not set")

        users_df = read_from_bytes(
            bts=gcs_client.read_bytes(
                bucket=raw_bucket, 
                obj="ml-1m/users.dat"
            ),
            columns=["user_id", "gender", "age", "occupation", "zip"]
        )
        movies_df = read_from_bytes(
            bts=gcs_client.read_bytes(
                bucket=raw_bucket, 
                obj="ml-1m/movies.dat"
            ),
            columns=["movie_id", "title", "genres"]
        )
        ratings_df = read_from_bytes(
            bts=gcs_client.read_bytes(
                bucket=raw_bucket, 
                obj="ml-1m/ratings.dat"
            ),
            columns=["user_id", "movie_id", "rating", "timestamp"]
        )
        app.state.movies = movies_df
        app.state.ratings = ratings_df
        app.state.users = users_df

    app.state.recommender = Recommender(
        storage=storage,
        model_type=settings.MODEL_TYPE,
        source=settings.SOURCE,
        movies=movies_df,
        rating_threshold=settings.RATING_THRESHOLD,
        model_path=model_path
    )
    async with SessionLocal() as session:
        ratings_repo = RatingsRepository(session=session)
        await app.state.recommender.preload()
    yield
    await engine.dispose()

app = FastAPI(
    title="MovieLens Recommender API",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(init_request_context)]
)

def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return cast(async_sessionmaker[AsyncSession], app.state.sessionmaker)


def get_recommender() -> Recommender:
    return cast(Recommender, app.state.recommender)

@app.get(
    "/users/{id}/recommendations",
    response_model=RecsResponse,
)
async def get_recommendations(
    id: int = Path(..., ge=1),
    k: int = Query(10, ge=1, le=100),
    recommender: Recommender = Depends(get_recommender),
    ctx = Depends(get_request_ctx),
):
    try:
        movies = await recommender.recommend(
            ctx=ctx,
            user_id=id,
            n_items=k,
        )
        return RecsResponse(movies=movies)
    except Exception as e:
        raise HTTPException(status_code=501, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recsys.api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
