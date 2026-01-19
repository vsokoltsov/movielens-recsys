from __future__ import annotations

import os
from typing import List, Any, cast, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from recsys.aggregates import ModelType, Movie
from recsys.utils import read_from_csv, read_from_bytes
from recsys.config import MOVIELENS_PATH, MODELS
from recsys.api.config import get_settings
from recsys.recommender import Recommender
from recsys.gcp import GCPStorageClient
from dotenv import load_dotenv


class RecsResponse(BaseModel):
    movies: List[Movie]

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    load_dotenv()

    settings = get_settings()
    model_path = os.path.join(MODELS, settings.MODEL_NAME)
    gcs_client = GCPStorageClient()
    model_bucket = os.environ.get('MODEL_BUCKET')
    raw_bucket = os.environ.get('RAW_BUCKET')
    movies_path = os.path.join(MOVIELENS_PATH, "movies.dat")
    
    if model_bucket:
        model_dest_path = os.path.join("/tmp", settings.MODEL_NAME)
        gcs_client.download(
            bucket_name=str(model_bucket),
            object_name=settings.MODEL_NAME,
            dst_path=model_dest_path
        )
        model_path = model_dest_path

    if raw_bucket:
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
    else:
        users_df = read_from_csv(
            path=os.path.join(MOVIELENS_PATH, "users.dat"),
            columns=["user_id", "gender", "age", "occupation", "zip"],
        )
        movies_df = read_from_csv(
            path=movies_path,
            columns=["movie_id", "title", "genres"],
        )
        ratings_df = read_from_csv(
            path=os.path.join(MOVIELENS_PATH, "ratings.dat"),
            columns=["user_id", "movie_id", "rating", "timestamp"],
        )
    
    app.state.movies = movies_df
    app.state.ratings = ratings_df
    app.state.users = users_df
    app.state.recommender = Recommender(
        model_type=settings.MODEL_TYPE,
        source=settings.SOURCE,
        movies=app.state.movies,
        rating_threshold=settings.RATING_THRESHOLD,
        model_path=model_path
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
        raise HTTPException(status_code=501, detail=e)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recsys.api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
