import asyncio
from typing import Optional, List

import click

from recsys.config import get_settings
from recsys.aggregates import Source, Setup
from recsys.aggregates import Movie
from recsys.modeling.dataset import Dataset
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.aggregates import ModelType
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.torch import PytorchRecommender
from recsys.storage import GCPModelStorage, LocalStorage, StorageProtocol
from recsys.db.session import build_sessionmaker, session_scope
from recsys.repository.protocols import (
    RatingsRepositoryProtocol,
    MoviesRepositoryProtocol,
)
from recsys.repository.db.ratings import RatingsDBRepository
from recsys.repository.db.movies import MoviesDBRepository
from recsys.repository.csv.ratings import RatingsCSVRepository
from recsys.repository.csv.movies import MoviesCSVRepository


async def _run_prediction(
    model_type: ModelType,
    user_id: int,
    ratings_repo: RatingsRepositoryProtocol,
    storage: StorageProtocol,
) -> List[int]:
    recommendations: List[int] = []
    if model_type == ModelType.ALS:
        model = AlternatingLeastSquaresRecommender(
            ratings_repo=ratings_repo,
            storage=storage,
            threshold=4,
            model_path="als/latest/model.npz",
            x_ui_path="als/latest/x_ui.npz",
            mappings_path="als/latest/mappings.json",
        )
        await model.preload()
        recommendations = await model.recommend(user_id=user_id)
    elif model_type == ModelType.ITEM_KNN:
        model_ = ItemKNNRecommender(
            ratings_repo=ratings_repo,
            storage=storage,
            artifact_prefix="item_knn/v1",
            k_neighbors=200,
            threshold=4,
        )
        await model_.preload()
        recommendations = await model_.recommend(int(user_id))
    elif model_type == ModelType.PYTORCH:
        ptr = PytorchRecommender(
            storage=storage,
            model_path="pytorch/latest/model.pt",
            x_ui_path="pytorch/latest/x_ui.npz",
            mappings_path="pytorch/latest/mappings.json",
            ratings_repo=ratings_repo,
        )
        await ptr.preload()
        recommendations = await ptr.recommend(int(user_id))
    return recommendations


async def predict(user_id: int, model_type: str, bucket_name: Optional[str]):
    settings = get_settings()
    storage: StorageProtocol
    ratings_repo: RatingsRepositoryProtocol
    movies_repo: MoviesRepositoryProtocol
    recommendations: List[int] = []

    if settings.SETUP == Setup.LOCAL:
        storage = LocalStorage(model_path=settings.MODELS_PATH)
    elif settings.SETUP == Setup.CLOUD:
        storage = GCPModelStorage(bucket_name=bucket_name or settings.MODEL_BUCKET)

    if settings.SOURCE == Source.DB:
        engine, session_local = build_sessionmaker(database_url=settings.DATABASE_URL)
        try:
            async with session_scope(session_local) as session:
                ratings_repo = RatingsDBRepository(session=session)
                movies_repo = MoviesDBRepository(session=session)
                recommendations = await _run_prediction(
                    model_type=ModelType(model_type),
                    user_id=user_id,
                    ratings_repo=ratings_repo,
                    storage=storage,
                )
        finally:
            await engine.dispose()
    elif settings.SOURCE == Source.CSV:
        dataset = Dataset(dataset_path=settings.MOVIELENS_PATH)
        ratings_repo = RatingsCSVRepository(df=dataset.ratings)
        movies_repo = MoviesCSVRepository(df=dataset.movies)
        recommendations = await _run_prediction(
            model_type=ModelType(model_type),
            user_id=user_id,
            ratings_repo=ratings_repo,
            storage=storage,
        )
    else:
        raise ValueError("Invalid source")

    movies = await movies_repo.fetch_movies_by_ids(recommendations)

    by_id = {int(m.id): m for m in movies}
    ordered = [by_id[mid] for mid in recommendations if mid in by_id]
    click.echo(
        [
            Movie(id=int(rec.id), title=str(rec.title), genre=str(rec.genre))
            for rec in ordered
        ]
    )

    # env_bucket = bucket_name
    # if not env_bucket:
    #     env_bucket = settings.MODEL_BUCKET
    # if settings.MOVIELENS_PATH:
    #     dataset = Dataset(dataset_path=settings.MOVIELENS_PATH)
    # engine, session_local = build_sessionmaker(database_url=settings.DATABASE_URL)
    # async with session_scope(session_local) as session:
    #     ratings_repo = RatingsRepository(session)
    #     model_storage = GCPModelStorage(bucket_name=str(env_bucket))
    #     recommendations: List[int] = []
    #     if model_type == ModelType.ALS:
    #         model = AlternatingLeastSquaresRecommender(
    #             ratings_repo=ratings_repo,
    #             storage=model_storage,
    #             threshold=4,
    #             model_path="als/latest/model.npz",
    #             x_ui_path="als/latest/x_ui.npz",
    #             mappings_path="als/latest/mappings.json",
    #         )
    #         await model.preload()
    #         recommendations = await model.recommend(user_id=user_id)
    #     elif model_type == ModelType.ITEM_KNN:
    #         model_ = ItemKNNRecommender(
    #             ratings_repo=ratings_repo,
    #             storage=model_storage,
    #             artifact_prefix="item_knn/v1",
    #             k_neighbors=200,
    #             threshold=4,
    #         )
    #         await model_.preload()
    #         recommendations = await model_.recommend(int(user_id))
    #     elif model_type == ModelType.PYTORCH:
    #         ptr = PytorchRecommender(
    #             storage=model_storage,
    #             model_path="pytorch/latest/model.pt",
    #             x_ui_path="pytorch/latest/x_ui.npz",
    #             mappings_path="pytorch/latest/mappings.json",
    #             ratings_repo=ratings_repo,
    #         )
    #         await ptr.preload()
    #         recommendations = await ptr.recommend(int(user_id))

    # if len(recommendations) > 0:
    #     movies = dataset.movies.iloc[recommendations].to_dict(orient="records")
    #     click.echo(movies)
    # else:
    #     click.echo("No suggested recommendations")
    #     await engine.dispose()


@click.command
@click.argument("user_id")
@click.option(
    "--model-type",
    default="als",
    help='Type of model. Available values are "als", "item_knn", "pytorch"',
)
@click.option(
    "--bucket-name",
    help="Use of Google Cloud Storage",
)
def main(user_id: int, model_type: str, bucket_name: Optional[str]) -> None:
    asyncio.run(
        predict(user_id=user_id, model_type=model_type, bucket_name=bucket_name)
    )


if __name__ == "__main__":
    main()
