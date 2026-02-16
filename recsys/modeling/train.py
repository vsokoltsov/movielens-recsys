import asyncio
import pandas as pd
from typing import Optional

import click

from recsys.config import get_settings, Source, Setup
from recsys.aggregates import ModelType
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.modeling.torch import PytorchRecommender
from recsys.modeling.dataset import Dataset
from recsys.storage import GCPModelStorage, LocalStorage, StorageProtocol
from recsys.db.session import build_sessionmaker, session_scope
from recsys.repository.protocols import RatingsRepositoryProtocol
from recsys.repository.db.ratings import RatingsDBRepository
from recsys.repository.csv.ratings import RatingsCSVRepository


async def _run_training(
    model_type: ModelType,
    ratings_df: pd.DataFrame,
    ratings_repo: RatingsRepositoryProtocol,
    storage: StorageProtocol,
):
    if model_type == ModelType.ALS:
        click.echo("Train alternating least squares model...")
        als = AlternatingLeastSquaresRecommender(
            ratings_repo=ratings_repo,
            storage=storage,
            threshold=4,
            model_path="als/latest/model.npz",
            x_ui_path="als/latest/x_ui.npz",
            mappings_path="als/latest/mappings.json",
        )
        await als.fit(ratings_df)
        await als.save()
    elif model_type == ModelType.ITEM_KNN:
        click.echo("Train item knn model...")
        knn = ItemKNNRecommender(
            ratings_repo=ratings_repo,
            storage=storage,
            artifact_prefix="item_knn/v1",
            k_neighbors=200,
            threshold=4,
        )
        await knn.fit()
        await knn.save()
    elif model_type == ModelType.PYTORCH:
        click.echo("Train pytorch model...")
        ptr = PytorchRecommender(
            storage=storage,
            model_path="pytorch/latest/model.pt",
            x_ui_path="pytorch/latest/x_ui.npz",
            mappings_path="pytorch/latest/mappings.json",
            ratings_repo=ratings_repo,
        )
        await ptr.fit(ratings_df)
        await ptr.save()


async def train_model(model_type: str, bucket_name: Optional[str]):
    settings = get_settings()
    storage: StorageProtocol
    ratings_repo: RatingsRepositoryProtocol

    if settings.SETUP == Setup.LOCAL:
        storage = LocalStorage(model_path=settings.MODELS_PATH)
    elif settings.SETUP == Setup.CLOUD:
        storage = GCPModelStorage(bucket_name=bucket_name or settings.MODEL_BUCKET)

    if settings.SOURCE == Source.DB:
        engine, session_local = build_sessionmaker(database_url=settings.DATABASE_URL)
        try:
            async with session_scope(session_local) as session:
                ratings_repo = RatingsDBRepository(session=session)
                ratings_df = await ratings_repo.fetch_ratings_df(min_rating=0)
                await _run_training(
                    model_type=ModelType(model_type),
                    ratings_df=ratings_df,
                    ratings_repo=ratings_repo,
                    storage=storage,
                )
        finally:
            await engine.dispose()
    elif settings.SOURCE == Source.CSV:
        dataset = Dataset(dataset_path=settings.MOVIELENS_PATH)
        ratings_repo = RatingsCSVRepository(df=dataset.ratings)
        await _run_training(
            model_type=ModelType(model_type),
            ratings_df=dataset.ratings,
            ratings_repo=ratings_repo,
            storage=storage,
        )
    else:
        raise ValueError("Invalid source")


@click.command()
@click.option(
    "--model-type",
    default="als",
    help='Type of model. Available values are "als", "item_knn", "pytorch"',
)
@click.option(
    "--bucket-name",
    help="Use of Google Cloud Storage",
)
def main(model_type: str, bucket_name: str) -> None:
    asyncio.run(train_model(model_type, bucket_name))


if __name__ == "__main__":
    main()
