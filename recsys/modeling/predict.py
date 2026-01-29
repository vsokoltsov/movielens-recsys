import asyncio
from typing import Optional, List

import click
from dotenv import load_dotenv

from recsys.modeling.dataset import Dataset
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.config import MOVIELENS_PATH, MODEL_BUCKET, DATABASE_URL
from recsys.aggregates import ModelType
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.torch import PytorchRecommender
from recsys.gcp import GCPModelStorage
from recsys.db.session import build_sessionmaker, session_scope
from recsys.db.repositories.ratings import RatingsRepository


async def predict(user_id: int, model_type: str, bucket_name: Optional[str]):
    load_dotenv()

    env_bucket = bucket_name
    if not env_bucket:
        env_bucket = MODEL_BUCKET
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    engine, session_local = build_sessionmaker(database_url=DATABASE_URL)
    async with session_scope(session_local) as session:
        ratings_repo = RatingsRepository(session)
        model_storage = GCPModelStorage(bucket_name=str(env_bucket))
        recommendations: List[int] = []
        if model_type == ModelType.ALS:
            model = AlternatingLeastSquaresRecommender(
                ratings_repo=ratings_repo,
                storage=model_storage,
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
                storage=model_storage,
                artifact_prefix="item_knn/v1",
                k_neighbors=200,
                threshold=4,
            )
            await model_.preload()
            recommendations = await model_.recommend(int(user_id))
        elif model_type == ModelType.PYTORCH:
            ptr = PytorchRecommender(
                storage=model_storage,
                model_path="pytorch/latest/model.pt",
                x_ui_path="pytorch/latest/x_ui.npz",
                mappings_path="pytorch/latest/mappings.json",
                ratings_repo=ratings_repo,
            )
            await ptr.preload()
            recommendations = await ptr.recommend(int(user_id))

        if len(recommendations) > 0:
            movies = dataset.movies.iloc[recommendations].to_dict(orient="records")
            click.echo(movies)
        else:
            click.echo("No suggested recommendations")
        await engine.dispose()


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
