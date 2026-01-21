import asyncio
import os
from typing import Optional

import click
from dotenv import load_dotenv

from recsys.modeling.dataset import Dataset
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.config import MODELS, MOVIELENS_PATH
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.gcp import GCPModelStorage
from recsys.db.session import AsyncSessionLocal
from recsys.db.repositories.ratings import RatingsRepository



async def predict(user_id: int, model_type: str, bucket_name: Optional[str]):
    load_dotenv()
    
    env_bucket = str(os.environ.get('MODEL_BUCKET', bucket_name))
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    async with AsyncSessionLocal() as session:
        ratings_repo = RatingsRepository(session)
        model_storage = GCPModelStorage(bucket_name=str(env_bucket))
        if model_type == "als":
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
        elif model_type == "item_knn":
            model_ = ItemKNNRecommender(
                ratings_repo=ratings_repo,
                storage=model_storage,
                artifact_prefix="item_knn/v1",
                k_neighbors=200, 
                threshold=4
            )
            await model_.preload()
            recommendations = await model_.recommend(int(user_id))

        movies = dataset.movies.iloc[recommendations].to_dict(orient="records")
        click.echo(movies)

@click.command
@click.argument("user_id")
@click.option(
    "--model-type",
    default="als",
    help='Type of model. Available values are "als", "item_knn", "pytorch"',
)
@click.option(
    "--bucket-name",
    help='Use of Google Cloud Storage',
)
def main(user_id: int, model_type: str, bucket_name: Optional[str]) -> None:
    asyncio.run(predict(
        user_id=user_id,
        model_type=model_type,
        bucket_name=bucket_name
    ))

if __name__ == "__main__":
    main()
