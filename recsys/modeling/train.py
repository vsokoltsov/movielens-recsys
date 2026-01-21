import asyncio
import os
from typing import Optional

import click
from dotenv import load_dotenv

from recsys.config import MOVIELENS_PATH, MODELS
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.modeling.dataset import Dataset
from recsys.gcp import GCPModelStorage
from recsys.db.session import AsyncSessionLocal
from recsys.db.repositories.ratings import RatingsRepository

async def train_model(model_type: str, bucket_name: Optional[str]):
    load_dotenv()

    env_bucket = os.environ.get('MODEL_BUCKET', bucket_name)
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    async with AsyncSessionLocal() as session:
        ratings_repo = RatingsRepository(session)
        model_storage = GCPModelStorage(bucket_name=str(env_bucket))
        if model_type == "als":
            click.echo("Train alternating least squares model...")
            als = AlternatingLeastSquaresRecommender(
                ratings_repo=ratings_repo,
                storage=model_storage,
                threshold=4,
                model_path="als/latest/model.npz",
                x_ui_path="als/latest/x_ui.npz",
                mappings_path="als/latest/mappings.json",
            )
            await als.fit(dataset.ratings)
            await als.save()
        elif model_type == "item_knn":
            click.echo("Train item knn model...")
            knn = ItemKNNRecommender(
                ratings_repo=ratings_repo,
                storage=model_storage,
                artifact_prefix="item_knn/v1",
                k_neighbors=200, 
                threshold=4
            )
            await knn.fit()
            await knn.save()

@click.command()
@click.option(
    "--model-type",
    default="als",
    help='Type of model. Available values are "als", "item_knn", "pytorch"',
)
@click.option(
    "--bucket-name",
    help='Use of Google Cloud Storage',
)
def main(model_type: str, bucket_name: str) -> None:
    asyncio.run(train_model(model_type, bucket_name))

if __name__ == "__main__":
    main()
