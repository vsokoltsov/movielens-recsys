import os
from typing import Optional

import click
from dotenv import load_dotenv

from recsys.config import MOVIELENS_PATH, MODELS
from recsys.modeling.training.item_knn import train_item_knn
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.modeling.dataset import Dataset
from recsys.gcp import GCPStorageClient


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

def train_model(model_type: str, bucket_name: Optional[str]):
    load_dotenv()

    env_bucket = os.environ.get('MODEL_BUCKET', bucket_name)
    gcs_client = GCPStorageClient()
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    if model_type == "als":
        model_name = "alternating_least_squares.npz"
        model_path = os.path.join(MODELS, "alternating_least_squares.npz")
        click.echo("Train alternating least squares model...")
        als = AlternatingLeastSquaresRecommender(
            threshold=4,
            model_path=model_path
        )
        als.fit(dataset.ratings)
        als.save()
        
        
    elif model_type == "item_knn":
        model_name = "item_knn_recommender.pkl"
        model_path = os.path.join("/tmp", model_name)
        click.echo("Train item knn model...")
        knn = ItemKNNRecommender(k_neighbors=200, threshold=4)
        knn.fit(dataset.ratings)
        knn.save(model_path)

    if env_bucket is not None:
            gcs_client.upload(
                path=model_path,
                bucket_name=env_bucket,
                object_name=model_name
            )


if __name__ == "__main__":
    train_model()
