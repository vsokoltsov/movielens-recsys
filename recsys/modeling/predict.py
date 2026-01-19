import os
from typing import Optional

import click
from dotenv import load_dotenv

from recsys.modeling.dataset import Dataset
from recsys.modeling.training.als import recommend_als
from recsys.modeling.training.item_knn import ItemKNNRecommender
from recsys.config import MODELS, MOVIELENS_PATH
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.gcp import GCPStorageClient


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
def predict(user_id: int, model_type: str, bucket_name: Optional[str]):
    load_dotenv()
    
    env_bucket = str(os.environ.get('MODEL_BUCKET', bucket_name))
    client = GCPStorageClient()
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    if model_type == "als":
        tmp_path = "/tmp/alternating_least_squares.npz"
        client.download(
            bucket_name=env_bucket,
            object_name="alternating_least_squares.npz",
            dst_path=tmp_path
        )
        model = AlternatingLeastSquaresRecommender(
            threshold=4,
            model_path=tmp_path
        )
        model.preload()
        recommendations = model.recommend(user_id=user_id)
    elif model_type == "item_knn":
        model_ = ItemKNNRecommender.load(
            os.path.join(MODELS, "item_knn_recommender.pkl")
        )
        recommendations = model_.recommend(int(user_id))

    movies = dataset.movies.iloc[recommendations].to_dict(orient="records")
    click.echo(movies)


if __name__ == "__main__":
    predict()
