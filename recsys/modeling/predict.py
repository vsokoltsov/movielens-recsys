import os
import click

from recsys.modeling.dataset import Dataset
from recsys.modeling.training.als import recommend_als
from recsys.modeling.training.item_knn import ItemKNNRecommender
from recsys.config import MODELS, MOVIELENS_PATH


@click.command
@click.argument("user_id")
@click.option(
    "--model-type",
    default="als",
    help='Type of model. Available values are "als", "item_knn", "pytorch"',
)
def predict(user_id: int, model_type: str):
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    if model_type == "als":
        recommendations = recommend_als(dataset.ratings, user_id)
    elif model_type == "item_knn":
        model = ItemKNNRecommender.load(
            os.path.join(MODELS, "item_knn_recommender.pkl")
        )
        recommendations = model.recommend(int(user_id))

    movies = dataset.movies.iloc[recommendations].to_dict(orient="records")
    click.echo(movies)


if __name__ == "__main__":
    predict()
