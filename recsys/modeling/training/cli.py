import click
from recsys.config import MOVIELENS_PATH
from recsys.modeling.training.als import train_als
from recsys.modeling.training.item_knn import train_item_knn
from recsys.modeling.training.utils import temporal_split_per_user
from recsys.dataset import Dataset

@click.command()
@click.option('--model-type', default='als', help='Type of model. Available values are "als", "item_knn", "pytorch"')
def train_model(model_type: str):
    dataset = Dataset(dataset_path=MOVIELENS_PATH)
    train_df, _, _ = temporal_split_per_user(dataset.ratings, n_val=1, n_test=5, min_train=5)
    if model_type == 'als':
        click.echo("Train alternating least squares model...")
        train_als(train_df)
    elif model_type == 'item_knn':
        click.echo("Train item knn model...")
        train_item_knn(train_df)



if __name__ == '__main__':
    train_model()