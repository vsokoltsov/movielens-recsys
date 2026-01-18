import os
import pandas as pd
from typing import List, Union, Optional
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix

from recsys.aggregates import Movie, ModelType, Source
from recsys.config import MOVIELENS_PATH
from recsys.utils import read_from_csv
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.item_knn import ItemKNNRecommender

@dataclass
class Recommender:
    model_type: ModelType
    source: Source
    movies: pd.DataFrame
    rating_threshold: int
    model_path: str
    model: Optional[
        Union[AlternatingLeastSquaresRecommender, ItemKNNRecommender]
    ] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.model_type == ModelType.ALS:
            self.model = AlternatingLeastSquaresRecommender(
                threshold=self.rating_threshold,
                model_path=self.model_path
            )
        elif self.model_type == ModelType.ITEM_KNN:
            self.model = ItemKNNRecommender.load(self.model_path)
 

    def recommend(self, user_id: int, n_items: int = 10) -> List[Movie]:
        if self.model is None:
            raise ValueError("model is not initialized")

        recs = self.model.recommend(user_id=user_id, n_records=n_items)
        rec_movies = self.movies.iloc[recs].to_dict(orient='records')
        return [
            Movie(
                id=rec['movie_id'],
                title=rec['title'],
                genre=rec['genres']
            ) for rec in rec_movies
        ]
        