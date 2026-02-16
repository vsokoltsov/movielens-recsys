import os
from typing import List, Optional
from dataclasses import dataclass, field

from recsys.aggregates import Movie, ModelType, Source
from recsys.modeling.als import AlternatingLeastSquaresRecommender
from recsys.modeling.item_knn import ItemKNNRecommender
from recsys.storage import StorageProtocol
from recsys.repository.protocols import RatingsRepositoryProtocol
from recsys.repository.protocols import MoviesRepositoryProtocol
from recsys.modeling.protocols import RecommenderModel
from recsys.context import RequestContext


@dataclass
class Recommender:
    model_type: ModelType
    source: Source
    rating_threshold: int
    model_path: str
    model: Optional[RecommenderModel] = field(default=None, init=False, repr=False)

    async def preload(self, storage: StorageProtocol) -> None:
        if self.model_type == ModelType.ALS:
            self.model = AlternatingLeastSquaresRecommender(
                storage=storage,
                threshold=self.rating_threshold,
                model_path=os.path.join(self.model_path, "model.npz"),
                x_ui_path=os.path.join(self.model_path, "x_ui.npz"),
                mappings_path=os.path.join(self.model_path, "mappings.json"),
            )
        elif self.model_type == ModelType.ITEM_KNN:
            self.model = ItemKNNRecommender(
                storage=storage,
                artifact_prefix=os.path.join(self.model_path, "item_knn"),
                k_neighbors=200,
                threshold=self.rating_threshold,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        await self.model.preload()

    async def recommend(
        self, ctx: RequestContext, user_id: int, n_items: int = 10
    ) -> List[Movie]:
        ratings_repo: RatingsRepositoryProtocol = ctx.ratings
        movies_repo: MoviesRepositoryProtocol = ctx.movies

        if self.model is None:
            raise ValueError("model is not initialized")

        candidate_ids = await self.model.recommend(user_id=user_id, n_records=n_items)
        seen = await ratings_repo.fetch_user_seen_movie_ids(
            user_id=user_id, min_rating=self.rating_threshold
        )

        filtered_ids = [mid for mid in candidate_ids if mid not in seen]
        filtered_ids = filtered_ids[:n_items]

        if not filtered_ids:
            return []

        movies_orm = await movies_repo.fetch_movies_by_ids(filtered_ids)

        by_id = {int(m.id): m for m in movies_orm}
        ordered = [by_id[mid] for mid in filtered_ids if mid in by_id]

        return [
            Movie(id=int(rec.id), title=str(rec.title), genre=str(rec.genre))
            for rec in ordered
        ]
