import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from recsys.aggregates import Movie
from recsys.config import MOVIELENS_PATH
from recsys.utils import read_from_csv
from recsys.db.repositories.ratings import RatingsRepository
from recsys.gcp import GCPModelStorage
from recsys.modeling.protocols import RecommenderModel

@dataclass
class AlternatingLeastSquaresRecommender(RecommenderModel):
    threshold: int
    model_path: str
    x_ui_path: str
    mappings_path: str
    ratings_repo: RatingsRepository
    storage: GCPModelStorage
    model: AlternatingLeastSquares = field(default_factory=AlternatingLeastSquares)
    X_ui: csr_matrix = field(default_factory=dict, repr=False)
    user2idx: Dict[int, int] = field(default_factory=dict, repr=False)
    idx2user: Dict[int, int] = field(default_factory=dict, repr=False)
    item2idx: Dict[int, int] = field(default_factory=dict, repr=False)
    idx2item: Dict[int, int] = field(default_factory=dict, repr=False)

    async def preload(self) -> None:
        self.model = await self.storage.load_als_model(self.model_path)
        self.X_ui = await self.storage.load_csr_npz(self.x_ui_path)
        meta = await self.storage.load_json(self.mappings_path)
        self.user2idx = {int(k): int(v) for k, v in meta["user2idx"].items()}
        self.idx2item = {int(k): int(v) for k, v in meta["idx2item"].items()}
        self.item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}

    async def fit(self, df: pd.DataFrame):
        df = await self.ratings_repo.fetch_ratings_df(min_rating=self.threshold)
        self._set_matrix(df=df)
        item_users = self.X_ui

        self.model = AlternatingLeastSquares(**self.best_params)
        self.model.fit(item_users)

    async def save(self):
        # self.model.save(self.model_path)

        await self.storage.save_als_model(self.model_path, self.model)
        await self.storage.save_csr_npz(self.x_ui_path, self.X_ui)
        await self.storage.save_json(self.mappings_path, {
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
            "threshold": self.threshold,
            "params": self.best_params,
        })

    async def recommend(self, user_id: int, n_records: int = 10) -> List[int]:
        user_id = int(user_id)

        if self.user2idx is not None and user_id not in self.user2idx:
            return []

        uidx = self.user2idx[user_id]

        # user_items must be 1 row for implicit==0.7.2
        user_items = self.X_ui[uidx]

        item_idxs, _ = self.model.recommend(
            userid=uidx,
            user_items=user_items,
            N=int(n_records),
            filter_already_liked_items=True,
            recalculate_user=False,
        )

        recs = []
        n_items = self.X_ui.shape[1]
        seen_movie_ids = await self.ratings_repo.fetch_user_seen_movie_ids(
            user_id=user_id,
            min_rating=self.threshold
        )
        seen_iidx = {self.item2idx[m] for m in seen_movie_ids if m in self.item2idx}

        for ii in item_idxs:
            ii = int(ii)

            if ii < 0 or ii >= n_items:
                raise RuntimeError(
                    f"ALS returned out-of-range item index {ii}, but n_items={n_items}. "
                    "This means the model and X_ui are from different runs."
                )

            if ii in seen_iidx:
                continue

            recs.append(self.idx2item[ii])
            if len(recs) == n_records:
                break

        return recs

    def _set_matrix(self, df: pd.DataFrame) -> None:
        u_uniques = np.sort(df["user_id"].unique())
        i_uniques = np.sort(df["movie_id"].unique())

        self.user2idx = {int(u): int(i) for i, u in enumerate(u_uniques)}
        self.idx2user = {int(i): int(u) for i, u in enumerate(u_uniques)}
        self.item2idx = {int(m): int(i) for i, m in enumerate(i_uniques)}
        self.idx2item = {int(i): int(m) for i, m in enumerate(i_uniques)}

        rows = df["user_id"].map(self.user2idx).to_numpy(np.int32)
        cols = df["movie_id"].map(self.item2idx).to_numpy(np.int32)

        self.X_ui = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (rows, cols)),
            shape=(len(u_uniques), len(i_uniques)),
        )

    @property
    def best_params(self) -> Dict[str, float]:
        return {
            "factors": 128,
            "regularization": 0.005562375053545441,
            "alpha": 5,
            "iterations": 30,
            "random_state": 42,
        }
