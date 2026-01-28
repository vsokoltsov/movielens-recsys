import os
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix, diags
from recsys.config import MODELS
from recsys.gcp import GCPModelStorage
from recsys.db.repositories.ratings import RatingsRepository
from recsys.modeling.protocols import RecommenderModel

class ItemKNNRecommender(RecommenderModel):
    def __init__(self,
        storage: GCPModelStorage,
        artifact_prefix: str,
        ratings_repo: Optional[RatingsRepository] = None,
        k_neighbors=200, 
        threshold=4):


        self.storage = storage
        self.ratings_repo = ratings_repo
        self.artifact_prefix = artifact_prefix
        self.k_neighbors = k_neighbors
        self.threshold = threshold

        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        self.X_ui: Optional[csr_matrix] = None  # user-item
        self.S_ii: Optional[csr_matrix] = None  # item-item similarity (sparse)

    async def preload(self) -> None:
        meta = await self.storage.load_json(f"{self.artifact_prefix}/meta.json")

        self.threshold = int(meta["threshold"])
        self.k_neighbors = int(meta["k_neighbors"])
        self.user2idx = {int(k): int(v) for k, v in meta["user2idx"].items()}
        self.item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}
        self.idx2item = {int(k): int(v) for k, v in meta["idx2item"].items()}

        self.X_ui = await self.storage.load_csr_npz(f"{self.artifact_prefix}/X_ui.npz")
        self.S_ii = await self.storage.load_csr_npz(f"{self.artifact_prefix}/S_ii.npz")

    async def fit(self, limit: Optional[int] = None):
        if not self.ratings_repo:
            raise ValueError("ratings repository is not defined")
            
        df = await self.ratings_repo.fetch_ratings_df(min_rating=self.threshold, limit=limit)
        if df.empty:
            self.user2idx = {}
            self.item2idx = {}
            self.idx2item = {}
            self.X_ui = csr_matrix((0, 0), dtype=np.float32)
            self.S_ii = csr_matrix((0, 0), dtype=np.float32)
            await self.save()
            return
    
        # factorize
        u_codes, u_uniques = pd.factorize(df["user_id"], sort=True)
        i_codes, i_uniques = pd.factorize(df["movie_id"], sort=True)

        self.user2idx = {int(u): int(i) for i, u in enumerate(u_uniques)}
        self.item2idx = {int(m): int(i) for i, m in enumerate(i_uniques)}
        self.idx2item = {int(i): int(m) for i, m in enumerate(i_uniques)}

        n_users = len(u_uniques)
        n_items = len(i_uniques)

        self.X_ui = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (u_codes, i_codes)),
            shape=(n_users, n_items),
        )

        self._build_similarity()

    async def save(self) -> None:
        if self.X_ui is None or self.S_ii is None:
            raise RuntimeError("Nothing to save: X_ui/S_ii not built.")

        await self.storage.save_csr_npz(f"{self.artifact_prefix}/X_ui.npz", self.X_ui.tocsr())
        await self.storage.save_csr_npz(f"{self.artifact_prefix}/S_ii.npz", self.S_ii.tocsr())

        meta = {
            "k_neighbors": int(self.k_neighbors),
            "threshold": int(self.threshold),
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
        }
        await self.storage.save_json(f"{self.artifact_prefix}/meta.json", meta)

    def _build_similarity(self):
        X_iu = self.X_ui.T.tocsr()  # (n_items, n_users)
        S = (X_iu @ X_iu.T).tocsr()  # co-occurrence

        # cosine normalize
        item_norm = np.sqrt(S.diagonal())
        item_norm[item_norm == 0] = 1.0
        D_inv = diags(1.0 / item_norm)
        S = (D_inv @ S @ D_inv).tocsr()
        S.setdiag(0.0)

        # leaving only top-k_neighbors for each item (speeds up and improves)
        if self.k_neighbors is not None:
            S = self._topk_per_row(S, self.k_neighbors)

        S.eliminate_zeros()
        self.S_ii = S

    @staticmethod
    def _topk_per_row(S: csr_matrix, k: int) -> csr_matrix:
        S = S.tolil()
        for i in range(S.shape[0]):
            row_data = np.array(S.data[i])
            row_cols = np.array(S.rows[i])
            if len(row_data) > k:
                idx = np.argpartition(-row_data, k)[:k]
                S.data[i] = row_data[idx].tolist()
                S.rows[i] = row_cols[idx].tolist()
        return S.tocsr()

    async def recommend(self, user_id: int, n_records: int = 10) -> List[int]:
        if self.X_ui is None or self.S_ii is None:
            raise RuntimeError("Model is not loaded. Call preload() or fit() first.")

        user_id = int(user_id)
        if user_id not in self.user2idx:
            return []

        uidx = self.user2idx[user_id]
        user_row = self.X_ui.getrow(uidx)
        seen = set(user_row.indices)
        if not seen:
            return []

        # (1, n_items) @ (n_items, n_items) -> (1, n_items)
        scores = (user_row @ self.S_ii).toarray().ravel()

        if seen:
            scores[list(seen)] = -np.inf

        n_records = int(n_records)
        n_records = min(n_records, scores.size)

        top = np.argpartition(-scores, n_records - 1)[:n_records]
        top = top[np.argsort(-scores[top])]
        return [self.idx2item[int(i)] for i in top]