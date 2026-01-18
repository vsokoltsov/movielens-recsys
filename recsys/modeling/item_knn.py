import os
import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix, diags
from recsys.config import MODELS


def _save_csr(path_prefix: str, M: csr_matrix):
    np.savez_compressed(
        path_prefix,
        data=M.data,
        indices=M.indices,
        indptr=M.indptr,
        shape=np.array(M.shape),
    )


def _load_csr(path_prefix: str) -> csr_matrix:
    loader = np.load(path_prefix + ".npz", allow_pickle=False)
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        shape=tuple(loader["shape"]),
    )


class ItemKNNRecommender:
    def __init__(self, k_neighbors=200, threshold=4):
        self.k_neighbors = k_neighbors
        self.threshold = threshold

        self.user2idx = None
        self.item2idx = None
        self.idx2item = None
        self.X_ui = None  # user-item
        self.S_ii = None  # item-item similarity (sparse)

    def fit(self, ratings_df: pd.DataFrame):
        df = ratings_df.copy()
        df = df[df["rating"] >= self.threshold][["user_id", "movie_id"]]

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

    def recommend(self, user_id: int, n_records: int = 10):
        if user_id not in self.user2idx:
            return []

        uidx = self.user2idx[user_id]
        user_row = self.X_ui.getrow(uidx)
        seen = set(user_row.indices)

        if not seen:
            return []

        scores = (user_row @ self.S_ii).toarray().ravel()
        if seen:
            scores[list(seen)] = -np.inf

        top = np.argpartition(-scores, n_records)[:n_records]
        top = top[np.argsort(-scores[top])]
        return [self.idx2item[int(i)] for i in top]

    def save(self, dir_path: str):
        import os

        os.makedirs(dir_path, exist_ok=True)

        # sparse matrices
        _save_csr(f"{dir_path}/X_ui", self.X_ui.tocsr())
        _save_csr(f"{dir_path}/S_ii", self.S_ii.tocsr())

        # metadata / mappings
        meta = {
            "k_neighbors": self.k_neighbors,
            "threshold": self.threshold,
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
        }
        with open(f"{dir_path}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, dir_path: str) -> "ItemKNNRecommender":
        with open(f"{dir_path}/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(k_neighbors=meta["k_neighbors"], threshold=meta["threshold"])
        # json keys may become strings → convert back to int
        model.user2idx = {int(k): int(v) for k, v in meta["user2idx"].items()}
        model.item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}
        model.idx2item = {int(k): int(v) for k, v in meta["idx2item"].items()}

        model.X_ui = _load_csr(f"{dir_path}/X_ui")
        model.S_ii = _load_csr(f"{dir_path}/S_ii")
        return model

