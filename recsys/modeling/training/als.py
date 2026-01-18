import os
import json
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from recsys.config import MODELS


def save_csr_npz(path: str, M: csr_matrix):
    M = M.tocsr()
    np.savez_compressed(
        path,
        data=M.data,
        indices=M.indices,
        indptr=M.indptr,
        shape=np.array(M.shape),
    )


def load_csr_npz(path: str) -> csr_matrix:
    loader = np.load(path, allow_pickle=False)
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        shape=tuple(loader["shape"]),
    )


def save_mappings(path, artifacts):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f)


def load_mappings(path):
    artifacts = {}
    with open(path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    user2idx = {int(k): int(v) for k, v in artifacts["user2idx"].items()}
    idx2user = {int(k): int(v) for k, v in artifacts["idx2user"].items()}
    item2idx = {int(k): int(v) for k, v in artifacts["item2idx"].items()}
    idx2item = {int(k): int(v) for k, v in artifacts["idx2item"].items()}
    return {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }


def train_als(train_df: pd.DataFrame):
    params = {
        "factors": 128,
        "regularization": 0.005562375053545441,
        "alpha": 5,
        "iterations": 30,
        "random_state": 42,
    }
    train_tmp = train_df.copy()
    train_tmp["interaction"] = (train_tmp["rating"] >= 4).astype(np.int8)
    train_pos = train_tmp[train_tmp["interaction"] == 1][["user_id", "movie_id"]].copy()
    u_codes, u_uniques = pd.factorize(train_pos["user_id"], sort=True)
    i_codes, i_uniques = pd.factorize(train_pos["movie_id"], sort=True)
    user2idx = pd.Series(np.arange(len(u_uniques)), index=u_uniques).to_dict()
    idx2user = pd.Series(u_uniques).to_dict()

    idx2item = pd.Series(i_uniques).to_dict()  # i_idx -> movie_id
    item2idx = pd.Series(np.arange(len(i_uniques)), index=i_uniques).to_dict()
    train_pos["u_idx"] = u_codes.astype(np.int32)
    train_pos["i_idx"] = i_codes.astype(np.int32)
    artifacts = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }

    X_ui = csr_matrix(
        (
            np.ones(len(train_pos), dtype=np.float32),
            (train_pos["u_idx"].to_numpy(), train_pos["i_idx"].to_numpy()),
        ),
        shape=(len(u_uniques), len(i_uniques)),
    )
    item_users = X_ui

    als = AlternatingLeastSquares(**params)

    als.fit(item_users)
    als.save(os.path.join(MODELS, "alternating_least_squares"))
    save_csr_npz(os.path.join(MODELS, "x_ui.npz"), X_ui)
    save_mappings(os.path.join(MODELS, "mappings.json"), artifacts)
    return als


def recommend_als(ratings_df, user_id: int, k: int = 10):
    model = AlternatingLeastSquares()
    model = model.load(os.path.join(MODELS, "alternating_least_squares.npz"))
    # X_ui = load_csr_npz(os.path.join(MODELS, "x_ui.npz"))
    mappings = load_mappings(os.path.join(MODELS, "mappings.json"))

    train_tmp = ratings_df.copy()
    train_tmp["interaction"] = (train_tmp["rating"] >= 4).astype(np.int8)
    train_pos = train_tmp[train_tmp["interaction"] == 1][["user_id", "movie_id"]].copy()
    u_codes, u_uniques = pd.factorize(train_pos["user_id"], sort=True)
    i_codes, i_uniques = pd.factorize(train_pos["movie_id"], sort=True)
    user2idx = pd.Series(np.arange(len(u_uniques)), index=u_uniques).to_dict()
    idx2user = pd.Series(u_uniques).to_dict()

    idx2item = pd.Series(i_uniques).to_dict()  # i_idx -> movie_id
    item2idx = pd.Series(np.arange(len(i_uniques)), index=i_uniques).to_dict()
    train_pos["u_idx"] = u_codes.astype(np.int32)
    train_pos["i_idx"] = i_codes.astype(np.int32)
    user_id = int(user_id)
    if user_id not in user2idx:
        return []

    uidx = user2idx[user_id]

    seen_movie_ids = set(ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"])
    seen_iidx = {item2idx[m] for m in seen_movie_ids if m in item2idx}
    X_ui = csr_matrix(
        (
            np.ones(len(train_pos), dtype=np.float32),
            (train_pos["u_idx"].to_numpy(), train_pos["i_idx"].to_numpy()),
        ),
        shape=(len(u_uniques), len(i_uniques)),
    )
    user_items = X_ui[uidx]

    item_idxs, _ = model.recommend(
        userid=uidx,
        user_items=user_items,
        N=int(k),
        filter_already_liked_items=True,
        recalculate_user=False,
    )

    recs = []
    n_items = X_ui.shape[1]

    for ii in item_idxs:
        ii = int(ii)

        if ii < 0 or ii >= n_items:
            raise RuntimeError(
                f"ALS returned out-of-range item index {ii}, but n_items={n_items}. "
                "This means the model and X_ui are from different runs."
            )

        if ii in seen_iidx:
            continue

        recs.append(idx2item[ii])
        if len(recs) == k:
            break

    return recs
