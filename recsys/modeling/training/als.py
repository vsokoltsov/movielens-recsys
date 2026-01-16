import os
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, diags
from recsys.config import MODELS

def train_als(train_df: pd.DataFrame):
    params = {
        'factors': 128,
        'regularization': 0.005562375053545441,
        'alpha': 5,
        'iterations': 30,
        'random_state': 42
    }
    train_tmp = train_df.copy()
    train_tmp["interaction"] = (train_tmp["rating"] >= 4).astype(np.int8)
    train_pos = train_tmp[train_tmp["interaction"] == 1][["user_id", "movie_id"]].copy()
    u_codes, u_uniques = pd.factorize(train_pos["user_id"], sort=True)
    i_codes, i_uniques = pd.factorize(train_pos["movie_id"], sort=True)
    train_pos["u_idx"] = u_codes.astype(np.int32)
    train_pos["i_idx"] = i_codes.astype(np.int32)

    X_ui = csr_matrix(
        (np.ones(len(train_pos), dtype=np.float32),
        (train_pos["u_idx"].to_numpy(), train_pos["i_idx"].to_numpy())),
        shape=(len(u_uniques), len(i_uniques)),
    )
    item_users = X_ui

    als = AlternatingLeastSquares(
        **params
    )

    als.fit(item_users)
    als.save(os.path.join(MODELS, "alternating_least_squares"))
    return als