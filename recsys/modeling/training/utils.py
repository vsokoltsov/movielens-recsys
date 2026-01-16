import pandas as pd

def temporal_split_per_user(
    ratings: pd.DataFrame,
    n_val: int = 1,
    n_test: int = 1,
    min_train: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user time-based split:
    - sort each user's interactions by timestamp
    - last n_test -> test
    - previous n_val -> val
    - rest -> train
    Users with too few interactions are kept in train only.
    """
    r = ratings.sort_values(["user_id", "timestamp"]).copy()

    r["rank"] = r.groupby("user_id").cumcount() + 1
    r["user_cnt"] = r.groupby("user_id")["movie_id"].transform("size")

    eligible = r["user_cnt"] >= (min_train + n_val + n_test)

    test_mask = eligible & (r["rank"] > r["user_cnt"] - n_test)
    val_mask  = eligible & (r["rank"] > r["user_cnt"] - (n_test + n_val)) & ~test_mask
    train_mask = ~test_mask & ~val_mask

    train = r.loc[train_mask].drop(columns=["rank", "user_cnt"])
    val   = r.loc[val_mask].drop(columns=["rank", "user_cnt"])
    test  = r.loc[test_mask].drop(columns=["rank", "user_cnt"])

    return train, val, test
