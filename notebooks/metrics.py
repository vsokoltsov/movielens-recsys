import math
import numpy as np
import pandas as pd
from typing import Callable, Iterable, Dict, List, Optional

RecommendFn = Callable[[int, int], List[int]]  # recommend(user_id, k) -> list[item_id]


# --------- Core ranking metrics ---------

def recall_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    recs = recommended[:k]
    return len(set(recs) & set(relevant)) / len(set(relevant))


def dcg_at_k(recommended: List[int], relevant_set: set, k: int = 10) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    dcg = dcg_at_k(recommended, rel_set, k=k)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
    rel_set = set(relevant)
    for rank, item in enumerate(recommended[:k], start=1):
        if item in rel_set:
            return 1.0 / rank
    return 0.0


def hitrate_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    return 1.0 if any(item in rel_set for item in recommended[:k]) else 0.0


# --------- Dataset-level evaluators ---------

def _get_test_positives(test_df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    tmp = test_df.copy()
    tmp["interaction"] = (tmp["rating"] >= threshold).astype(np.int8)
    return tmp[tmp["interaction"] == 1]


def evaluate_ranking_metrics(
    recommend: RecommendFn,
    test_df: pd.DataFrame,
    k: int = 10,
    threshold: int = 4,
    user_col: str = "user_id",
    item_col: str = "movie_id",
) -> Dict[str, float]:
    """
    Average value Recall@K / NDCG@K / MRR@K / HitRate@K per user.
    """
    test_pos = _get_test_positives(test_df, threshold=threshold)

    recalls, ndcgs, mrrs, hits = [], [], [], []
    for user_id, g in test_pos.groupby(user_col):
        relevant = g[item_col].tolist()
        recs = recommend(int(user_id), k)

        if not recs or not relevant:
            continue

        recalls.append(recall_at_k(recs, relevant, k=k))
        ndcgs.append(ndcg_at_k(recs, relevant, k=k))
        mrrs.append(mrr_at_k(recs, relevant, k=k))
        hits.append(hitrate_at_k(recs, relevant, k=k))

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"hitrate@{k}": float(np.mean(hits)) if hits else 0.0,
        "n_users_eval": int(len(recalls)),
    }


def coverage_at_k(
    recommend: RecommendFn,
    users: Iterable[int],
    all_items: Optional[Iterable[int]] = None,
    k: int = 10,
) -> float:
    """
    Coverage@K = |unique recommended items| / |catalog size|
    all_items: list of all item_id (movies["movie_id"].unique()).
              if None, coverage is share of unique recommended.
    """
    rec_items = set()
    for u in users:
        rec_items.update(recommend(int(u), k))

    if all_items is None:
        return float(len(rec_items))

    catalog = set(all_items)
    return 0.0 if not catalog else len(rec_items) / len(catalog)


def avg_popularity_at_k(
    recommend: RecommendFn,
    users: Iterable[int],
    item_popularity: Dict[int, float],
    k: int = 10,
) -> float:
    """
    Average popularity of recommended items.
    item_popularity: dict {item_id: count} или {item_id: popularity_score}
    """
    vals = []
    for u in users:
        recs = recommend(int(u), k)
        vals.extend([float(item_popularity.get(i, 0.0)) for i in recs])
    return float(np.mean(vals)) if vals else 0.0


# --------- One "universal" runner ---------

def evaluate_all(
    recommend: RecommendFn,
    test_df: pd.DataFrame,
    users_for_coverage: Optional[Iterable[int]] = None,
    all_items: Optional[Iterable[int]] = None,
    item_popularity: Optional[Dict[int, float]] = None,
    k: int = 10,
    threshold: int = 4,
) -> Dict[str, float]:
    """
    Collect all metrics
    """
    out = evaluate_ranking_metrics(recommend, test_df, k=k, threshold=threshold)

    if users_for_coverage is not None:
        out[f"coverage@{k}"] = coverage_at_k(recommend, users_for_coverage, all_items=all_items, k=k)

    if item_popularity is not None and users_for_coverage is not None:
        out[f"avg_popularity@{k}"] = avg_popularity_at_k(recommend, users_for_coverage, item_popularity, k=k)

    return out