from __future__ import annotations

from typing import List, Optional, Dict
from dataclasses import dataclass, field
import onnxruntime as ort
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from recsys.db.repositories.ratings import RatingsRepository
from recsys.gcp import GCPModelStorage
import anyio
import tempfile

class PairDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class NeuralMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        x = torch.cat([ue, ie], dim=1)
        logit = self.mlp(x).squeeze(1)
        return logit

def build_train_pairs_with_negatives(train_pos: pd.DataFrame, X_ui, n_items: int, n_neg: int = 4, seed: int = 42):
    """
    train_pos: DataFrame [u_idx, i_idx] positive interactions
    X_ui: csr_matrix (n_users, n_items) positives as 1
    returns arrays: users, items, labels
    """
    rng = np.random.default_rng(seed)

    pos_u = train_pos["u_idx"].to_numpy()
    pos_i = train_pos["i_idx"].to_numpy()
    n_pos = len(pos_u)

    users = [pos_u]
    items = [pos_i]
    labels = [np.ones(n_pos, dtype=np.float32)]

    neg_users = np.repeat(pos_u, n_neg)
    neg_items = np.empty(n_pos * n_neg, dtype=np.int64)

    idx = 0
    for u in pos_u:
        liked = set(X_ui.getrow(u).indices)
        for _ in range(n_neg):
            j = int(rng.integers(0, n_items))
            while j in liked:
                j = int(rng.integers(0, n_items))
            neg_items[idx] = j
            idx += 1

    users.append(neg_users)
    items.append(neg_items)
    labels.append(np.zeros(len(neg_users), dtype=np.float32))

    users = np.concatenate(users) # type: ignore
    items = np.concatenate(items)  # type: ignore
    labels = np.concatenate(labels)  # type: ignore

    perm = rng.permutation(len(users))
    return users[perm], items[perm], labels[perm]

def train_neural_mf(
    model: nn.Module,
    train_pos: pd.DataFrame,
    X_ui,
    n_items: int,
    n_neg: int = 4,
    epochs: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    weight_decay=1e-6,
):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        users, items, labels = build_train_pairs_with_negatives(
            train_pos=train_pos,
            X_ui=X_ui,
            n_items=n_items,
            n_neg=n_neg,
            seed=42 + epoch,
        )
        ds = PairDataset(users, items, labels)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        model.train()
        total_loss = 0.0
        for u, i, y in dl:
            u, i, y = u.to(device), i.to(device), y.to(device)

            opt.zero_grad()
            logit = model(u, i)
            loss = criterion(logit, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(u)

    return model

@dataclass
class PytorchRecommender:
    model_path: str
    ratings_repo: RatingsRepository
    storage: GCPModelStorage
    x_ui_path: str
    mappings_path: str

    # hyperparams
    emb_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    n_neg: int = 4
    epochs: int = 3
    batch_size: int = 4096
    lr: float = 1e-3
    threshold: int = 4
    input_user: str = field(default="user_id", init=False)
    input_item: str = field(default="item_id", init=False)
    output_name: str = field(default="logit", init=False)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    session: Optional[ort.InferenceSession] = field(default=None, init=False, repr=False)
    model: Optional[nn.Module] = field(default=None, init=False, repr=False)
    X_ui: Optional[csr_matrix] = field(default=None, init=False, repr=False)

    user2idx: Dict[int, int] = field(default_factory=dict, init=False)
    idx2user: Dict[int, int] = field(default_factory=dict, init=False)
    item2idx: Dict[int, int] = field(default_factory=dict, init=False)
    idx2item: Dict[int, int] = field(default_factory=dict, init=False)

    async def preload(self) -> None:
        local_onnx = await self.storage.load_onnx(
            self.model_path
        )
        self.session = ort.InferenceSession(local_onnx, providers=["CPUExecutionProvider"])

        self.X_ui = await self.storage.load_csr_npz(self.x_ui_path)
        meta = await self.storage.load_json(self.mappings_path)
        self.user2idx = {int(k): int(v) for k, v in meta["user2idx"].items()}
        self.idx2item = {int(k): int(v) for k, v in meta["idx2item"].items()}
        self.item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}

    async def fit(self, df: pd.DataFrame):
        ratings_df: pd.DataFrame = await self.ratings_repo.fetch_ratings_df(
            min_rating=self.threshold
        )

        tmp = ratings_df[["user_id", "movie_id", "rating"]].copy()
        tmp["interaction"] = (tmp["rating"] >= self.threshold).astype(np.int8)
        pos = tmp[tmp["interaction"] == 1][["user_id", "movie_id"]].drop_duplicates()

        if pos.empty:
            raise RuntimeError("No positive interactions found (check threshold / data).")

        users = np.sort(pos["user_id"].unique())
        items = np.sort(pos["movie_id"].unique())

        self.user2idx = {int(u): int(i) for i, u in enumerate(users)}
        self.idx2user = {int(i): int(u) for i, u in enumerate(users)}
        self.item2idx = {int(m): int(i) for i, m in enumerate(items)}
        self.idx2item = {int(i): int(m) for i, m in enumerate(items)}

        pos_idx = pos.copy()
        pos_idx["u_idx"] = pos_idx["user_id"].map(self.user2idx).astype(np.int64)
        pos_idx["i_idx"] = pos_idx["movie_id"].map(self.item2idx).astype(np.int64)

        n_users = len(users)
        n_items = len(items)

        rows = pos_idx["u_idx"].to_numpy()
        cols = pos_idx["i_idx"].to_numpy()
        data = np.ones(len(pos_idx), dtype=np.int8)
        self.X_ui = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        self.model = NeuralMF(
            n_users=n_users,
            n_items=n_items,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        def _train_sync():
            assert self.model is not None
            assert self.X_ui is not None

            trained = train_neural_mf(
                model=self.model,
                train_pos=pos_idx[["u_idx", "i_idx"]],
                X_ui=self.X_ui,
                n_items=n_items,
                n_neg=self.n_neg,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
            )
            return trained

        self.model = await anyio.to_thread.run_sync(_train_sync)

    async def save(self) -> None:
        assert self.model is not None, "Pytorch model is not defined"

        await self.storage.save_csr_npz(self.x_ui_path, self.X_ui)
        await self.storage.save_json(self.mappings_path, {
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
            "threshold": self.threshold
        })

        self.model.eval()
        u = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        i = torch.tensor([10, 11, 12, 13], dtype=torch.long)

        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            local_onnx_path = tmp.name

        torch.onnx.export(
            self.model,
            (u, i),
            local_onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["user_id", "item_id"],
            output_names=["logit"],
            dynamic_axes={
                "user_id": {0: "batch"},
                "item_id": {0: "batch"},
                "logit":   {0: "batch"},
            },
        )
        await self.storage.save_onnx(self.model_path, local_onnx_path)

    async def recommend(self, user_id: int, n_records: int = 10) -> List[int]:
        if user_id not in self.user2idx:
            return []

        if self.session is None:
            raise ValueError("Session is not defined")

        uidx = int(self.user2idx[user_id])
        n_items = len(self.idx2item)

        bs = 20_000
        scores = np.empty(n_items, dtype=np.float32)
        item_idx_all = np.arange(n_items, dtype=np.int64)

        for start in range(0, n_items, bs):
            end = min(start + bs, n_items)
            item_batch = item_idx_all[start:end]
            user_batch = np.full((end - start,), uidx, dtype=np.int64)

            out = self.session.run(
                [self.output_name],
                {self.input_user: user_batch, self.input_item: item_batch},
            )[0]

            scores[start:end] = out.reshape(-1).astype(np.float32)

        seen_movie_ids = await self.ratings_repo.fetch_user_seen_movie_ids(
            user_id=user_id,
            min_rating=self.threshold,
        )
        seen_iidx = {self.item2idx[m] for m in seen_movie_ids if m in self.item2idx}

        if seen_iidx:
            scores[list(seen_iidx)] = -np.inf

        cand_k = min(n_items, max(n_records * 10, n_records))

        if cand_k >= n_items:
            item_idxs = np.argsort(-scores)
        else:
            item_idxs = np.argpartition(-scores, cand_k - 1)[:cand_k]
            item_idxs = item_idxs[np.argsort(-scores[item_idxs])]

        recs: List[int] = []
        for ii in item_idxs:
            ii = int(ii)

            if ii < 0 or ii >= n_items:
                raise RuntimeError(
                    f"NMF returned out-of-range item index {ii}, but n_items={n_items}. "
                    "This means the model and mappings are from different runs."
                )

            if ii in seen_iidx:
                continue

            recs.append(self.idx2item[ii])
            if len(recs) == n_records:
                break

        return recs