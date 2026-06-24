"""Base class + network for deep group recommenders (profile-aggregation paradigm).

These reimplement the canonical AGREE-family mechanism: per-user/item embeddings, a
group representation pooled from member embeddings (mean for NCF, item-conditioned
attention for AGREE), and an NCF-style predict layer on ``[g⊙i, g, i]``. Trained
jointly on user-item and group-item interactions with BPR.

Design choice: the group representation is computed **from member embeddings** (not a
per-group-id embedding), so models recommend for arbitrary member sets and fit the
``recommend(members, k)`` contract — the same interface aggregators use, which is what
lets deep models share a leaderboard with them (the bridge). Attribution: the
attention + joint-training design follows Cao et al., AGREE (SIGIR'18).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data import Dataset, Groups
from .data import normalize_group_interactions


class _Net(nn.Module):
    def __init__(self, n_users: int, n_items: int, d: int, attention: bool) -> None:
        super().__init__()
        self.attention = attention
        self.user_emb = nn.Embedding(n_users + 1, d, padding_idx=n_users)  # last = cold/pad
        self.item_emb = nn.Embedding(n_items, d)
        if attention:
            self.att = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.predict = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        with torch.no_grad():
            self.user_emb.weight[n_users].zero_()

    def _score(self, g, item_vecs):
        x = torch.cat([g * item_vecs, g, item_vecs], dim=-1)
        return self.predict(x).squeeze(-1)

    def pair_scores(self, user_idx, item_idx):
        """User-item scores for aligned (B,) index tensors."""
        return self._score(self.user_emb(user_idx), self.item_emb(item_idx))

    def group_rep(self, member_idx, item_vecs):
        """Group representation per candidate item. member_idx: (M,), item_vecs: (N,d)."""
        memb = self.user_emb(member_idx)                      # (M, d)
        if not self.attention:
            return memb.mean(dim=0, keepdim=True).expand(item_vecs.size(0), -1)
        M, d = memb.shape
        N = item_vecs.size(0)
        mm = memb.unsqueeze(1).expand(M, N, d)               # (M, N, d)
        ii = item_vecs.unsqueeze(0).expand(M, N, d)
        a = self.att(torch.cat([mm, mm * ii], dim=-1)).squeeze(-1)  # (M, N)
        a = torch.softmax(a, dim=0)
        return (a.unsqueeze(-1) * mm).sum(dim=0)             # (N, d)

    def group_item_scores(self, member_idx, item_idx):
        """Scores of ``item_idx`` (N,) for one group given member indices (M,)."""
        item_vecs = self.item_emb(item_idx)
        g = self.group_rep(member_idx, item_vecs)
        return self._score(g, item_vecs)


class GroupNNModel:
    """Trainable deep group recommender. Subclasses set ``attention``."""

    paradigm = "profile"
    attention = False

    def __init__(
        self,
        groups: Groups,
        group_interactions,
        *,
        factors: int = 32,
        epochs: int = 20,
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        neg_samples: int = 4,
        user_item: bool = True,
        batch_size: int = 1024,
        seed: int | None = 0,
        device: str = "cpu",
    ) -> None:
        self.groups = groups
        self._raw_group_interactions = group_interactions
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_samples = neg_samples
        self.user_item = user_item
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.dataset_: Dataset | None = None
        self.net_: _Net | None = None

    # -- fit ---------------------------------------------------------------- #
    def fit(self, dataset: Dataset) -> "GroupNNModel":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.dataset_ = dataset
        n_users, n_items = dataset.n_users, dataset.n_items
        ui = dataset.user_index
        ii = dataset.item_index

        gi = normalize_group_interactions(self._raw_group_interactions, len(self.groups))
        self._members = [
            np.array([ui[u] for u in members if u in ui], dtype=np.int64)
            for members in self.groups
        ]
        self._group_pos = [
            np.array([ii[it] for it in gi.get(idx, []) if it in ii], dtype=np.int64)
            for idx in range(len(self.groups))
        ]
        u_pos = dataset.interactions["user"].map(ui).to_numpy()
        i_pos = dataset.interactions["item"].map(ii).to_numpy()

        self.net_ = _Net(n_users, n_items, self.factors, self.attention).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        self._train(opt, rng, u_pos, i_pos, n_items)
        return self

    def _train(self, opt, rng, u_pos, i_pos, n_items) -> None:
        net = self.net_
        net.train()
        for _ in range(self.epochs):
            # group-item BPR (per group, items vary in count)
            for members, pos in zip(self._members, self._group_pos):
                if members.size == 0 or pos.size == 0:
                    continue
                neg = rng.integers(0, n_items, size=pos.size * self.neg_samples)
                m = torch.as_tensor(members, device=self.device)
                items = torch.as_tensor(np.concatenate([pos, neg]), device=self.device)
                scores = net.group_item_scores(m, items)
                ps = scores[: pos.size].repeat(self.neg_samples)
                ns = scores[pos.size:]
                loss = -torch.log(torch.sigmoid(ps - ns) + 1e-9).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            # user-item BPR (batched)
            if self.user_item and u_pos.size:
                order = rng.permutation(u_pos.size)
                for s in range(0, u_pos.size, self.batch_size):
                    b = order[s: s + self.batch_size]
                    users = torch.as_tensor(u_pos[b], device=self.device)
                    pos = torch.as_tensor(i_pos[b], device=self.device)
                    neg = torch.as_tensor(rng.integers(0, n_items, size=b.size),
                                          device=self.device)
                    ps = net.pair_scores(users, pos)
                    ns = net.pair_scores(users, neg)
                    loss = -torch.log(torch.sigmoid(ps - ns) + 1e-9).mean()
                    opt.zero_grad(); loss.backward(); opt.step()

    # -- recommend ---------------------------------------------------------- #
    def recommend(self, members, k: int, *, exclude=None, candidates=None) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError(f"{type(self).__name__} must be fit() before recommending.")
        ui = self.dataset_.user_index
        midx = np.array([ui[u] for u in members if u in ui], dtype=np.int64)
        if midx.size == 0:
            midx = np.array([self.dataset_.n_users], dtype=np.int64)  # cold -> pad row
        self.net_.eval()

        if candidates is not None:
            cand = list(candidates)
            cidx = np.array([self.dataset_.item_index[c] for c in cand], dtype=np.int64)
            with torch.no_grad():
                m = torch.as_tensor(midx, device=self.device)
                scores = self.net_.group_item_scores(
                    m, torch.as_tensor(cidx, device=self.device)).cpu().numpy()
            order = np.argsort(-scores, kind="stable")[:k]
            return np.asarray(cand)[order]

        n_items = self.dataset_.n_items
        with torch.no_grad():
            m = torch.as_tensor(midx, device=self.device)
            all_items = torch.arange(n_items, device=self.device)
            scores = self.net_.group_item_scores(m, all_items).cpu().numpy()
        if exclude:
            ex = [self.dataset_.item_index[i] for i in exclude if i in self.dataset_.item_index]
            scores[ex] = -np.inf
        budget = int(min(k, np.isfinite(scores).sum()))
        top = np.argsort(-scores, kind="stable")[:budget]
        return self.dataset_.items[top]
