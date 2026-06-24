"""HyperGroup -- a hypergraph-only group recommender baseline.

A member-level hypergraph (users + items as nodes, each **group a hyperedge**) with an
HGNN convolution producing group (hyperedge) representations, scored against item
embeddings by dot product; trained with BPR (group-item + user-item). This is the
hypergraph view used by hypergraph GRS models (HyperGroup/HHGR/S2-HHGR family) in
isolation, reusing the graph construction from :mod:`grouprec.models.consrec`.
Transductive (per-group hyperedge), so ``recommend`` maps a member set to its group.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data import Dataset, Groups
from .consrec import _build_hypergraph
from .data import normalize_group_interactions


class _HyperNet(nn.Module):
    def __init__(self, U, I, G, d, layers, user_hyper, item_hyper, full_hyper):
        super().__init__()
        self.U, self.I, self.layers = U, I, layers
        self.user_emb = nn.Embedding(U, d)
        self.item_emb = nn.Embedding(I, d)
        self.group_emb = nn.Embedding(G, d)
        for e in (self.user_emb, self.item_emb, self.group_emb):
            nn.init.xavier_uniform_(e.weight)
        self.user_hyper, self.item_hyper, self.full_hyper = user_hyper, item_hyper, full_hyper
        self.agg = nn.ModuleList([nn.Linear(3 * d, d) for _ in range(layers)])

    def fused(self):
        u, i, he = self.user_emb.weight, self.item_emb.weight, self.group_emb.weight
        nodes_acc = [torch.cat([u, i], 0)]
        he_acc = [he]
        for lin in self.agg:
            um = torch.sparse.mm(self.user_hyper, u)
            im = torch.sparse.mm(self.item_hyper, i)
            msg = lin(torch.cat([um, im, im * he], dim=1))
            nodes = torch.mm(self.full_hyper, msg)
            u, i = torch.split(nodes, [self.U, self.I])
            nodes_acc.append(nodes)
            he_acc.append(msg)
        nodes = torch.sum(torch.stack(nodes_acc), 0)
        he = torch.sum(torch.stack(he_acc), 0)
        i_emb = torch.split(nodes, [self.U, self.I])[1]
        return he, i_emb

    def group_pair(self, g_idx, it_idx, fused=None):
        he, i_emb = fused if fused is not None else self.fused()
        return (he[g_idx] * i_emb[it_idx]).sum(-1)

    def user_pair(self, u_idx, it_idx):
        return (self.user_emb(u_idx) * self.item_emb(it_idx)).sum(-1)


class HyperGroup:
    """Hypergraph-only group recommender (``paradigm="profile"``, transductive)."""

    paradigm = "profile"

    def __init__(self, groups: Groups, group_interactions, *, emb_dim: int = 32,
                 layers: int = 2, epochs: int = 30, lr: float = 0.01,
                 weight_decay: float = 1e-5, user_item: bool = True, seed: int | None = 0,
                 device: str = "cpu") -> None:
        self.groups = groups
        self._raw_gi = group_interactions
        self.emb_dim, self.layers, self.epochs = emb_dim, layers, epochs
        self.lr, self.weight_decay, self.user_item = lr, weight_decay, user_item
        self.seed, self.device = seed, device
        self.dataset_ = None
        self.net_ = None

    def fit(self, dataset: Dataset) -> "HyperGroup":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.dataset_ = dataset
        U, I, G = dataset.n_users, dataset.n_items, len(self.groups)
        ui, ii = dataset.user_index, dataset.item_index
        members = [np.array([ui[u] for u in m if u in ui], dtype=np.int64) for m in self.groups]
        gi = normalize_group_interactions(self._raw_gi, G)
        group_items = [[ii[it] for it in gi.get(g, []) if it in ii] for g in range(G)]
        self._lookup = {tuple(sorted(m.tolist())): g for g, m in enumerate(members)}

        user_hyper, item_hyper, full_hyper = _build_hypergraph(members, group_items, U, I, G, self.device)
        self.net_ = _HyperNet(U, I, G, self.emb_dim, self.layers,
                              user_hyper, item_hyper, full_hyper).to(self.device)
        gp = [(g, it) for g in range(G) for it in group_items[g]]
        self._g_pos = np.array(gp, dtype=np.int64) if gp else np.zeros((0, 2), np.int64)
        self._u_pos = np.vstack([dataset.interactions["user"].map(ui).to_numpy(),
                                 dataset.interactions["item"].map(ii).to_numpy()]).T
        self._train()
        return self

    def _train(self):
        net, I = self.net_, self.dataset_.n_items
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        net.train()
        for _ in range(self.epochs):
            if self._g_pos.shape[0]:
                fused = net.fused()
                g = torch.as_tensor(self._g_pos[:, 0], device=self.device)
                pos = torch.as_tensor(self._g_pos[:, 1], device=self.device)
                neg = torch.as_tensor(rng.integers(0, I, self._g_pos.shape[0]), device=self.device)
                loss = -torch.log(torch.sigmoid(net.group_pair(g, pos, fused) - net.group_pair(g, neg, fused)) + 1e-9).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if self.user_item and self._u_pos.shape[0]:
                u = torch.as_tensor(self._u_pos[:, 0], device=self.device)
                pos = torch.as_tensor(self._u_pos[:, 1], device=self.device)
                neg = torch.as_tensor(rng.integers(0, I, self._u_pos.shape[0]), device=self.device)
                loss = -torch.log(torch.sigmoid(net.user_pair(u, pos) - net.user_pair(u, neg)) + 1e-9).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def recommend(self, members, k: int, *, exclude=None, candidates=None) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("HyperGroup must be fit() before recommending.")
        ui = self.dataset_.user_index
        key = tuple(sorted(ui[u] for u in members if u in ui))
        gi = self._lookup.get(key)
        self.net_.eval()
        with torch.no_grad():
            he, i_emb = self.net_.fused()
            g_vec = he[gi] if gi is not None else (
                self.net_.user_emb.weight[[ui[u] for u in members if u in ui]].mean(0)
                if any(u in ui for u in members) else he.mean(0))
            scores = (g_vec.unsqueeze(0) * i_emb).sum(-1).cpu().numpy()
        if candidates is not None:
            cand = list(candidates)
            cidx = np.array([self.dataset_.item_index[c] for c in cand], dtype=np.int64)
            return np.asarray(cand)[np.argsort(-scores[cidx], kind="stable")[:k]]
        if exclude:
            ex = [self.dataset_.item_index[i] for i in exclude if i in self.dataset_.item_index]
            scores[ex] = -np.inf
        budget = int(min(k, np.isfinite(scores).sum()))
        return self.dataset_.items[np.argsort(-scores, kind="stable")[:budget]]


__all__ = ["HyperGroup"]
