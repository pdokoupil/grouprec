"""AlignGroup -- group recommendation with member/group alignment (Xu et al.).
Reimplemented from https://github.com/Jinfeng-Xu/AlignGroup (attribution retained).

Overlap group-graph + a member-level hypergraph (2-input HGNN), with an **InfoNCE
alignment** between each group's hyperedge embedding and the *geometric center*
``(max+min)/2`` of its (refined) member embeddings. Loss = BPR (softplus) +
``cl_weight * InfoNCE``. Transductive (group-id embeddings); ``recommend`` maps a
member set to its group index.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Dataset, Groups
from .consrec import _PredictLayer, _build_hypergraph, _build_overlap
from .data import normalize_group_interactions


class _AlignNet(nn.Module):
    def __init__(self, U, I, G, d, layers, overlap, user_hyper, item_hyper, full_hyper, predictor):
        super().__init__()
        self.U, self.I, self.layers, self.predictor = U, I, layers, predictor
        self.user_emb = nn.Embedding(U, d)
        self.item_emb = nn.Embedding(I, d)
        self.group_emb = nn.Embedding(G, d)
        for e in (self.user_emb, self.item_emb, self.group_emb):
            nn.init.xavier_uniform_(e.weight)
        self.overlap = overlap
        self.user_hyper, self.item_hyper, self.full_hyper = user_hyper, item_hyper, full_hyper
        self.agg = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.predict = _PredictLayer(d) if predictor == "MLP" else None

    def fused(self):
        group0 = torch.mm(self.overlap, self.group_emb.weight)     # overlap conv (single)
        u, i, he = self.user_emb.weight, self.item_emb.weight, group0
        ui_acc, g_acc = [torch.cat([u, i], 0)], [he]
        for lin in self.agg:
            um = torch.sparse.mm(self.user_hyper, u)
            im = torch.sparse.mm(self.item_hyper, i)
            msg = lin(torch.cat([um, im], dim=1))
            nodes = torch.mm(self.full_hyper, msg)
            u, i = torch.split(nodes, [self.U, self.I])
            ui_acc.append(nodes); g_acc.append(msg)
        ui = torch.sum(torch.stack(ui_acc), 0)
        g = torch.sum(torch.stack(g_acc), 0)
        u_emb, i_emb = torch.split(ui, [self.U, self.I])
        return u_emb, i_emb, g

    def _score(self, a, b):
        if self.predictor == "MLP":
            return torch.sigmoid(self.predict(a * b)).squeeze(-1)
        return (a * b).sum(-1)

    def group_pair(self, g_idx, it_idx, fused):
        _, i_emb, g_emb = fused
        return self._score(g_emb[g_idx], i_emb[it_idx])

    def user_pair(self, u_idx, it_idx):
        return self._score(self.user_emb(u_idx), self.item_emb(it_idx))

    @staticmethod
    def infonce(view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos = torch.exp((view1 * view2).sum(-1) / temp)
        ttl = torch.exp(view1 @ view2.t() / temp).sum(1)
        return -torch.log(pos / ttl).mean()


class AlignGroup:
    """Alignment-based group recommender (``paradigm="profile"``, transductive)."""

    paradigm = "profile"

    def __init__(self, groups: Groups, group_interactions, *, emb_dim: int = 32,
                 layers: int = 4, epochs: int = 100, lr: float = 0.001,
                 num_negatives: int = 8, predictor: str = "MLP", batch_size: int = 512,
                 cl_weight: float = 0.1, temp: float = 0.2, weight_decay: float = 1e-5,
                 user_item: bool = True, seed: int | None = 0, device: str = "cpu") -> None:
        self.groups = groups
        self._raw_gi = group_interactions
        self.emb_dim, self.layers, self.epochs, self.lr = emb_dim, layers, epochs, lr
        self.num_negatives, self.predictor, self.batch_size = num_negatives, predictor, batch_size
        self.cl_weight, self.temp = cl_weight, temp
        self.weight_decay, self.user_item, self.seed, self.device = weight_decay, user_item, seed, device
        self.dataset_, self.net_ = None, None
        self._fused_cache = None

    def fit(self, dataset: Dataset) -> "AlignGroup":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.dataset_ = dataset
        U, I, G = dataset.n_users, dataset.n_items, len(self.groups)
        ui, ii = dataset.user_index, dataset.item_index
        members = [np.array([ui[u] for u in m if u in ui], dtype=np.int64) for m in self.groups]
        gi = normalize_group_interactions(self._raw_gi, G)
        group_items = [[ii[it] for it in gi.get(g, []) if it in ii] for g in range(G)]
        self._lookup = {tuple(sorted(m.tolist())): g for g, m in enumerate(members)}
        self._members_t = [torch.as_tensor(m, device=self.device) for m in members]

        overlap = _build_overlap(members, G, self.device)
        uh, ih, fh = _build_hypergraph(members, group_items, U, I, G, self.device)
        self.net_ = _AlignNet(U, I, G, self.emb_dim, self.layers, overlap, uh, ih, fh, self.predictor).to(self.device)
        gp = [(g, it) for g in range(G) for it in group_items[g]]
        self._g_pos = np.array(gp, dtype=np.int64) if gp else np.zeros((0, 2), np.int64)
        self._u_pos = np.vstack([dataset.interactions["user"].map(ui).to_numpy(),
                                 dataset.interactions["item"].map(ii).to_numpy()]).T
        self._fused_cache = None
        self._train()
        return self

    def _centers(self, group_ids, u_emb):
        rows = []
        for g in group_ids:
            m = self._members_t[g]
            emb = u_emb[m] if m.numel() else u_emb[:1]
            rows.append(((emb.max(0).values + emb.min(0).values) / 2))
        return torch.stack(rows)

    def _expand(self, pairs, rng):
        """num_negatives expansion -> shuffled (entity, pos, neg) rows, like the
        original get_train_instances + DataLoader (one negative per row)."""
        ent = np.repeat(pairs[:, 0], self.num_negatives)
        pos = np.repeat(pairs[:, 1], self.num_negatives)
        neg = rng.integers(0, self.dataset_.n_items, size=ent.shape[0])
        perm = rng.permutation(ent.shape[0])
        return ent[perm], pos[perm], neg[perm]

    def _train(self):
        self._opt = torch.optim.RMSprop(self.net_.parameters(), lr=self.lr)  # matches the original
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            self._train_epoch(rng)

    def _train_epoch(self, rng):
        net, opt, bs = self.net_, self._opt, self.batch_size
        net.train()
        # group BPR + InfoNCE alignment: row-minibatched (graph conv per batch)
        if self._g_pos.shape[0]:
            ent, pos, neg = self._expand(self._g_pos, rng)
            for s in range(0, ent.shape[0], bs):
                gids = ent[s: s + bs]
                fused = net.fused()
                u_emb, _, g_emb = fused
                gt = torch.as_tensor(gids, device=self.device)
                pt = torch.as_tensor(pos[s: s + bs], device=self.device)
                nt = torch.as_tensor(neg[s: s + bs], device=self.device)
                bpr = F.softplus(net.group_pair(gt, nt, fused) - net.group_pair(gt, pt, fused)).mean()
                cl = net.infonce(self._centers(gids, u_emb), g_emb[gt], self.temp)
                loss = bpr + self.cl_weight * cl
                opt.zero_grad(); loss.backward(); opt.step()
        # user-item BPR: row-minibatched (no graph conv; trains shared user/item embs)
        if self.user_item and self._u_pos.shape[0]:
            ent, pos, neg = self._expand(self._u_pos, rng)
            for s in range(0, ent.shape[0], bs):
                ut = torch.as_tensor(ent[s: s + bs], device=self.device)
                pt = torch.as_tensor(pos[s: s + bs], device=self.device)
                nt = torch.as_tensor(neg[s: s + bs], device=self.device)
                loss = F.softplus(net.user_pair(ut, nt) - net.user_pair(ut, pt)).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def recommend(self, members, k: int, *, exclude=None, candidates=None) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("AlignGroup must be fit() before recommending.")
        ui = self.dataset_.user_index
        key = tuple(sorted(ui[u] for u in members if u in ui))
        gi = self._lookup.get(key)
        self.net_.eval()
        with torch.no_grad():
            if self._fused_cache is None:       # embeddings are static after fit -> cache once
                self._fused_cache = self.net_.fused()
            u_emb, i_emb, g_emb = self._fused_cache
            g_vec = g_emb[gi] if gi is not None else (
                u_emb[[ui[u] for u in members if u in ui]].mean(0) if any(u in ui for u in members)
                else g_emb.mean(0))
            # score with the *same* head used in training (MLP predictor or dot) -- a raw
            # dot product would ignore the learned nonlinear head and break the ranking
            g_mat = g_vec.unsqueeze(0).expand(i_emb.shape[0], -1)
            scores = self.net_._score(g_mat, i_emb).cpu().numpy()
        if candidates is not None:
            cand = list(candidates)
            cidx = np.array([self.dataset_.item_index[c] for c in cand], dtype=np.int64)
            return np.asarray(cand)[np.argsort(-scores[cidx], kind="stable")[:k]]
        if exclude:
            ex = [self.dataset_.item_index[i] for i in exclude if i in self.dataset_.item_index]
            scores[ex] = -np.inf
        budget = int(min(k, np.isfinite(scores).sum()))
        return self.dataset_.items[np.argsort(-scores, kind="stable")[:budget]]


__all__ = ["AlignGroup"]
