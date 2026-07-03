"""ConsRec -- Consensus-based group recommendation (Wu et al., WWW'23).
Reimplemented from https://github.com/FDUDSDE/WWW2023ConsRec (attribution retained).

Fuses three consensus views of a group: a group-group **overlap graph**, a
member-level **user/item hypergraph**, and a group-item **LightGCN** bipartite graph,
combined by learned gates; trained with BPR (group-item + user-item). ConsRec is
transductive (per-group-id embeddings), so it recommends for the *training* groups —
``recommend(members, ...)`` maps a member set back to its group index.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..data import Dataset, Groups
from .data import _reject_member_options, normalize_group_interactions


def _sp_to_tensor(x, device) -> torch.Tensor:
    coo = x.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(coo.data), torch.Size(coo.shape),
                                   device=device).coalesce()


def _build_overlap(members, G, device):
    M = np.eye(G, dtype=np.float64)
    sets = [set(m.tolist()) for m in members]
    for i in range(G):
        for j in range(i + 1, G):
            u = len(sets[i] | sets[j])
            if u:
                M[i, j] = M[j, i] = len(sets[i] & sets[j]) / u
    deg = M.sum(1)
    return torch.tensor(np.diag(1.0 / deg) @ M, dtype=torch.float32, device=device)


def _incidence(group_dict, n_nodes, G, axis, device):
    nodes, groups = [], []
    for g in range(G):
        nodes.extend(group_dict.get(g, []))
        groups.extend([g] * len(group_dict.get(g, [])))
    H = sp.csr_matrix((np.ones(len(nodes)), (nodes, groups)), shape=(n_nodes, G))
    deg = np.array(H.sum(axis=axis)).squeeze()
    deg[deg == 0.0] = 1.0
    return _sp_to_tensor(H, device), _sp_to_tensor(sp.diags(1.0 / deg), device)


def _build_hypergraph(members_idx, group_items_idx, U, I, G, device):
    member_dict = {g: list(members_idx[g]) for g in range(G)}
    item_dict = {g: list(group_items_idx[g]) for g in range(G)}
    user_hg, user_deg = _incidence(member_dict, U, G, axis=0, device=device)
    item_hg, item_deg = _incidence(item_dict, I, G, axis=0, device=device)
    full_dict = {g: list(members_idx[g]) + [it + U for it in group_items_idx[g]] for g in range(G)}
    full_hg, node_deg = _incidence(full_dict, U + I, G, axis=1, device=device)
    user_hyper = torch.sparse.mm(user_deg, user_hg.t()).coalesce()
    item_hyper = torch.sparse.mm(item_deg, item_hg.t()).coalesce()
    full_hyper = torch.sparse.mm(node_deg, full_hg).coalesce()
    return user_hyper, item_hyper, full_hyper


def _build_lightgcn(members_group_items, G, device):
    lgcn_items = sorted({it for items in members_group_items for it in items})
    col = {it: c for c, it in enumerate(lgcn_items)}
    n = len(lgcn_items)
    R = sp.dok_matrix((G, n), dtype=np.float32)
    for g, items in enumerate(members_group_items):
        for it in items:
            R[g, col[it]] = 1.0
    adj = sp.dok_matrix((G + n, G + n), dtype=np.float32).tolil()
    R = R.tolil()
    adj[:G, G:] = R
    adj[G:, :G] = R.T
    adj = adj.todok()
    d = np.power(np.array(adj.sum(1)).flatten(), -0.5)
    d[np.isinf(d)] = 0.0
    D = sp.diags(d)
    norm = D.dot(adj).dot(D).tocsr()
    return _sp_to_tensor(norm, device), np.array(lgcn_items, dtype=np.int64)


class _PredictLayer(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        # LeakyReLU (not ReLU): the element-wise g*i product is often all-negative, which
        # kills a ReLU head ("dead ReLU" — see the ConsRec/AlignGroup code comments).
        self.net = nn.Sequential(nn.Linear(d, 8), nn.LeakyReLU(), nn.Dropout(drop), nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x)


class _ConsRecNet(nn.Module):
    def __init__(self, U, I, G, d, layers, overlap, user_hyper, item_hyper, full_hyper,
                 lgcn_graph, lgcn_items, predictor="DOT"):
        super().__init__()
        self.U, self.I, self.G, self.layers = U, I, G, layers
        self.predictor = predictor
        self.predict = _PredictLayer(d) if predictor == "MLP" else None
        self.user_emb = nn.Embedding(U, d)
        self.item_emb = nn.Embedding(I, d)
        self.group_emb = nn.Embedding(G, d)
        for e in (self.user_emb, self.item_emb, self.group_emb):
            nn.init.xavier_uniform_(e.weight)
        self.overlap = overlap
        self.user_hyper, self.item_hyper, self.full_hyper = user_hyper, item_hyper, full_hyper
        self.lgcn_graph = lgcn_graph
        self.register_buffer("lgcn_items", torch.as_tensor(lgcn_items))
        self.hyper_agg = nn.ModuleList([nn.Linear(3 * d, d) for _ in range(layers)])
        self.overlap_gate = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.hyper_gate = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.lgcn_gate = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())

    def _overlap_conv(self):
        emb = self.group_emb.weight
        out = [emb]
        for _ in range(self.layers):
            emb = torch.mm(self.overlap, emb)
            out.append(emb)
        return torch.sum(torch.stack(out), dim=0)

    def _hyper_conv(self, group_emb):
        u, i = self.user_emb.weight, self.item_emb.weight
        node_final = [torch.cat([u, i], 0)]
        he_final = [group_emb]
        for lin in self.hyper_agg:
            um = torch.sparse.mm(self.user_hyper, u)
            im = torch.sparse.mm(self.item_hyper, i)
            msg = lin(torch.cat([um, im, im * group_emb], dim=1))
            nodes = torch.mm(self.full_hyper, msg)
            u, i = torch.split(nodes, [self.U, self.I])
            node_final.append(nodes)
            he_final.append(msg)
        return torch.sum(torch.stack(node_final), 0), torch.sum(torch.stack(he_final), 0)

    def _lightgcn(self):
        emb = torch.cat([self.group_emb.weight, self.item_emb.weight[self.lgcn_items]])
        out = [emb]
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.lgcn_graph, emb)
            out.append(emb)
        emb = torch.mean(torch.stack(out, 1), 1)
        return torch.split(emb, [self.G, self.lgcn_items.shape[0]])[0]

    def fused(self):
        g_overlap = self._overlap_conv()
        nodes, he = self._hyper_conv(g_overlap)
        i_emb = torch.split(nodes, [self.U, self.I])[1]
        g_lgcn = self._lightgcn()
        oc, hc, lc = self.overlap_gate(g_overlap), self.hyper_gate(he), self.lgcn_gate(g_lgcn)
        group = oc * g_overlap + hc * he + lc * g_lgcn
        return group, i_emb

    def _score(self, a, b):
        if self.predictor == "MLP":
            return torch.sigmoid(self.predict(a * b)).squeeze(-1)
        return (a * b).sum(-1)

    def group_pair(self, g_idx, it_idx, fused=None):
        group, i_emb = fused if fused is not None else self.fused()
        return self._score(group[g_idx], i_emb[it_idx])

    def user_pair(self, u_idx, it_idx):
        return self._score(self.user_emb(u_idx), self.item_emb(it_idx))


class ConsRec:
    """Consensus-based group recommender (``paradigm="profile"``, transductive)."""

    paradigm = "profile"
    supports_member_weights = False   # transductive (group-id node); no member pooling to steer

    def __init__(self, groups: Groups, group_interactions, *, emb_dim: int = 32,
                 layers: int = 3, epochs: int = 100, lr: float = 0.001,
                 num_negatives: int = 8, predictor: str = "MLP", batch_size: int = 512,
                 weight_decay: float = 1e-5, user_item: bool = True, seed: int | None = 0,
                 device: str = "cpu") -> None:
        self.groups = groups
        self._raw_gi = group_interactions
        self.emb_dim = emb_dim
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.num_negatives = num_negatives
        self.predictor = predictor
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.user_item = user_item
        self.seed = seed
        self.device = device
        self.dataset_: Dataset | None = None
        self.net_: _ConsRecNet | None = None
        self._fused_cache = None

    def fit(self, dataset: Dataset) -> "ConsRec":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.dataset_ = dataset
        U, I, G = dataset.n_users, dataset.n_items, len(self.groups)
        ui, ii = dataset.user_index, dataset.item_index
        members = [np.array([ui[u] for u in m if u in ui], dtype=np.int64) for m in self.groups]
        gi = normalize_group_interactions(self._raw_gi, G)
        group_items = [[ii[it] for it in gi.get(g, []) if it in ii] for g in range(G)]
        # member-set -> group index, for recommend() lookup (transductive)
        self._lookup = {tuple(sorted(m.tolist())): g for g, m in enumerate(members)}

        overlap = _build_overlap(members, G, self.device)
        user_hyper, item_hyper, full_hyper = _build_hypergraph(members, group_items, U, I, G, self.device)
        lgcn_graph, lgcn_items = _build_lightgcn(group_items, G, self.device)

        self.net_ = _ConsRecNet(U, I, G, self.emb_dim, self.layers, overlap, user_hyper,
                                item_hyper, full_hyper, lgcn_graph, lgcn_items,
                                predictor=self.predictor).to(self.device)

        # training pairs
        gp = [(g, it) for g in range(G) for it in group_items[g]]
        self._g_pos = np.array(gp, dtype=np.int64) if gp else np.zeros((0, 2), np.int64)
        self._u_pos = np.vstack([dataset.interactions["user"].map(ui).to_numpy(),
                                 dataset.interactions["item"].map(ii).to_numpy()]).T
        self._fused_cache = None
        self._train()
        return self

    def _expand(self, pairs, rng):
        """num_negatives expansion -> shuffled (entity, pos, neg) rows, like the
        original get_train_instances + DataLoader (one negative per row)."""
        ent = np.repeat(pairs[:, 0], self.num_negatives)
        pos = np.repeat(pairs[:, 1], self.num_negatives)
        neg = rng.integers(0, self.dataset_.n_items, size=ent.shape[0])
        perm = rng.permutation(ent.shape[0])
        return ent[perm], pos[perm], neg[perm]

    def _train(self):
        net = self.net_
        opt = torch.optim.RMSprop(net.parameters(), lr=self.lr)  # matches the original ConsRec
        rng = np.random.default_rng(self.seed)
        bs = self.batch_size
        net.train()
        for _ in range(self.epochs):
            # group-item BPR: row-minibatched (graph-conv per batch)
            if self._g_pos.shape[0]:
                ent, pos, neg = self._expand(self._g_pos, rng)
                for s in range(0, ent.shape[0], bs):
                    fused = net.fused()
                    g = torch.as_tensor(ent[s: s + bs], device=self.device)
                    pt = torch.as_tensor(pos[s: s + bs], device=self.device)
                    nt = torch.as_tensor(neg[s: s + bs], device=self.device)
                    loss = torch.nn.functional.softplus(net.group_pair(g, nt, fused)
                                                        - net.group_pair(g, pt, fused)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            # user-item BPR: row-minibatched (no graph conv; trains shared user/item embs)
            if self.user_item and self._u_pos.shape[0]:
                ent, pos, neg = self._expand(self._u_pos, rng)
                for s in range(0, ent.shape[0], bs):
                    u = torch.as_tensor(ent[s: s + bs], device=self.device)
                    pt = torch.as_tensor(pos[s: s + bs], device=self.device)
                    nt = torch.as_tensor(neg[s: s + bs], device=self.device)
                    loss = torch.nn.functional.softplus(net.user_pair(u, nt) - net.user_pair(u, pt)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()

    def group_scores(self, members, items=None, *, member_weights=None,
                     return_attention=False) -> np.ndarray:
        """Per-item group scores for a member set (the trained head over the fused
        group embedding). ConsRec is transductive, so ``member_weights`` /
        ``return_attention`` are not defined and raise."""
        _reject_member_options(self, member_weights, return_attention)
        if self.net_ is None:
            raise RuntimeError("ConsRec must be fit() before scoring.")
        ui = self.dataset_.user_index
        gi = self._lookup.get(tuple(sorted(ui[u] for u in members if u in ui)))
        self.net_.eval()
        with torch.no_grad():
            if self._fused_cache is None:       # embeddings are static after fit -> cache once
                self._fused_cache = self.net_.fused()
            group, i_emb = self._fused_cache
            if gi is None:                      # ephemeral group: fall back to member mean
                midx = [ui[u] for u in members if u in ui]
                g_vec = self.net_.user_emb.weight[midx].mean(0) if midx else group.mean(0)
            else:
                g_vec = group[gi]
            # score with the trained head (MLP or dot) -- a raw dot product would ignore
            # the learned nonlinear predictor and break the ranking
            g_mat = g_vec.unsqueeze(0).expand(i_emb.shape[0], -1)
            scores = self.net_._score(g_mat, i_emb).cpu().numpy()
        if items is not None:
            scores = scores[np.array([self.dataset_.item_index[i] for i in items], dtype=np.int64)]
        return scores

    def recommend(self, members, k: int, *, exclude=None, candidates=None,
                  member_weights=None) -> np.ndarray:
        scores = self.group_scores(members, member_weights=member_weights)
        if candidates is not None:
            cand = list(candidates)
            cidx = np.array([self.dataset_.item_index[c] for c in cand], dtype=np.int64)
            return np.asarray(cand)[np.argsort(-scores[cidx], kind="stable")[:k]]
        if exclude:
            ex = [self.dataset_.item_index[i] for i in exclude if i in self.dataset_.item_index]
            scores[ex] = -np.inf
        budget = int(min(k, np.isfinite(scores).sum()))
        return self.dataset_.items[np.argsort(-scores, kind="stable")[:budget]]


__all__ = ["ConsRec"]
