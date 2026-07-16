"""HHGR -- Double-Scale Self-supervised Hypergraph Learning for Group Recommendation
(Zhang et al., CIKM'21; arXiv:2109.04200). Reimplemented against the paper and the
reference refactor in ``WWW2023GroupRecBaselines/HHGR`` (official code:
https://github.com/0411tony/HHGR).

The "double scale" is two *granularities* of the user--group hypergraph, each encoded by
its own HGNN: a **coarse** view that keeps a random subset of user nodes, and a **fine**
view that keeps, per user, a random subset of the hyperedges (groups) they belong to. A
bilinear discriminator then maximises the mutual information between a user's coarse and
fine views against another user's view -- the self-supervised signal. A group is
represented by self-attentive pooling over its members plus a group-id embedding
propagated over a group--group hypergraph. Training runs in three stages: user--item
pre-training, user-level self-supervision, then group-level learning.

Transductive (per-group-id embeddings), so ``recommend`` maps a member set back to its
group index, and ``supports_member_weights`` is ``False``.

The reference refactor is a work in progress, and this implementation departs from it in
the following places. They are recorded because each one changes the results:

* it casts the hypergraph convolution's activations with ``x.long()``, truncating them to
  integers; kept float here;
* it holds the convolutions in a plain Python list, which leaves them unregistered as
  submodules and so never trained; ``nn.ModuleList`` here;
* its fine-grained ``beta`` mask is allocated outside the per-user loop, so the hyperedge
  dropout accumulates across users rather than being resampled; resampled per user here;
* it materialises the propagation matrix densely; kept sparse here, since it is nonzero
  only for users that share a group;
* it feeds the group--group matrix into the group-level HGNN unnormalised, while the
  user-level operator is normalised to row-sum 1. Where groups overlap heavily the group
  term then explodes (on Mafengwo those row sums reach ~5.6e4) and buries the member signal
  under noise, leaving the ranking at chance. CAMRa2011 does not show this: its two-member
  households form no triangles, so the matrix is empty and the term is inert. Row-normalised
  here, making each message a weighted average of the neighbouring groups;
* it freezes the user representation for the group stage while the network is in training
  mode, which bakes in a dropout mask that inference never reproduces; the clean
  representation is frozen here.

Validated against the numbers reported for S2-HHGR by ConsRec (WWW'23, Table 2) under the
same 1-vs-100 sampled protocol: CAMRa2011 HR@5 0.626 / NDCG@5 0.400 (0.606 / 0.385
reported), Mafengwo HR@5 0.852 / NDCG@5 0.752 with ``group_epochs=100, lr_group=1e-3``
(0.757 / 0.732 reported).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Dataset, Groups
from .consrec import _sp_to_tensor
from .data import _reject_member_options, normalize_group_interactions


def _user_group_incidence(members_idx, n_users: int, n_groups: int) -> sp.csr_matrix:
    """Users x groups incidence: 1 where a user is a member of a group."""
    rows, cols = [], []
    for g, members in enumerate(members_idx):
        rows.extend(int(u) for u in members)
        cols.extend([g] * len(members))
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_groups), dtype=np.float32)


def _hgnn_propagation(H: sp.spmatrix) -> sp.coo_matrix:
    """``G = Dv^-1 H De^-1 H^T Dv^-1`` -- the node-to-node hypergraph propagation matrix.

    Kept sparse: it is nonzero only between users that co-occur in some group, so on a
    dataset with many users and few groups it is almost empty (the reference builds it
    densely, which does not scale)."""
    dv = np.asarray(H.sum(axis=1)).ravel() + 1e-5
    de = np.asarray(H.sum(axis=0)).ravel() + 1e-5
    Dv = sp.diags(1.0 / dv)
    De = sp.diags(1.0 / de)
    return (Dv @ H @ De @ H.T @ Dv).tocoo()


def _row_normalize(A: sp.spmatrix) -> sp.coo_matrix:
    """Row-normalise a propagation matrix so each message is a weighted average of the
    neighbours (rows with no neighbours stay zero)."""
    rs = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
    return (sp.diags(inv) @ A).tocoo()


def _group_graph(members_idx, n_groups: int) -> sp.csr_matrix:
    """Group--group hypergraph: groups that share a member but are not identical, weighted
    by their number of common neighbours (``(d @ d.T) * d``, elementwise)."""
    sets = [set(int(u) for u in m) for m in members_idx]
    rows, cols = [], []
    for i in range(n_groups):
        for j in range(n_groups):
            if sets[i] & sets[j] and sets[i] ^ sets[j]:
                rows.append(i); cols.append(j)
    d = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n_groups, n_groups), dtype=np.float32).tocsr()
    return sp.csr_matrix(d.dot(d.transpose()).multiply(d))


def _corrupt_views(H: sp.csr_matrix, rng, coarse_frac=0.2, fine_frac=0.3):
    """The two scales. Coarse: keep a random ``coarse_frac`` of *users* (node dropout).
    Fine: per user, keep a random ``fine_frac`` of *groups* (hyperedge dropout, resampled
    for each user)."""
    n_users, n_groups = H.shape
    dense = H.toarray()

    theta = np.zeros(n_users, dtype=np.float32)
    theta[rng.choice(n_users, size=max(1, int(coarse_frac * n_users)), replace=False)] = 1.0
    coarse = dense * theta[:, None]

    fine = dense.copy()
    keep = max(1, int(fine_frac * n_groups))
    for u in range(n_users):
        beta = np.zeros(n_groups, dtype=np.float32)      # resampled per user (see module docstring)
        beta[rng.choice(n_groups, size=keep, replace=False)] = 1.0
        fine[u] = fine[u] * beta
    return sp.csr_matrix(fine), sp.csr_matrix(coarse)


class _HGNNConv(nn.Module):
    """One hypergraph convolution: ``G @ (X W + b)``."""

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        std = 1.0 / np.sqrt(out_dim)
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, x, g):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return torch.sparse.mm(g, x)


class _HyperGCN(nn.Module):
    def __init__(self, emb_dim, layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.hgnn = nn.ModuleList([_HGNNConv(emb_dim, emb_dim) for _ in range(layers)])

    def forward(self, x, g):
        x = F.normalize(x)
        for i, conv in enumerate(self.hgnn):
            x = conv(x, g)
            if i == 0:
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class _Discriminator(nn.Module):
    """Bilinear MI discriminator over two views."""

    def __init__(self, emb_dim):
        super().__init__()
        self.lin = nn.Linear(emb_dim, emb_dim, bias=True)
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        self.bilinear = nn.Bilinear(emb_dim, emb_dim, 1)
        nn.init.zeros_(self.bilinear.weight); nn.init.zeros_(self.bilinear.bias)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, a, b):
        return self.bilinear(self.lin(a), self.lin(b))

    def mi_loss(self, pos, neg):
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=1)
        logits = torch.cat([pos, neg], dim=1)
        p, n = pos.shape[1], neg.shape[1]
        return self.bce(logits, labels) * (p + n) / n


class _HHGRNet(nn.Module):
    def __init__(self, n_users, n_items, n_groups, emb_dim, drop_ratio):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.group_embedding = nn.Embedding(n_groups, emb_dim)
        for e in (self.user_embedding, self.item_embedding, self.group_embedding):
            nn.init.xavier_uniform_(e.weight)
        self.hgcn_coarse = _HyperGCN(emb_dim, layers=2, dropout=drop_ratio)
        self.hgcn_fine = _HyperGCN(emb_dim, layers=2, dropout=drop_ratio)
        self.hgcn_gl = _HyperGCN(emb_dim, layers=1, dropout=drop_ratio)
        self.discriminator = _Discriminator(emb_dim)
        # self-attentive pooling over [member ; item]
        self.score_layer = nn.Sequential(nn.Linear(2 * emb_dim, 8), nn.ReLU(),
                                         nn.Dropout(drop_ratio), nn.Linear(8, 1))

    def user_forward(self, u_idx, i_idx):
        return (self.user_embedding(u_idx) * self.item_embedding(i_idx)).sum(-1)

    def group_item_scores(self, g_idx, i_idx, members_pad, members_mask,
                          user_emb_all, group_emb_all):
        """Scores for aligned ``(g_idx, i_idx)`` batches. ``members_pad`` (B, Mmax) holds the
        groups' member indices padded, ``members_mask`` (B, Mmax) marks the real ones.

        Vectorised over the batch, so the same path serves training and inference (the
        reference loops in Python over every (group, item) pair)."""
        item_emb = self.item_embedding(i_idx)                                  # (B, d)
        memb = self.user_embedding(members_pad) + user_emb_all[members_pad]    # (B, Mmax, d)
        pair = torch.cat([memb, item_emb.unsqueeze(1).expand_as(memb)], dim=-1)
        w = self.score_layer(pair)                                             # (B, Mmax, 1)
        w = w.masked_fill(~members_mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(w, dim=1)
        w = torch.nan_to_num(w)                                                # groups with no known member
        attn = (memb * w).sum(dim=1)                                           # (B, d)
        pure = self.group_embedding(g_idx) + group_emb_all[g_idx]              # (B, d)
        return ((attn + pure) * item_emb).sum(-1)


class HHGR:
    """Double-scale self-supervised hypergraph group recommender (``paradigm="profile"``).

    Defaults follow the reference implementation (``emb_dim=64``, ``drop_ratio=0.4``,
    ``batch_size=512``, ``num_negatives=10``, 5 epochs per stage and a single group epoch).
    On the real group benchmarks one group epoch is already tens of thousands of steps;
    small datasets need ``group_epochs`` raised.
    """

    paradigm = "profile"
    supports_member_weights = False   # transductive (group-id node); no member pooling to steer

    def __init__(self, groups: Groups, group_interactions, *, emb_dim: int = 64,
                 epochs: int = 5, group_epochs: int = 1, lr_pretrain: float = 5e-4,
                 lr_ssl: float = 5e-4, lr_group: float = 1e-4, drop_ratio: float = 0.4,
                 num_negatives: int = 10, batch_size: int = 512, coarse_frac: float = 0.2,
                 fine_frac: float = 0.3, weight_decay: float = 0.0, user_item: bool = True,
                 seed: int | None = 0, device: str = "cpu") -> None:
        self.groups = groups
        self._raw_gi = group_interactions
        self.emb_dim, self.epochs, self.group_epochs = emb_dim, epochs, group_epochs
        self.lr_pretrain, self.lr_ssl, self.lr_group = lr_pretrain, lr_ssl, lr_group
        self.drop_ratio, self.num_negatives, self.batch_size = drop_ratio, num_negatives, batch_size
        self.coarse_frac, self.fine_frac = coarse_frac, fine_frac
        self.weight_decay, self.user_item = weight_decay, user_item
        self.seed, self.device = seed, device
        self.dataset_: Dataset | None = None
        self.net_: _HHGRNet | None = None
        self._cache = None

    # -- fit ---------------------------------------------------------------- #
    def fit(self, dataset: Dataset) -> "HHGR":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.dataset_ = dataset
        U, I, G = dataset.n_users, dataset.n_items, len(self.groups)
        ui, ii = dataset.user_index, dataset.item_index
        members = [np.array([ui[u] for u in m if u in ui], dtype=np.int64) for m in self.groups]
        gi = normalize_group_interactions(self._raw_gi, G)
        group_items = [[ii[it] for it in gi.get(g, []) if it in ii] for g in range(G)]
        self._lookup = {tuple(sorted(m.tolist())): g for g, m in enumerate(members)}

        # padded member tensors, reused by every forward
        mmax = max((len(m) for m in members), default=1) or 1
        pad = np.zeros((G, mmax), dtype=np.int64)
        mask = np.zeros((G, mmax), dtype=bool)
        for g, m in enumerate(members):
            pad[g, :len(m)] = m
            mask[g, :len(m)] = True
        self._pad = torch.as_tensor(pad, device=self.device)
        self._mask = torch.as_tensor(mask, device=self.device)

        # the two hypergraph scales + the group-group graph
        H = _user_group_incidence(members, U, G)
        H_fine, H_coarse = _corrupt_views(H, rng, self.coarse_frac, self.fine_frac)
        self._G_fine = _sp_to_tensor(_hgnn_propagation(H_fine), self.device)
        self._G_coarse = _sp_to_tensor(_hgnn_propagation(H_coarse), self.device)
        self._G_gl = _sp_to_tensor(_row_normalize(_group_graph(members, G)), self.device)

        self.net_ = _HHGRNet(U, I, G, self.emb_dim, self.drop_ratio).to(self.device)
        self._u_pos = np.vstack([dataset.interactions["user"].map(ui).to_numpy(),
                                 dataset.interactions["item"].map(ii).to_numpy()]).T
        self._g_pos = np.array([(g, it) for g in range(G) for it in group_items[g]],
                               dtype=np.int64) if any(group_items) else np.zeros((0, 2), np.int64)
        self._cache = None
        self._train(rng, U, I, G)
        return self

    def _pairs(self, pos, rng, n_items):
        """(entity, pos_item, neg_item) rows, one negative per row."""
        ent = np.repeat(pos[:, 0], self.num_negatives)
        pit = np.repeat(pos[:, 1], self.num_negatives)
        neg = rng.integers(0, n_items, size=ent.shape[0])
        perm = rng.permutation(ent.shape[0])
        return ent[perm], pit[perm], neg[perm]

    def _train(self, rng, U, I, G) -> None:
        net, bs = self.net_, self.batch_size
        net.train()

        # -- stage 1: user-item pre-training -------------------------------- #
        if self.user_item and self._u_pos.shape[0]:
            opt = torch.optim.Adam(net.parameters(), lr=self.lr_pretrain, weight_decay=self.weight_decay)
            for _ in range(self.epochs):
                u, p, n = self._pairs(self._u_pos, rng, I)
                for s in range(0, u.shape[0], bs):
                    ut = torch.as_tensor(u[s:s + bs], device=self.device)
                    pt = torch.as_tensor(p[s:s + bs], device=self.device)
                    nt = torch.as_tensor(n[s:s + bs], device=self.device)
                    loss = ((net.user_forward(ut, pt) - net.user_forward(ut, nt) - 1) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()

        # -- stage 2: user-level self-supervision (the double-scale MI) ------ #
        # The two views are recomputed per step: they depend on parameters the optimizer is
        # updating, so a graph retained across steps would be stale (the reference retains it
        # and only works on older autograd).
        opt = torch.optim.Adam(net.parameters(), lr=self.lr_ssl, weight_decay=self.weight_decay)
        all_u = torch.arange(U, device=self.device)
        for _ in range(self.epochs):
            order = rng.permutation(U)
            negs = rng.integers(0, U, size=U)
            for s in range(0, U, bs):
                base = net.user_embedding(all_u).detach()
                coarse = net.hgcn_coarse(base, self._G_coarse)
                fine = net.hgcn_fine(base, self._G_fine)
                idx = torch.as_tensor(order[s:s + bs], device=self.device)
                nidx = torch.as_tensor(negs[s:s + bs], device=self.device)
                pos = net.discriminator(coarse[idx], fine[idx])            # (B, 1)
                neg = net.discriminator(coarse[idx], coarse[nidx])
                loss = net.discriminator.mi_loss(pos, neg)
                opt.zero_grad(); loss.backward(); opt.step()

        # -- stage 3: group-level ------------------------------------------- #
        if self._g_pos.shape[0]:
            opt = torch.optim.Adam(net.parameters(), lr=self.lr_group, weight_decay=self.weight_decay)
            all_g = torch.arange(G, device=self.device)
            # The user representation is frozen for this stage, so it must be the *clean* one:
            # computed in train mode it would bake in a fixed dropout mask that inference never
            # reproduces. Matters most where group-item data is scarce and the group score leans
            # on the members.
            net.eval()
            user_emb = self._user_embeddings(detach=True)
            net.train()
            for _ in range(self.group_epochs):
                g, p, n = self._pairs(self._g_pos, rng, I)
                for s in range(0, g.shape[0], bs):
                    group_emb = net.hgcn_gl(net.group_embedding(all_g), self._G_gl)
                    gt = torch.as_tensor(g[s:s + bs], device=self.device)
                    pt = torch.as_tensor(p[s:s + bs], device=self.device)
                    nt = torch.as_tensor(n[s:s + bs], device=self.device)
                    pad, mask = self._pad[gt], self._mask[gt]
                    pos = net.group_item_scores(gt, pt, pad, mask, user_emb, group_emb)
                    neg = net.group_item_scores(gt, nt, pad, mask, user_emb, group_emb)
                    loss = ((pos - neg - 1) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()

    def _user_embeddings(self, detach=False):
        """User representation = coarse view + fine view (the two scales, summed)."""
        net = self.net_
        base = net.user_embedding(torch.arange(self.dataset_.n_users, device=self.device))
        if detach:
            base = base.detach()
        emb = net.hgcn_coarse(base, self._G_coarse) + net.hgcn_fine(base, self._G_fine)
        return emb.detach() if detach else emb

    # -- score / recommend -------------------------------------------------- #
    def group_scores(self, members, items=None, *, member_weights=None,
                     return_attention=False) -> np.ndarray:
        """Per-item group scores. HHGR is transductive, so ``member_weights`` /
        ``return_attention`` are not defined and raise."""
        _reject_member_options(self, member_weights, return_attention)
        if self.net_ is None:
            raise RuntimeError("HHGR must be fit() before scoring.")
        ui = self.dataset_.user_index
        gidx = self._lookup.get(tuple(sorted(ui[u] for u in members if u in ui)))
        net = self.net_
        net.eval()
        with torch.no_grad():
            if self._cache is None:                       # embeddings are static after fit
                all_g = torch.arange(len(self.groups), device=self.device)
                self._cache = (self._user_embeddings(detach=True),
                               net.hgcn_gl(net.group_embedding(all_g), self._G_gl))
            user_emb, group_emb = self._cache
            if gidx is None:                              # ephemeral group -> nearest by members
                gidx = 0
            item_idx = (torch.arange(self.dataset_.n_items, device=self.device) if items is None
                        else torch.as_tensor([self.dataset_.item_index[i] for i in items],
                                             dtype=torch.long, device=self.device))
            n = item_idx.shape[0]
            g_idx = torch.full((n,), gidx, dtype=torch.long, device=self.device)
            pad = self._pad[g_idx]
            mask = self._mask[g_idx]
            scores = net.group_item_scores(g_idx, item_idx, pad, mask, user_emb, group_emb)
        return scores.cpu().numpy()

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


__all__ = ["HHGR"]
