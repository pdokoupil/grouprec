"""HyperGroup -- Hierarchical Hyperedge Embedding-based Representation Learning for Group
Recommendation (Guo et al., TOIS'21; arXiv:2103.13506). Reimplemented against the paper and
the reference implementation in ``WWW2023GroupRecBaselines/HyperGroup``.

Each group is a hyperedge. A group's representation starts as a mean pool of its members'
embeddings and is then refined by hyperedge-level convolution over a group--group graph
(two groups are adjacent when they share members, weighted by how many): every layer mixes
in a message from the neighbouring groups' embeddings *and* a message built from the
members those groups have in common, then combines it with the group's own embedding
through a learned linear layer. Items are scored by an MLP over the elementwise product of
the group and item embeddings, and the model is trained with a softplus pairwise ranking
loss on group--item and user--item interactions.

Transductive (a group is addressed by its index in the hypergraph), so ``recommend`` maps a
member set back to its group index and ``supports_member_weights`` is ``False``.

Note on the common-member message: the reference computes it with a Python double loop over
every (group, neighbour) pair, averaging the shared members' embeddings and weighting by the
overlap count. Because the overlap count *is* the number of shared members, that weight
cancels against the average exactly, so the whole loop reduces to
``(M * (A_bin @ M)) @ user_emb`` -- which is what we compute. It is the same quantity, but
vectorised instead of quadratic in the number of groups.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Dataset, Groups
from .data import _reject_member_options, normalize_group_interactions


class _HyperGroupNet(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers):
        super().__init__()
        self.emb_dim, self.layers = emb_dim, layers
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.weight = nn.ModuleList([nn.Linear(2 * emb_dim, emb_dim) for _ in range(layers)])
        # Returns the pre-sigmoid logit. The reference ends this MLP in a Sigmoid and both
        # trains and ranks on its output; we apply the sigmoid only in the loss. Ranking is
        # unaffected in principle (the sigmoid is monotone) but not in practice: once trained,
        # its output saturates to exactly 0/1 in float, collapsing ~101 candidates to a handful
        # of distinct scores. The resulting ties then decide the ranking rather than the model.
        self.predictor = nn.Sequential(nn.Linear(emb_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def user_scores(self, u_idx, i_idx):
        return self.predictor(self.user_embedding(u_idx) * self.item_embedding(i_idx)).squeeze(-1)

    def group_embeddings(self, membership, member_mask, adj, adj_bin):
        """Hierarchical hyperedge embeddings for every group.

        ``membership`` (G, Mmax) padded member indices, ``member_mask`` (G, Mmax) bool,
        ``adj`` (G, G) overlap counts with zero diagonal, ``adj_bin`` its binary form.
        """
        # Step 1 -- mean pool the members
        memb = self.user_embedding(membership) * member_mask.unsqueeze(2)
        emb = memb.sum(1) / member_mask.sum(1, keepdim=True).clamp(min=1)

        # Step 2.1 -- message from the members shared with neighbouring groups.
        # Independent of `emb`, so computed once (see the module docstring for why the
        # reference's overlap weighting cancels).
        M = member_mask.new_zeros((membership.shape[0], self.user_embedding.num_embeddings),
                                  dtype=torch.float32)
        M.scatter_(1, membership, member_mask.float())
        shared = M * (adj_bin @ M)                          # (G, n_users)
        neigh_member_msg = shared @ self.user_embedding.weight

        for i in range(self.layers):
            # Step 2.2 -- message from the neighbouring groups themselves
            neigh_group_msg = adj @ emb
            messages = neigh_group_msg + neigh_member_msg
            # Step 2.3 -- combine with the group's own embedding
            emb = self.weight[i](torch.cat([emb, messages], dim=1))
            emb = F.normalize(emb, dim=-1)
        return emb

    def group_scores(self, g_idx, i_idx, all_group_emb):
        return self.predictor(all_group_emb[g_idx] * self.item_embedding(i_idx)).squeeze(-1)


class HyperGroup:
    """Hierarchical hyperedge embedding group recommender (``paradigm="profile"``)."""

    paradigm = "profile"
    supports_member_weights = False   # transductive (group addressed by hyperedge index)

    def __init__(self, groups: Groups, group_interactions, *, emb_dim: int = 64,
                 layers: int = 2, epochs: int = 30, lr: float = 1e-2,
                 num_negatives: int = 4, batch_size: int = 512, weight_decay: float = 0.0,
                 user_item: bool = True, seed: int | None = 0, device: str = "cpu") -> None:
        self.groups = groups
        self._raw_gi = group_interactions
        self.emb_dim, self.layers, self.epochs, self.lr = emb_dim, layers, epochs, lr
        self.num_negatives, self.batch_size = num_negatives, batch_size
        self.weight_decay, self.user_item = weight_decay, user_item
        self.seed, self.device = seed, device
        self.dataset_: Dataset | None = None
        self.net_: _HyperGroupNet | None = None
        self._cache = None

    def fit(self, dataset: Dataset) -> "HyperGroup":
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

        mmax = max((len(m) for m in members), default=1) or 1
        pad = np.zeros((G, mmax), dtype=np.int64)
        mask = np.zeros((G, mmax), dtype=bool)
        for g, m in enumerate(members):
            pad[g, :len(m)] = m
            mask[g, :len(m)] = True
        self._membership = torch.as_tensor(pad, device=self.device)
        self._member_mask = torch.as_tensor(mask, device=self.device)

        # group-group adjacency = shared-member counts, without self-loops
        H = np.zeros((G, U), dtype=np.float32)
        for g, m in enumerate(members):
            H[g, m] = 1.0
        adj = H @ H.T
        np.fill_diagonal(adj, 0.0)
        self._adj = torch.as_tensor(adj, device=self.device)
        self._adj_bin = torch.as_tensor((adj > 0).astype(np.float32), device=self.device)

        self.net_ = _HyperGroupNet(U, I, self.emb_dim, self.layers).to(self.device)
        self._u_pos = np.vstack([dataset.interactions["user"].map(ui).to_numpy(),
                                 dataset.interactions["item"].map(ii).to_numpy()]).T
        self._g_pos = np.array([(g, it) for g in range(G) for it in group_items[g]],
                               dtype=np.int64) if any(group_items) else np.zeros((0, 2), np.int64)
        self._cache = None
        self._train(rng, I)
        return self

    def _pairs(self, pos, rng, n_items):
        ent = np.repeat(pos[:, 0], self.num_negatives)
        pit = np.repeat(pos[:, 1], self.num_negatives)
        neg = rng.integers(0, n_items, size=ent.shape[0])
        perm = rng.permutation(ent.shape[0])
        return ent[perm], pit[perm], neg[perm]

    def _train(self, rng, I) -> None:
        net, bs = self.net_, self.batch_size
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        net.train()
        for _ in range(self.epochs):
            if self._g_pos.shape[0]:
                g, p, n = self._pairs(self._g_pos, rng, I)
                for s in range(0, g.shape[0], bs):
                    emb = net.group_embeddings(self._membership, self._member_mask,
                                               self._adj, self._adj_bin)
                    gt = torch.as_tensor(g[s:s + bs], device=self.device)
                    pt = torch.as_tensor(p[s:s + bs], device=self.device)
                    nt = torch.as_tensor(n[s:s + bs], device=self.device)
                    # sigmoid here keeps the reference's objective; scoring stays on the logit
                    loss = F.softplus(torch.sigmoid(net.group_scores(gt, nt, emb))
                                      - torch.sigmoid(net.group_scores(gt, pt, emb))).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            if self.user_item and self._u_pos.shape[0]:
                u, p, n = self._pairs(self._u_pos, rng, I)
                for s in range(0, u.shape[0], bs):
                    ut = torch.as_tensor(u[s:s + bs], device=self.device)
                    pt = torch.as_tensor(p[s:s + bs], device=self.device)
                    nt = torch.as_tensor(n[s:s + bs], device=self.device)
                    loss = F.softplus(torch.sigmoid(net.user_scores(ut, nt))
                                      - torch.sigmoid(net.user_scores(ut, pt))).mean()
                    opt.zero_grad(); loss.backward(); opt.step()

    def group_scores(self, members, items=None, *, member_weights=None,
                     return_attention=False) -> np.ndarray:
        """Per-item group scores. HyperGroup is transductive, so ``member_weights`` /
        ``return_attention`` are not defined and raise."""
        _reject_member_options(self, member_weights, return_attention)
        if self.net_ is None:
            raise RuntimeError("HyperGroup must be fit() before scoring.")
        ui = self.dataset_.user_index
        gidx = self._lookup.get(tuple(sorted(ui[u] for u in members if u in ui)))
        net = self.net_
        net.eval()
        with torch.no_grad():
            if self._cache is None:                # embeddings are static after fit
                self._cache = net.group_embeddings(self._membership, self._member_mask,
                                                   self._adj, self._adj_bin)
            emb = self._cache
            if gidx is None:
                gidx = 0
            item_idx = (torch.arange(self.dataset_.n_items, device=self.device) if items is None
                        else torch.as_tensor([self.dataset_.item_index[i] for i in items],
                                             dtype=torch.long, device=self.device))
            g_idx = torch.full((item_idx.shape[0],), gidx, dtype=torch.long, device=self.device)
            scores = net.group_scores(g_idx, item_idx, emb)
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


__all__ = ["HyperGroup"]
