"""GroupIM -- Mutual-Information-maximization group recommendation (Sako/Sankar et al.,
SIGIR'20). Reimplemented from https://github.com/CrowdDynamicsLab/GroupIM.

Unlike the AGREE family, users are represented by **encoding their item bag-of-words**
(not an id embedding); member encodings are pooled (attention/max/mean) into a group
embedding, a linear item predictor scores items, and a bilinear **InfoMax
discriminator** maximizes mutual information between the group and its members
(contrasting real members against corrupted random users). Loss =
``group_loss + lambda_mi * user_group_loss + mi_loss``.

Attribution: architecture/loss follow the original GroupIM repo (retained here).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Dataset, Groups
from .data import normalize_group_interactions


class _Encoder(nn.Module):
    def __init__(self, n_items, layers, d, drop):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.enc = nn.ModuleList()
        for in_s, out_s in zip([n_items] + list(layers[:-1]), layers):
            lin = nn.Linear(in_s, out_s)
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            self.enc.append(lin)
        self.transform = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.transform.weight); nn.init.zeros_(self.transform.bias)
        self.user_predictor = nn.Linear(d, n_items, bias=False)
        nn.init.xavier_uniform_(self.user_predictor.weight)

    def pretrain(self, user_items):
        h = self.drop(F.normalize(user_items, dim=-1))
        for lin in self.enc:
            h = torch.tanh(lin(h))
        return self.user_predictor(h), h

    def forward(self, user_items):
        _, h = self.pretrain(user_items)
        return torch.tanh(self.transform(h))


class _AttentionAggregator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = nn.Linear(d, 1)

    def forward(self, x, mask):  # x: [B,G,D], mask: [B,G] (-inf/0)
        a = torch.tanh(self.attention(x))                       # [B,G,1]
        w = torch.softmax(a + mask.unsqueeze(2), dim=1)         # [B,G,1]
        return torch.matmul(x.transpose(2, 1), w).squeeze(2)    # [B,D]


class _Discriminator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.bilinear = nn.Bilinear(d, d, 1)
        nn.init.zeros_(self.bilinear.weight); nn.init.zeros_(self.bilinear.bias)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, group_embed, user_embed, _mask):
        g = torch.tanh(self.fc(group_embed))                    # [B,D]
        u = torch.tanh(self.fc(user_embed))                     # [B,n,D]
        return self.bilinear(u, g.unsqueeze(1).repeat(1, user_embed.shape[1], 1))  # [B,n,1]

    def mi_loss(self, s_obs, mask, s_corr, device):
        B = s_obs.shape[0]
        P, N = s_obs.shape[1], s_corr.shape[1]
        labels = torch.cat([torch.ones(B, P, device=device), torch.zeros(B, N, device=device)], 1)
        logits = torch.cat([s_obs, s_corr], 1).squeeze(2)
        m = torch.cat([torch.exp(mask), torch.ones(B, N, device=device)], 1)
        return self.bce(logits * m, labels * m) * (B * (P + N)) / (torch.exp(mask).sum() + B * N)


class _GroupIMNet(nn.Module):
    def __init__(self, n_items, layers, drop, lambda_mi):
        super().__init__()
        d = layers[-1]
        self.lambda_mi = lambda_mi
        self.encoder = _Encoder(n_items, layers, d, drop)
        self.aggregator = _AttentionAggregator(d)
        self.group_predictor = nn.Linear(d, n_items, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)
        self.discriminator = _Discriminator(d)

    def group_logits(self, member_vecs, mask):
        embeds = self.encoder(member_vecs)                      # [B,G,D]
        g = self.aggregator(embeds, mask)                       # [B,D]
        return self.group_predictor(g), g

    @staticmethod
    def _multinomial(logits, target):
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * target, -1))

    def loss(self, member_vecs, mask, group_target, corrupt_vecs, device):
        logits, g = self.group_logits(member_vecs, mask)
        member_embeds = self.encoder(member_vecs)              # [B,G,D]
        corrupt_embeds = self.encoder(corrupt_vecs)           # [B,N,D]
        s_obs = self.discriminator(g, member_embeds, mask)    # [B,G,1]
        s_corr = self.discriminator(g, corrupt_embeds, mask)  # [B,N,1]
        mi = self.discriminator.mi_loss(s_obs, mask, s_corr, device)

        # user-group loss: member item vectors weighted by (detached) MI scores
        ui = member_vecs.sum(2, keepdim=True)
        uv_norm = member_vecs / torch.max(torch.ones_like(ui), ui)
        w = torch.sigmoid(s_obs.detach())                     # [B,G,1]
        mzero = torch.exp(mask).unsqueeze(2)                  # [B,G,1]
        ug_target = (uv_norm * w * mzero).sum(1) / mzero.sum(1)
        ug_loss = self._multinomial(logits, ug_target)
        group_loss = self._multinomial(logits, group_target)
        return group_loss + mi + self.lambda_mi * ug_loss


class GroupIM:
    """Mutual-information group recommender (``paradigm="profile"``)."""

    paradigm = "profile"

    def __init__(self, groups: Groups, group_interactions, *, embedding_dim: int = 64,
                 layers: tuple = (64,), lambda_mi: float = 0.1, drop: float = 0.4,
                 epochs: int = 30, pretrain_epochs: int = 10, lr: float = 0.005,
                 weight_decay: float = 0.0, neg_users: int = 5, batch_size: int = 128,
                 seed: int | None = 0, device: str = "cpu") -> None:
        self.groups = groups
        self._raw_gi = group_interactions
        self.layers = list(layers[:-1]) + [embedding_dim]
        self.lambda_mi = lambda_mi
        self.drop = drop
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_users = neg_users
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.dataset_: Dataset | None = None
        self.net_: _GroupIMNet | None = None

    def fit(self, dataset: Dataset) -> "GroupIM":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.dataset_ = dataset
        n_items = dataset.n_items
        self.M_ = torch.as_tensor(dataset.user_item_matrix(value="binary"),
                                  dtype=torch.float32, device=self.device)
        ui = dataset.user_index
        ii = dataset.item_index
        self._members = [np.array([ui[u] for u in m if u in ui], dtype=np.int64)
                         for m in self.groups]
        gi = normalize_group_interactions(self._raw_gi, len(self.groups))
        self._group_items = []
        for idx in range(len(self.groups)):
            vec = np.zeros(n_items, dtype=np.float32)
            for it in gi.get(idx, []):
                if it in ii:
                    vec[ii[it]] = 1.0
            self._group_items.append(vec)

        self.net_ = _GroupIMNet(n_items, self.layers, self.drop, self.lambda_mi).to(self.device)
        rng = np.random.default_rng(self.seed)
        self._pretrain(rng)
        self._train_groups(rng)
        return self

    def _pretrain(self, rng):
        if self.pretrain_epochs <= 0:
            return
        net, M = self.net_, self.M_
        opt = torch.optim.Adam(net.encoder.parameters(), lr=self.lr)
        n = M.shape[0]
        for _ in range(self.pretrain_epochs):
            for s in range(0, n, self.batch_size):
                uv = M[s: s + self.batch_size]
                logits, _ = net.encoder.pretrain(uv)
                tgt = uv / torch.clamp(uv.sum(1, keepdim=True), min=1.0)
                loss = net._multinomial(logits, tgt)
                opt.zero_grad(); loss.backward(); opt.step()

    def _batch(self, idxs):
        gmax = max(self._members[i].size for i in idxs)
        gmax = max(gmax, 1)
        B, I = len(idxs), self.M_.shape[1]
        vecs = torch.zeros(B, gmax, I, device=self.device)
        mask = torch.full((B, gmax), float("-inf"), device=self.device)
        target = torch.zeros(B, I, device=self.device)
        for b, gidx in enumerate(idxs):
            members = self._members[gidx]
            for j, m in enumerate(members):
                vecs[b, j] = self.M_[m]
                mask[b, j] = 0.0
            gv = torch.as_tensor(self._group_items[gidx], device=self.device)
            target[b] = gv / torch.clamp(gv.sum(), min=1.0)
        return vecs, mask, target

    def _train_groups(self, rng):
        net, M = self.net_, self.M_
        n_users = M.shape[0]
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        order = np.arange(len(self.groups))
        net.train()
        for _ in range(self.epochs):
            rng.shuffle(order)
            for s in range(0, len(order), self.batch_size):
                idxs = order[s: s + self.batch_size]
                vecs, mask, target = self._batch(idxs)
                corrupt = M[rng.integers(0, n_users, size=(len(idxs), self.neg_users))]
                loss = net.loss(vecs, mask, target, corrupt, self.device)
                opt.zero_grad(); loss.backward(); opt.step()

    def group_scores(self, members, items=None, *, member_weights=None,
                     return_attention=False):
        """Per-item group scores for a member set.

        ``member_weights`` (one non-negative weight per member, any scale) reweights the
        attention pooling -- ``g = sum_m w'_m * h_m`` with ``w'_m proportional to
        w_m * alpha_m`` -- yielding a *steerable* group representation; ``None``/uniform
        reproduces the native model exactly. With ``return_attention=True`` also returns
        the per-member pooling weights ``w'`` (an interpretable attribution).
        """
        if self.net_ is None:
            raise RuntimeError("GroupIM must be fit() before scoring.")
        ui, ii = self.dataset_.user_index, self.dataset_.item_index
        midx = [ui[u] for u in members if u in ui]
        net = self.net_
        net.eval()
        with torch.no_grad():
            if midx:
                vecs = self.M_[midx].unsqueeze(0)                       # [1,M,I]
            else:
                vecs = torch.zeros(1, 1, self.M_.shape[1], device=self.device)
            embeds = net.encoder(vecs)                                  # [1,M,D]
            a = torch.tanh(net.aggregator.attention(embeds))           # [1,M,1]
            w = torch.softmax(a, dim=1).squeeze(0).squeeze(-1)         # [M] native pooling
            if member_weights is not None:
                mw = torch.as_tensor(member_weights, dtype=torch.float32, device=self.device)
                w = w * mw
                w = w / (w.sum() + 1e-12)
            g = (embeds.squeeze(0) * w[:, None]).sum(0, keepdim=True)   # [1,D]
            scores = net.group_predictor(g).squeeze(0).cpu().numpy()
        if items is not None:
            scores = scores[np.array([ii[i] for i in items], dtype=np.int64)]
        return (scores, w.cpu().numpy()) if return_attention else scores

    def recommend(self, members, k: int, *, exclude=None, candidates=None,
                  member_weights=None) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("GroupIM must be fit() before recommending.")
        if candidates is not None:
            cand = list(candidates)
            scores = self.group_scores(members, cand, member_weights=member_weights)
            return np.asarray(cand)[np.argsort(-scores, kind="stable")[:k]]
        scores = self.group_scores(members, member_weights=member_weights)
        if exclude:
            ex = [self.dataset_.item_index[i] for i in exclude if i in self.dataset_.item_index]
            scores[ex] = -np.inf
        budget = int(min(k, np.isfinite(scores).sum()))
        return self.dataset_.items[np.argsort(-scores, kind="stable")[:budget]]


__all__ = ["GroupIM"]
