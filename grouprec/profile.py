"""Profile-first (profile-aggregation) group recommendation.

The complement of results aggregation: instead of *recommend-then-aggregate* (score
each member, combine the lists), this does *aggregate-then-recommend* -- merge the
members' interaction profiles into one **pseudo-user profile**, then query the base RS
once. This is the classic "profile aggregation / pseudo-user" GRS strategy (Masthoff,
*Group Recommender Systems*, 2015).

Requires a base recommender exposing ``score_profile(profiles)`` (the built-in
``EASE`` / ``ItemKNN`` do; they score ``profile @ W``). ``merge`` is the
profile-aggregation function:

* ``"average"`` — mean of member interaction vectors (a.k.a. virtual/average user).
* ``"union"``   — element-wise max (any member interacted) — concatenated profile.
* ``"sum"``     — additive profile.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse

from .backends import BaseRecommender
from .data import Dataset


class ProfileGroupRecommender:
    """Profile-first group recommender (``paradigm="profile"``)."""

    paradigm = "profile"

    def __init__(self, base: BaseRecommender, *, merge: str = "average",
                 binarize: bool = True) -> None:
        if merge not in ("average", "union", "sum"):
            raise ValueError("merge must be 'average', 'union', or 'sum'.")
        self.base = base
        self.merge = merge
        self.binarize = binarize
        self.dataset_: Dataset | None = None

    def fit(self, dataset: Dataset) -> "ProfileGroupRecommender":
        if not hasattr(self.base, "score_profile"):
            raise TypeError(
                f"{type(self.base).__name__} has no score_profile(); profile-first needs a "
                "profile-scoring backend (e.g. EASE, ItemKNN)."
            )
        self.base.fit(dataset)
        self.dataset_ = dataset
        self.M_ = dataset.user_item_csr(value="binary" if self.binarize else "rating")
        return self

    def _profile(self, members) -> np.ndarray:
        idx = [self.dataset_.user_index[u] for u in members if u in self.dataset_.user_index]
        if not idx:
            return np.zeros(self.dataset_.n_items)
        rows = self.M_[idx]
        if sparse.issparse(rows):
            # Only the members' own rows are densified -- (n_members, n_items), tiny.
            rows = rows.toarray()
        if self.merge == "average":
            return rows.mean(axis=0)
        if self.merge == "union":
            return rows.max(axis=0)
        return rows.sum(axis=0)

    def recommend(self, members, k: int, *, exclude: Iterable | None = None,
                  candidates: Iterable | None = None) -> np.ndarray:
        if self.dataset_ is None:
            raise RuntimeError("ProfileGroupRecommender must be fit() before recommending.")
        profile = self._profile(members)[None, :]
        if candidates is not None:
            cand = list(candidates)
            scores = self.base.score_profile(profile, items=cand)[0]
            return np.asarray(cand)[np.argsort(-scores, kind="stable")[:k]]
        scores = np.asarray(self.base.score_profile(profile), dtype=float)[0]
        if exclude is not None:
            ex = [self.dataset_.item_index[i] for i in exclude if i in self.dataset_.item_index]
            scores[ex] = -np.inf
        budget = int(min(k, np.isfinite(scores).sum()))
        return self.dataset_.items[np.argsort(-scores, kind="stable")[:budget]]


__all__ = ["ProfileGroupRecommender"]
