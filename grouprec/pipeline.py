"""Group recommenders: bind a base (single-user) recommender to an aggregator.

This is the unifying layer from the design: a :class:`GroupRecommender` declares its
paradigm and the (forthcoming) evaluator drives both paradigms uniformly. Step 4
ships the results-first path; profile-first lands with the profile aggregators.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .aggregators._normalize import normalize_mgains
from .aggregators.base import Aggregator
from .backends import BaseRecommender
from .data import Dataset


class GroupRecommender:
    """Results-first group recommender: score each member, then aggregate.

    Parameters
    ----------
    base : a fitted-or-unfitted :class:`~grouprec.backends.BaseRecommender`.
    aggregator : a results-first :class:`~grouprec.aggregators.base.Aggregator`.

    For stateful (sequential) aggregators, call :meth:`recommend` once per session;
    the aggregator carries its fairness state across calls (call ``aggregator.reset()``
    to start a fresh group).

    Parameters
    ----------
    normalize : per-member normalization applied to the base RS scores *before*
        aggregation (``None``, ``"minmax"``, ``"standard"``, ``"robust"``,
        ``"quantile"``). The proportional-fairness aggregators (GFAR, EP-FuzzDA,
        RLProp, LTP) assume **commensurable** member scores; the GFAR / EP-FuzzDA /
        KAIS papers min-max normalize ``r̂`` per user. Use ``"minmax"`` to reproduce
        them — raw MF scores on different per-user scales make these methods degenerate.
    """

    paradigm = "results"

    def __init__(self, base: BaseRecommender, aggregator: Aggregator,
                 *, normalize: str | None = None) -> None:
        self.base = base
        self.aggregator = aggregator
        self.normalize = normalize
        self.dataset_: Dataset | None = None

    def fit(self, dataset: Dataset) -> "GroupRecommender":
        self.base.fit(dataset)
        self.dataset_ = dataset
        return self

    def recommend(
        self,
        members,
        k: int,
        *,
        exclude: Iterable | None = None,
        candidates: Iterable | None = None,
    ) -> np.ndarray:
        """Recommend ``k`` item **ids** to a group given its member user ids.

        ``exclude`` is an iterable of item *ids* never to recommend. ``candidates``
        restricts scoring/ranking to a given item-id set (the sampled-ranking
        protocol) -- the aggregator then ranks *within* the candidates, which is
        essential for ranking-aware aggregators (GFAR etc.).
        """
        if self.dataset_ is None:
            raise RuntimeError("GroupRecommender must be fit() before recommending.")

        if candidates is not None:
            cand = list(candidates)
            scores = np.asarray(self.base.score(members, items=cand), dtype=float)
            if self.normalize:
                scores = normalize_mgains(scores, self.normalize)
            idx = self.aggregator.aggregate(scores, k)
            return np.asarray(cand)[idx]

        scores = np.asarray(self.base.score(members), dtype=float)
        if scores.shape != (len(list(members)), self.dataset_.n_items):
            raise ValueError(
                f"base.score returned {scores.shape}, expected "
                f"{(len(list(members)), self.dataset_.n_items)}."
            )
        if self.normalize:
            scores = normalize_mgains(scores, self.normalize)
        ex_idx = None
        if exclude is not None:
            ex_idx = [self.dataset_.item_index[i] for i in exclude
                      if i in self.dataset_.item_index]
        idx = self.aggregator.aggregate(scores, k, exclude=ex_idx)
        return self.dataset_.items[idx]


__all__ = ["GroupRecommender"]
