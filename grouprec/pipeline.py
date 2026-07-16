"""Group recommenders: bind a base (single-user) recommender to an aggregator.

This is the unifying layer: a :class:`GroupRecommender` declares its ``paradigm``, so
the evaluator drives results-first and profile-first models through one interface.
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

    For stateful (sequential) aggregators, call :meth:`recommend` once per session;
    the aggregator carries its fairness state across calls (call ``aggregator.reset()``
    to start a fresh group).

    Parameters
    ----------
    base : a fitted-or-unfitted :class:`~grouprec.backends.BaseRecommender`.
    aggregator : a results-first :class:`~grouprec.aggregators.base.Aggregator`.
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

    @classmethod
    def from_fitted(cls, base: BaseRecommender, aggregator: Aggregator,
                    dataset: Dataset, *, normalize: str | None = None
                    ) -> "GroupRecommender":
        """Bind an **already fitted** ``base`` to ``aggregator``, skipping the refit.

        Scoring the members is independent of how their scores are aggregated, so one
        fitted base serves any number of aggregators. This is the usual way to compare
        the results-aggregation family: fit the base once, then sweep aggregators
        (or member weights) over it, rather than paying for a refit per configuration.

        ``base`` must already be fitted on ``dataset``; nothing here checks that.
        """
        rec = cls(base, aggregator, normalize=normalize)
        rec.dataset_ = dataset
        return rec

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

    # -- score introspection ----------------------------------------------- #
    def member_scores(self, members, items=None) -> np.ndarray:
        """Per-member item scores -- the base recommender's predictions, normalised as
        configured. Shape ``(n_members, n_items or len(items))``, columns in
        ``dataset.items`` order (or ``items`` order if given).

        This is the *individual* view that the aggregator consumes; it is exactly the
        base RS output (this is the sense in which "per-member estimate == base RS").
        """
        if self.dataset_ is None:
            raise RuntimeError("GroupRecommender must be fit() before scoring.")
        scores = np.asarray(self.base.score(members, items=items), dtype=float)
        return normalize_mgains(scores, self.normalize) if self.normalize else scores

    def group_scores(self, members, items=None) -> np.ndarray:
        """Aggregated per-item *group* utility, shape ``(n_items or len(items),)``.

        This is the score the group ranking sorts on -- **not** the per-member base RS
        output (use :meth:`member_scores` for that). Defined only when the bound
        aggregator is score-reduction-based (ADD/AVG/wAVG/LMS/MUL/MPL/AVGNM/BDC);
        selection-based aggregators (EP-FuzzDA, GFAR, FAI, sequential) raise, since they
        emit an ordering rather than item scores -- use :meth:`recommend` for those.
        """
        return self.aggregator.score_items(self.member_scores(members, items=items))


__all__ = ["GroupRecommender"]
