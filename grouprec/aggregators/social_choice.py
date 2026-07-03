"""Social-choice (score/preference) aggregators -- numpy only, core install.

All operate on a dense ``(n_members, n_items)`` score matrix and are vectorized
ports of the reference pandas implementations in
``raw/aggregator_and_sequential/kais/aggregators_new.py``. Tie-breaking matches
the reference convention (stable descending sort -> smaller item index wins); see
:func:`grouprec.aggregators.base.top_k_indices`.
"""

from __future__ import annotations


import numpy as np
from scipy.stats import rankdata

from .base import Aggregator, as_score_matrix, available_mask, top_k_indices


def _mask_excluded(util: np.ndarray, exclude) -> np.ndarray:
    """Return ``util`` with excluded item positions set to ``-inf`` (never selected)."""
    if exclude is None:
        return util
    return np.where(available_mask(util.shape[0], exclude), util, -np.inf)


class _ReductionAggregator(Aggregator):
    """Shared machinery for score-reduction aggregators: a per-item utility (``_reduce``)
    that both the ranking (:meth:`aggregate`) and the public :meth:`score_items` use."""

    produces_item_scores = True

    def _reduce(self, rm: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_items(self, scores, *, exclude=None):
        return _mask_excluded(self._reduce(as_score_matrix(scores)), exclude)

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        return top_k_indices(self._reduce(rm), k, available_mask(rm.shape[1], exclude))


class AdditiveAggregator(_ReductionAggregator):
    """ADD -- sum of member scores."""

    name = "ADD"

    def _reduce(self, rm):
        return rm.sum(axis=0)


class AverageAggregator(_ReductionAggregator):
    """AVG -- mean of member scores."""

    name = "AVG"

    def _reduce(self, rm):
        return rm.mean(axis=0)


class WeightedAverageAggregator(_ReductionAggregator):
    """wAVG -- member-importance-weighted mean of member scores.

    ``member_weights`` are relative per-member importances (any non-negative scale);
    they are normalized to sum to one, so uniform weights recover :class:`AverageAggregator`.
    The member order must match the rows of the score matrix.
    """

    name = "wAVG"

    def __init__(self, member_weights=None) -> None:
        self._member_weights = None if member_weights is None else np.asarray(member_weights, dtype=float)

    def _weights(self, m: int) -> np.ndarray:
        if self._member_weights is None:
            return np.full(m, 1.0 / m)
        w = self._member_weights
        if w.size != m:
            raise ValueError(f"member_weights has {w.size} entries but group has {m} members.")
        s = w.sum()
        return np.full(m, 1.0 / m) if s <= 0 else w / s

    def _reduce(self, rm):
        return (self._weights(rm.shape[0])[:, None] * rm).sum(axis=0)


class LeastMiseryAggregator(_ReductionAggregator):
    """LMS -- minimum (least misery) of member scores."""

    name = "LMS"

    def _reduce(self, rm):
        return rm.min(axis=0)


class MultiplicativeAggregator(_ReductionAggregator):
    """MUL -- product of member scores."""

    name = "MUL"

    def _reduce(self, rm):
        return rm.prod(axis=0)


class MostPleasureAggregator(_ReductionAggregator):
    """MPL -- maximum (most pleasure) of member scores."""

    name = "MPL"

    def _reduce(self, rm):
        return rm.max(axis=0)


class AVGNoMiseryAggregator(_ReductionAggregator):
    """AVGNM -- average, restricted to items whose worst member score exceeds a
    misery threshold.

    Items where ``min_member_score <= threshold`` are dropped (utility ``-inf``); the
    rest are ranked by mean score. Matches ``avgnm_algorithm`` (strict ``> threshold``).
    The reference hard-codes ``threshold=1`` for graded ML ratings; here it is a
    parameter (default ``0.0``).
    """

    name = "AVGNM"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def _reduce(self, rm):
        return np.where(rm.min(axis=0) > self.threshold, rm.mean(axis=0), -np.inf)

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        allowed = (rm.min(axis=0) > self.threshold) & available_mask(rm.shape[1], exclude)
        if not allowed.any():
            return np.empty(0, dtype=np.int64)
        return top_k_indices(rm.mean(axis=0), k, allowed)


class BordaCountAggregator(_ReductionAggregator):
    """BDC -- Borda count.

    Each member ranks the items; per-member ranks (``scipy`` ``method="min"``, so
    the lowest score gets rank 1 and ties share the minimum rank) are summed across
    members and the top scorers are returned. Mirrors ``bdc_algorithm``.
    """

    name = "BDC"

    def _reduce(self, rm):
        # rankdata per member (row); method="min" matches the reference.
        return np.vstack([rankdata(row, method="min") for row in rm]).sum(axis=0)


class FAIAggregator(Aggregator):
    """FAI -- fairness round-robin.

    Members take turns (in index order, starting from ``start``); on each turn the
    current member picks their highest-scoring not-yet-selected item. Deterministic
    port of ``fai_algorithm`` (the reference "fast" variant randomizes the start
    user; we default to a fixed start for reproducibility and accept an rng).
    """

    name = "FAI"

    def __init__(self, start: int = 0, seed: int | None = None) -> None:
        self.start = int(start)
        self.seed = seed

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        n_members, n_items = rm.shape
        avail = available_mask(n_items, exclude)
        budget = int(min(k, int(avail.sum())))
        if budget <= 0:
            return np.empty(0, dtype=np.int64)

        if self.seed is not None:
            user = int(np.random.default_rng(self.seed).integers(0, n_members))
        else:
            user = self.start % n_members

        chosen = np.empty(budget, dtype=np.int64)
        for i in range(budget):
            masked = np.where(avail, rm[user], -np.inf)
            best = int(np.argmax(masked))  # ties -> smallest item index
            chosen[i] = best
            avail[best] = False
            user = (user + 1) % n_members
        return chosen


__all__ = [
    "AdditiveAggregator",
    "AverageAggregator",
    "LeastMiseryAggregator",
    "MultiplicativeAggregator",
    "MostPleasureAggregator",
    "AVGNoMiseryAggregator",
    "BordaCountAggregator",
    "FAIAggregator",
]
