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


class AdditiveAggregator(Aggregator):
    """ADD -- sum of member scores."""

    name = "ADD"

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        return top_k_indices(rm.sum(axis=0), k, avail)


class AverageAggregator(Aggregator):
    """AVG -- mean of member scores."""

    name = "AVG"

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        return top_k_indices(rm.mean(axis=0), k, avail)


class LeastMiseryAggregator(Aggregator):
    """LMS -- minimum (least misery) of member scores."""

    name = "LMS"

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        return top_k_indices(rm.min(axis=0), k, avail)


class MultiplicativeAggregator(Aggregator):
    """MUL -- product of member scores."""

    name = "MUL"

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        return top_k_indices(rm.prod(axis=0), k, avail)


class MostPleasureAggregator(Aggregator):
    """MPL -- maximum (most pleasure) of member scores."""

    name = "MPL"

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        return top_k_indices(rm.max(axis=0), k, avail)


class AVGNoMiseryAggregator(Aggregator):
    """AVGNM -- average, restricted to items whose worst member score exceeds a
    misery threshold.

    Items where ``min_member_score <= threshold`` are dropped; the rest are ranked
    by mean score. Matches ``avgnm_algorithm`` (strict ``> threshold``). The
    reference hard-codes ``threshold=1`` for graded ML ratings; here it is a
    parameter (default ``0.0``).
    """

    name = "AVGNM"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        allowed = (rm.min(axis=0) > self.threshold) & avail
        if not allowed.any():
            return np.empty(0, dtype=np.int64)
        return top_k_indices(rm.mean(axis=0), k, allowed)


class BordaCountAggregator(Aggregator):
    """BDC -- Borda count.

    Each member ranks the items; per-member ranks (``scipy`` ``method="min"``, so
    the lowest score gets rank 1 and ties share the minimum rank) are summed across
    members and the top scorers are returned. Mirrors ``bdc_algorithm``.
    """

    name = "BDC"

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        # rankdata per member (row); method="min" matches the reference.
        borda = np.vstack([rankdata(row, method="min") for row in rm])
        return top_k_indices(borda.sum(axis=0), k, avail)


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
