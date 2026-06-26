"""Fairness / list-wise aggregators (numpy/scipy only, core install).

Contains:

* EP-FuzzDA (``epfd``) -- proportional-utility predecessor of RLProp/LTP.
* GFAR -- ranking-aware fairness via Borda relevance probabilities.
* The Xiao (2017) greedy scalarization framework, exposed as GreedyLM (=GLM,
  least-misery fairness) and PAR (variance fairness) -- the *same* greedy
  algorithm differing only in the fairness term.
* SPGreedy -- proportionality set-cover greedy (Serbos et al., 2017).

References
----------
EP-FuzzDA  : https://dl.acm.org/doi/10.1145/3450614.3461679
GFAR       : Kaya, Bridge, Tintarev, *Ensuring Fairness in Group
             Recommendations...*, RecSys 2020. doi:10.1145/3383313.3412232
Greedy/GLM/PAR : Xiao et al., *Fairness-Aware Group Recommendation with
             Pareto-Efficiency*, RecSys 2017. doi:10.1145/3109859.3109887
SPGreedy   : Serbos et al., *Fairness in Package-to-Group Recommendations*,
             WWW 2017. doi:10.1145/3038912.3052612
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata

from .base import Aggregator, as_score_matrix, available_mask


# --------------------------------------------------------------------------- #
# EP-FuzzDA
# --------------------------------------------------------------------------- #
def _member_weights(member_weights: Sequence[float] | None, m: int) -> np.ndarray:
    """Uniform ``1/m`` by default; otherwise the given weights, used as-is.

    Mirrors the reference, which does **not** renormalize caller-supplied weights
    (the sequential EPFuzzDAWeighted relies on this to pass un-normalized deltas).
    """
    if member_weights is None:
        return np.full(m, 1.0 / m)
    w = np.asarray(member_weights, dtype=float)
    if w.size != m:
        raise ValueError(f"member_weights has {w.size} entries but group has {m} members.")
    return w


def epfuzzda_select(
    rm: np.ndarray,
    k: int,
    weights: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Core EP-FuzzDA selection; returns (chosen indices, per-member awarded utility)."""
    m, n = rm.shape
    sum_util = rm.sum(axis=0)
    awarded = np.zeros(m)
    total_awarded = 0.0
    avail = available.copy()
    budget = int(min(k, int(avail.sum())))
    chosen = np.empty(budget, dtype=np.int64)

    for i in range(budget):
        prospected = sum_util + total_awarded
        allowed = np.outer(weights, prospected)
        unfulfilled = np.maximum(0.0, allowed - awarded[:, None])
        relevance = np.minimum(unfulfilled, rm).sum(axis=0)
        relevance = np.where(avail, relevance, -np.inf)
        best = int(np.argmax(relevance))
        chosen[i] = best
        avail[best] = False
        winner = rm[:, best]
        awarded = awarded + winner
        total_awarded += float(winner.sum())

    return chosen, awarded


class EPFuzzDAAggregator(Aggregator):
    """EP-FuzzDA -- exponentially-proportional fuzzy D'Hondt aggregation."""

    name = "EPFuzzDA"

    def __init__(self, member_weights: Sequence[float] | None = None) -> None:
        self._member_weights = member_weights

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        weights = _member_weights(self._member_weights, rm.shape[0])
        avail = available_mask(rm.shape[1], exclude)
        chosen, _ = epfuzzda_select(rm, k, weights, avail)
        return chosen


# --------------------------------------------------------------------------- #
# GFAR
# --------------------------------------------------------------------------- #
class GFARAggregator(Aggregator):
    """GFAR -- Greedy For Average Relevance / ranking-aware group fairness.

    For each member, item relevance probability ``p(rel|u,i)`` is the Borda
    relevance of ``i`` within the member's top-``N`` items, normalized to sum to 1
    (RecSys'20, Eq. 1). The list is built greedily, each step adding the item with
    the largest marginal gain ``sum_u p(rel|u,i) * prod_{j in S}(1-p(rel|u,j))``
    (Eq. 5) -- the probability that the item gives each member their first relevant
    hit.

    Parameters
    ----------
    relevant_max_items : ``N`` -- size of each member's top-N considered relevant
        (items outside it get zero relevance). Default 20 (matches the reference).
    """

    name = "GFAR"

    def __init__(self, relevant_max_items: int = 20) -> None:
        self.relevant_max_items = int(relevant_max_items)

    def _p_relevant(self, rm: np.ndarray) -> np.ndarray:
        m, n = rm.shape
        N = min(self.relevant_max_items, n)
        p_rel = np.zeros((m, n))
        for u in range(m):
            top = np.argsort(-rm[u], kind="stable")[:N]
            # Borda-rel within top-N: rank by score (ties share the max rank, as in
            # the reference scipy rankdata(method="max")), minus 1 so best == N-1.
            borda = rankdata(rm[u, top], method="max") - 1.0
            s = borda.sum()
            if s > 0:
                p_rel[u, top] = borda / s
        return p_rel

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        p_rel = self._p_relevant(rm)
        avail = available_mask(n, exclude)
        prob_not_rel = np.ones(m)  # prob none of the selected items is relevant, per member
        budget = int(min(k, int(avail.sum())))
        chosen = np.empty(budget, dtype=np.int64)
        for i in range(budget):
            marginal = (p_rel * prob_not_rel[:, None]).sum(axis=0)
            marginal = np.where(avail, marginal, -np.inf)
            best = int(np.argmax(marginal))
            chosen[i] = best
            avail[best] = False
            prob_not_rel = prob_not_rel * (1.0 - p_rel[:, best])
        return chosen


# --------------------------------------------------------------------------- #
# Xiao (2017) greedy scalarization: GreedyLM (GLM) + PAR
# --------------------------------------------------------------------------- #
def _individual_utility_denominator(rm: np.ndarray, k: int, utility: str) -> np.ndarray:
    """Per-member denominator for the individual utility ``U(u, L)``.

    ``proportional`` -> sum of each member's top-``k`` scores (so ``U`` is the share
    of a member's ideal utility captured by the list); ``average`` -> ones.
    """
    if utility == "average":
        return np.ones(rm.shape[0])
    if utility == "proportional":
        kk = min(rm.shape[1], k)
        denom = np.sort(rm, axis=1)[:, -kk:].sum(axis=1)
        denom[denom == 0] = 1.0
        return denom
    raise ValueError(f"unknown utility {utility!r}; use 'proportional' or 'average'.")


class GreedyScalarizationAggregator(Aggregator):
    """Xiao (2017) greedy fairness-aware aggregation (Algorithm 1).

    Builds the list incrementally; each step adds the item maximizing
    ``lam * SW(L u {i}) + (1 - lam) * F(L u {i})`` where ``SW`` is the mean member
    utility and ``F`` is one of the fairness semantics below, all computed on the
    *cumulative* per-member utilities.

    Parameters
    ----------
    fairness : ``"least_misery"`` (min U), ``"variance"`` (1 - Var U),
        ``"minmax"`` (min/max), or ``"jain"`` (Jain's index).
    lam : scalarization weight ``lambda`` on social welfare.
    utility : ``"proportional"`` (default, Xiao's headline) or ``"average"``.
    """

    def __init__(self, fairness: str = "least_misery", lam: float = 0.5,
                 utility: str = "proportional") -> None:
        if fairness not in {"least_misery", "variance", "minmax", "jain"}:
            raise ValueError(f"unknown fairness semantic {fairness!r}")
        self.fairness = fairness
        self.lam = float(lam)
        self.utility = utility

    name = "Greedy"

    def _fairness(self, cand: np.ndarray) -> np.ndarray:
        # cand: (n_members, n_candidates) cumulative utilities if each candidate added.
        if self.fairness == "least_misery":
            return cand.min(axis=0)
        if self.fairness == "variance":
            return 1.0 - cand.var(axis=0)
        if self.fairness == "minmax":
            mx = cand.max(axis=0)
            mn = cand.min(axis=0)
            return np.divide(mn, mx, out=np.zeros_like(mn), where=mx != 0)
        # jain
        s1 = cand.sum(axis=0)
        s2 = (cand ** 2).sum(axis=0)
        m = cand.shape[0]
        return np.divide(s1 ** 2, m * s2, out=np.zeros_like(s1), where=s2 != 0)

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        denom = _individual_utility_denominator(rm, k, self.utility)
        avail = available_mask(n, exclude)
        budget = int(min(k, int(avail.sum())))
        cum = np.zeros(m)  # cumulative numerator (sum of selected scores) per member
        chosen = np.empty(budget, dtype=np.int64)
        for i in range(budget):
            cand = (cum[:, None] + rm) / denom[:, None]   # U(u, L u {i})  (m, n)
            sw = cand.mean(axis=0)
            f = self._fairness(cand)
            score = self.lam * sw + (1.0 - self.lam) * f
            score = np.where(avail, score, -np.inf)
            best = int(np.argmax(score))
            chosen[i] = best
            avail[best] = False
            cum = cum + rm[:, best]
        return chosen


class GreedyLMAggregator(GreedyScalarizationAggregator):
    """GreedyLM (GLM) -- Xiao greedy with least-misery fairness, proportional
    utility, ``lambda = 0.5``."""

    name = "GreedyLM"

    def __init__(self, lam: float = 0.5) -> None:
        super().__init__(fairness="least_misery", lam=lam, utility="proportional")


class PARAggregator(GreedyScalarizationAggregator):
    """PAR -- Xiao greedy with **variance** fairness, proportional utility,
    ``lambda = 0.8`` (per s10844-021-00652-x).

    NOTE: this is the faithful greedy form. The earlier ``aggregators_new.py`` PAR
    was a one-shot per-item ``mean + (1-var)`` score (no greedy, no proportional
    utility, lambda=0.5) and does not match the source.
    """

    name = "PAR"

    def __init__(self, lam: float = 0.8) -> None:
        super().__init__(fairness="variance", lam=lam, utility="proportional")


# --------------------------------------------------------------------------- #
# SPGreedy (proportionality set-cover greedy)
# --------------------------------------------------------------------------- #
class SPGreedyAggregator(Aggregator):
    """SPGreedy -- single-proportionality set-cover greedy (Serbos et al., 2017).

    A member "likes" an item if it falls in the top-``delta`` fraction of that
    member's preferences. A member is *covered* once the list contains >=1 item
    they like (single proportionality, m=1). Each step adds the item covering the
    most not-yet-covered members; once coverage saturates, ties are broken by group
    mean relevance (then smallest index).

    Parameters
    ----------
    delta : fraction of each member's items considered "liked" (default 0.1 = top 10%).
    """

    name = "SPGreedy"

    def __init__(self, delta: float = 0.1) -> None:
        if not 0 < delta <= 1:
            raise ValueError("delta must be in (0, 1].")
        self.delta = float(delta)

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        n_like = max(1, int(round(self.delta * n)))
        likes = np.zeros((m, n), dtype=bool)
        for u in range(m):
            top = np.argsort(-rm[u], kind="stable")[:n_like]
            likes[u, top] = True

        mean_rel = rm.mean(axis=0)
        avail = available_mask(n, exclude)
        covered = np.zeros(m, dtype=bool)
        budget = int(min(k, int(avail.sum())))
        chosen = np.empty(budget, dtype=np.int64)
        for i in range(budget):
            gain = (likes & (~covered)[:, None]).sum(axis=0).astype(float)
            gain = np.where(avail, gain, -1.0)
            maxg = gain.max()
            cand = np.flatnonzero(gain == maxg)
            # tie-break: highest group mean relevance, then smallest index
            best = int(cand[np.argmax(mean_rel[cand])])
            chosen[i] = best
            avail[best] = False
            covered |= likes[:, best]
        return chosen


__all__ = [
    "EPFuzzDAAggregator",
    "epfuzzda_select",
    "GFARAggregator",
    "GreedyScalarizationAggregator",
    "GreedyLMAggregator",
    "PARAggregator",
    "SPGreedyAggregator",
]
