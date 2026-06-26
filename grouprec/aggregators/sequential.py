"""Sequential / stateful group aggregators.

Contains the RLProp mandate-allocation core (``rlprop_step``), the single-session
``RLPropAggregator``, the cross-session ``LTPAggregator`` (KAIS Algorithm 1),
``PeriodicFAIAggregator``, the sequential ``EPFuzzDAWeightedAggregator``, and the
satisfaction-driven ``SDAAAggregator`` / ``SIAAAggregator`` (Stratigi et al.).

(PAR is *not* sequential -- it is Xiao's single-shot greedy and lives in
``fairness.py``.)

References
----------
LTP / RLProp : Dokoupil & Peska, *Long-term fairness in sequential group
    recommendations*, KIS (2026). https://doi.org/10.1007/s10115-025-02642-9
SDAA / SIAA  : Stratigi et al., *Sequential group recommendations based on
    satisfaction and disagreement scores*, JIIS (2022).
    doi:10.1007/s10844-021-00652-x
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ._normalize import normalize_mgains
from .base import (
    Aggregator,
    SequentialAggregator,
    as_score_matrix,
    available_mask,
    top_k_indices,
)
from .fairness import epfuzzda_select


def rlprop_step(
    rm: np.ndarray,
    gm: np.ndarray,
    tot: float,
    weights: np.ndarray,
    available: np.ndarray,
    tie_breaking: float = 0.0,
) -> int:
    """Select one item via RLProp / LTP mandate allocation (KAIS Algorithm 1, l.6-12).

    Parameters
    ----------
    rm : (n_members, n_items) score matrix (already normalized if desired).
    gm : (n_members,) current per-member accumulated utility *state* ``s`` (already
        averaged over sessions where applicable).
    tot : scalar ``T`` -- the (clipped) sum of ``gm``.
    weights : (n_members,) member importance weights ``w``.
    available : (n_items,) boolean mask of selectable items.
    tie_breaking : ``beta`` in the paper -- contribution of residual utility beyond
        the proportional cap (``0`` ⇒ EP-FuzzDA behaviour).

    Returns
    -------
    int : index of the selected item.
    """
    # T_c = max(T, T + sum_u R_{u,c})  (line 7)
    tots = np.maximum(tot, tot + rm.sum(axis=0))                 # (n_items,)
    remainder = tots * weights[:, None] - gm[:, None]           # (m, n)  line 8

    gain = np.zeros_like(rm)
    pos = rm >= 0.0
    neg = ~pos
    # line 9
    gain[pos] = np.maximum(0.0, np.minimum(rm, remainder)[pos])
    gain[neg] = np.minimum(0.0, (rm - remainder)[neg])

    # line 10: residual-utility (tie-breaking) penalty where the cap binds
    if tie_breaking != 0.0:
        tie = remainder <= rm
        gain[tie] = gain[tie] + tie_breaking * (rm - remainder)[tie]

    scores = gain.sum(axis=0)
    scores = np.where(available, scores, -np.inf)
    return int(np.argmax(scores))


class RLPropAggregator(Aggregator):
    """RLProp -- single-session proportional aggregation with negative-preference
    handling (EP-FuzzDA generalized).

    Stateless across calls. ``with_norm`` applies per-member quantile/CDF
    normalization before aggregation.
    """

    name = "RLProp"

    def __init__(self, with_norm: bool = False) -> None:
        self.with_norm = bool(with_norm)

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        if self.with_norm:
            rm = normalize_mgains(rm, "quantile")
        weights = np.full(m, 1.0 / m)
        avail = available_mask(n, exclude)
        budget = int(min(k, int(avail.sum())))
        gm = np.zeros(m)
        tot = 0.0
        chosen = np.empty(budget, dtype=np.int64)
        for i in range(budget):
            best = rlprop_step(rm, gm, tot, weights, avail, tie_breaking=0.0)
            avail[best] = False
            gm = gm + rm[:, best]
            tot = float(np.clip(gm, 0.0, None).sum())
            chosen[i] = best
        return chosen


class LTPAggregator(SequentialAggregator):
    """LTP -- Long-Term Proportionality (KAIS Algorithm 1).

    Create one per group; call :meth:`aggregate` once per session (per-member
    fairness state ``s`` carries across calls). :meth:`reset` starts a new group.

    Parameters mirror ``ltp_prospective.py`` so this is a drop-in for the external
    ``ltp`` package:

    gamma : carry-over of accumulated gains between sessions (``1.0`` ⇒ paper's pure
        accumulation; ``<1`` forgets older sessions).
    beta : in-list positional factor (paper's ``alpha``); item at position ``p``
        updates the state with weight ``exp(-beta * p)``.
    tie_breaking : paper's ``beta`` residual-utility penalty (``0`` ⇒ EP-FuzzDA).
    normalize : per-member normalization each session (see :mod:`._normalize`).
    member_weights : relative member importance ``w`` (default uniform); normalized
        to sum to 1.
    avoid_repeats : if ``True`` (default), items recommended in earlier sessions of
        this group are not recommended again.

    With ``gamma=1, beta=0, tie_breaking=0, normalize=None`` and a :meth:`reset`
    before each call this reproduces single-session :class:`RLPropAggregator`.
    """

    name = "LTP"

    def __init__(
        self,
        gamma: float = 1.0,
        beta: float = 0.0,
        tie_breaking: float = 0.0,
        normalize: str | None = None,
        member_weights: Sequence[float] | None = None,
        avoid_repeats: bool = True,
    ) -> None:
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.tie_breaking = float(tie_breaking)
        self.normalize = normalize
        self._member_weights = (
            None if member_weights is None else np.asarray(member_weights, dtype=float)
        )
        self.avoid_repeats = bool(avoid_repeats)
        self.reset()

    def reset(self) -> "LTPAggregator":
        self.gm: np.ndarray | None = None
        self.n_sessions = 0
        self._recommended: set[int] = set()
        return self

    def _weights(self, m: int) -> np.ndarray:
        if self._member_weights is None:
            return np.full(m, 1.0 / m)
        w = self._member_weights
        if w.size != m:
            raise ValueError(f"member_weights has {w.size} entries but group has {m} members.")
        s = w.sum()
        if s <= 0:
            raise ValueError("member_weights must sum to a positive value.")
        return w / s

    def aggregate(self, scores, k, *, exclude: Iterable[int] | None = None):
        rm = as_score_matrix(scores)
        m, n = rm.shape

        if self.gm is None:
            self.gm = np.zeros(m)
        elif self.gm.shape[0] != m:
            raise ValueError("group size changed between sessions; call reset() first.")

        weights = self._weights(m)
        if self.normalize:
            rm = normalize_mgains(rm, self.normalize)

        avail = available_mask(n, exclude)
        if self.avoid_repeats and self._recommended:
            past = [i for i in self._recommended if i < n]
            if past:
                avail[past] = False

        sessions = max(self.n_sessions, 1)
        gm_norm = self.gm / sessions
        tot = float(gm_norm.sum())  # initial T unclipped (matches reference code)

        budget = int(min(k, int(avail.sum())))
        chosen = np.empty(budget, dtype=np.int64)
        for position in range(budget):
            best = rlprop_step(rm, gm_norm, tot, weights, avail, self.tie_breaking)
            avail[best] = False
            position_factor = float(np.exp(-self.beta * position))
            self.gm = self.gamma * self.gm + position_factor * rm[:, best]
            gm_norm = self.gm / sessions
            tot = float(np.clip(gm_norm, 0.0, None).sum())
            chosen[position] = best

        self.n_sessions += 1
        if self.avoid_repeats:
            self._recommended.update(int(x) for x in chosen)
        return chosen


class PeriodicFAIAggregator(SequentialAggregator):
    """PeriodicFAI -- FAI round-robin whose member pointer persists across sessions.

    The starting member of each session continues the global round-robin from where
    the previous session ended (``total_items_so_far % n_members``), so over many
    sessions every member leads equally often.
    """

    name = "PeriodicFAI"

    def __init__(self, avoid_repeats: bool = False) -> None:
        self.avoid_repeats = bool(avoid_repeats)
        self.reset()

    def reset(self) -> "PeriodicFAIAggregator":
        self._n_recommended = 0
        self._recommended: set[int] = set()
        return self

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        avail = available_mask(n, exclude)
        if self.avoid_repeats and self._recommended:
            past = [i for i in self._recommended if i < n]
            if past:
                avail[past] = False

        budget = int(min(k, int(avail.sum())))
        user = self._n_recommended % m
        chosen = np.empty(budget, dtype=np.int64)
        for i in range(budget):
            masked = np.where(avail, rm[user], -np.inf)
            best = int(np.argmax(masked))
            chosen[i] = best
            avail[best] = False
            user = (user + 1) % m

        self._n_recommended += budget
        if self.avoid_repeats:
            self._recommended.update(int(x) for x in chosen)
        return chosen


class EPFuzzDAWeightedAggregator(SequentialAggregator):
    """EPFuzzDAWeighted -- sequential EP-FuzzDA that steers member weights toward
    members under-served in previous sessions.

    Port of ``EPFuzzDAWeightedAggregator``. Each session, member weights are set to
    ``max(0, target_share - cumulative_gain_share)``; ``use_all_iter_weights``
    selects whether the target share grows with the session count or stays uniform.
    """

    name = "EPFuzzDAWeighted"

    def __init__(self, use_all_iter_weights: bool = False) -> None:
        self.use_all_iter_weights = bool(use_all_iter_weights)
        self.reset()

    def reset(self) -> "EPFuzzDAWeightedAggregator":
        self.gains_so_far: np.ndarray | None = None
        self.gains_last_iter: np.ndarray | None = None
        self.iter = 0
        return self

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        avail = available_mask(n, exclude)

        if self.gains_so_far is None:
            self.gains_so_far = np.zeros(m)
            self.gains_last_iter = np.zeros(m)

        if self.iter == 0:
            weights = np.full(m, 1.0 / m)
        else:
            if self.use_all_iter_weights:
                should_receive = np.full(m, self.iter + 1.0) / m
            else:
                should_receive = np.ones(m) / m
            weights = np.maximum(0.0, should_receive - self.gains_so_far)

        chosen, awarded = epfuzzda_select(rm, k, weights, avail)

        total = float(awarded.sum())
        share = awarded / total if total > 0 else np.zeros(m)
        self.gains_last_iter = share
        self.gains_so_far = self.gains_so_far + share
        self.iter += 1
        return chosen


# --------------------------------------------------------------------------- #
# SDAA / SIAA -- satisfaction-driven sequential aggregation (Stratigi et al.)
# --------------------------------------------------------------------------- #
def _satisfaction(rm: np.ndarray, selected: np.ndarray, k: int) -> np.ndarray:
    """Per-member satisfaction for a recommendation list (Stratigi Eq. 1).

    ``sat(u) = sum_{d in Gr} p(u, d) / sum_{d in top-k(u)} p(u, d)`` -- the share of
    the member's *ideal* top-k utility that the produced list captures. Uses each
    member's own predicted scores (not the aggregated ones), in ``[0, 1]``.
    """
    m, n = rm.shape
    kk = min(n, k)
    ideal = np.sort(rm, axis=1)[:, -kk:].sum(axis=1)
    selected = np.asarray(selected, dtype=int)
    got = rm[:, selected].sum(axis=1) if selected.size else np.zeros(m)
    return np.divide(got, ideal, out=np.zeros(m), where=ideal > 0)


class SDAAAggregator(SequentialAggregator):
    """SDAA -- Sequential Dynamic Adaptation Aggregation (Stratigi et al., 2022).

    A *group-level* dynamic blend of Average and (modified) Least Misery:
    ``score(d) = (1 - a) * avg(d) + a * least(d)`` (Eq. 8), where ``avg`` is the mean
    member score for the item and ``least`` is the previous round's least-satisfied
    member's score. The weight ``a`` self-regulates as the previous round's group
    disagreement ``max_u sat(u) - min_u sat(u)`` (Eq. 9); the first round (``a=0``)
    is plain Average.

    Parameters
    ----------
    avoid_repeats : exclude items recommended in earlier rounds (default False, to
        match the paper's pseudocode; enable for realistic consumption settings).
    """

    name = "SDAA"

    def __init__(self, avoid_repeats: bool = False) -> None:
        self.avoid_repeats = bool(avoid_repeats)
        self.reset()

    def reset(self) -> "SDAAAggregator":
        self._sat_prev: np.ndarray | None = None
        self._recommended: set[int] = set()
        return self

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        avail = available_mask(n, exclude)
        if self.avoid_repeats and self._recommended:
            past = [i for i in self._recommended if i < n]
            if past:
                avail[past] = False

        avg = rm.mean(axis=0)
        if self._sat_prev is None:
            score = avg  # first round: alpha = 0 -> Average
        else:
            alpha = float(self._sat_prev.max() - self._sat_prev.min())
            least_user = int(np.argmin(self._sat_prev))
            score = (1.0 - alpha) * avg + alpha * rm[least_user]

        chosen = top_k_indices(score, k, avail)
        self._sat_prev = _satisfaction(rm, chosen, k)
        if self.avoid_repeats:
            self._recommended.update(int(x) for x in chosen)
        return chosen


class SIAAAggregator(SequentialAggregator):
    """SIAA -- Sequential Individual Adaptation Aggregation (Stratigi et al., 2022).

    A *member-level* weighted aggregation: ``score(d) = sum_u w(u) * p(u, d)``
    (Eq. 10), with per-member weight (Eq. 12)
    ``w(u) = (1 - b) * (1 - satO(u)) + b * userDis(u)`` where ``satO`` is the member's
    overall (mean) satisfaction over all previous rounds and ``userDis`` is the
    previous round's user disagreement ``max_l sat(l) - sat(u)`` (Eq. 7). The first
    round reduces to Average.

    Parameters
    ----------
    b : balance between overall-satisfaction and last-round-disagreement terms.
    avoid_repeats : see :class:`SDAAAggregator`.
    """

    name = "SIAA"

    def __init__(self, b: float = 0.5, avoid_repeats: bool = False) -> None:
        self.b = float(b)
        self.avoid_repeats = bool(avoid_repeats)
        self.reset()

    def reset(self) -> "SIAAAggregator":
        self._satO_sum: np.ndarray | None = None
        self._rounds = 0
        self._sat_prev: np.ndarray | None = None
        self._recommended: set[int] = set()
        return self

    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        m, n = rm.shape
        avail = available_mask(n, exclude)
        if self.avoid_repeats and self._recommended:
            past = [i for i in self._recommended if i < n]
            if past:
                avail[past] = False

        if self._rounds == 0:
            w = np.full(m, 1.0 - self.b)  # satO=0, userDis=0 -> uniform -> Average
        else:
            satO = self._satO_sum / self._rounds
            user_dis = self._sat_prev.max() - self._sat_prev
            w = (1.0 - self.b) * (1.0 - satO) + self.b * user_dis

        score = w @ rm
        chosen = top_k_indices(score, k, avail)

        sat_now = _satisfaction(rm, chosen, k)
        self._satO_sum = sat_now if self._satO_sum is None else self._satO_sum + sat_now
        self._rounds += 1
        self._sat_prev = sat_now
        if self.avoid_repeats:
            self._recommended.update(int(x) for x in chosen)
        return chosen


__all__ = [
    "rlprop_step",
    "RLPropAggregator",
    "LTPAggregator",
    "PeriodicFAIAggregator",
    "EPFuzzDAWeightedAggregator",
    "SDAAAggregator",
    "SIAAAggregator",
]
