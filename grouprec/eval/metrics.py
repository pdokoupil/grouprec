"""Per-member ranking metrics and group-level aggregations.

A **base metric** scores one member's view of a recommendation list, given that
member's relevant-item set and per-item gains. A **group aggregation** collapses the
vector of per-member scores into one group number -- and *this* is the fairness
lens: ``mean`` is social welfare, ``min`` is least misery, ``minmax`` is the
min/max ratio (balance), ``jain`` is Jain's index, ``zero`` is the fraction of
members served nothing (zero-recall when applied to recall).

Base metrics are ported to match ``gmap2023/evaluation_metrics`` and the
nDCG/DCG/min-max conventions in the KAIS paper.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# base (per-member) metrics: f(rec_items, gains, relevant, k) -> float
# --------------------------------------------------------------------------- #
def _dcg(rec_items, gains, k) -> float:
    return float(sum(gains.get(it, 0.0) / np.log2(r + 2) for r, it in enumerate(rec_items[:k])))


def _idcg(gains, k) -> float:
    top = sorted(gains.values(), reverse=True)[:k]
    return float(sum(g / np.log2(r + 2) for r, g in enumerate(top)))


def ndcg(rec_items, gains, relevant, k) -> float:
    idcg = _idcg(gains, k)
    return _dcg(rec_items, gains, k) / idcg if idcg > 0 else 0.0


def dcg(rec_items, gains, relevant, k) -> float:
    return _dcg(rec_items, gains, k)


def recall(rec_items, gains, relevant, k) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for it in rec_items[:k] if it in relevant)
    return hits / len(relevant)


def bounded_recall(rec_items, gains, relevant, k) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for it in rec_items[:k] if it in relevant)
    return hits / min(len(relevant), k)


def precision(rec_items, gains, relevant, k) -> float:
    hits = sum(1 for it in rec_items[:k] if it in relevant)
    return hits / k


def hr(rec_items, gains, relevant, k) -> float:
    return 1.0 if any(it in relevant for it in rec_items[:k]) else 0.0


def dfh(rec_items, gains, relevant, k) -> float:
    """Discounted first hit: 1/log2(rank+2) of the first relevant item, else 0."""
    for r, it in enumerate(rec_items[:k]):
        if it in relevant:
            return 1.0 / np.log2(r + 2)
    return 0.0


def mrr(rec_items, gains, relevant, k) -> float:
    """Mean of the reciprocal ranks of *all* relevant hits (matches gmap2023)."""
    rr = [1.0 / (r + 1) for r, it in enumerate(rec_items[:k]) if it in relevant]
    return float(np.mean(rr)) if rr else 0.0


def ar(rec_items, gains, relevant, k) -> float:
    """Average relevance of the recommended items -- mean over the top-k list of the
    item relevances (feedback under coupled, predicted ``r̂`` under decoupled)."""
    window = rec_items[:k]
    if not window:
        return 0.0
    return float(sum(gains.get(it, 0.0) for it in window) / len(window))


BASE_METRICS: dict[str, Callable] = {
    "ndcg": ndcg,
    "dcg": dcg,
    "ar": ar,
    "recall": recall,
    "brecall": bounded_recall,
    "precision": precision,
    "hr": hr,
    "dfh": dfh,
    "mrr": mrr,
}


# --------------------------------------------------------------------------- #
# group aggregations: g(values: np.ndarray) -> float
# --------------------------------------------------------------------------- #
def _minmax(v: np.ndarray) -> float:
    mx = v.max()
    return float(v.min() / mx) if mx > 0 else 0.0


def _jain(v: np.ndarray) -> float:
    s2 = float((v ** 2).sum())
    return float((v.sum() ** 2) / (len(v) * s2)) if s2 > 0 else 0.0


GROUP_AGGREGATIONS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda v: float(v.mean()),
    "min": lambda v: float(v.min()),
    "max": lambda v: float(v.max()),
    "minmax": _minmax,
    "std": lambda v: float(v.std()),
    "jain": _jain,
    "zero": lambda v: float(np.mean(v == 0)),
}


# --------------------------------------------------------------------------- #
# list-level metrics: f(rec_items, k, ctx) -> float  (computed once per group list)
# --------------------------------------------------------------------------- #
# Metrics live at three levels: per-member (relevance, aggregated), per-**list**
# (diversity/novelty/coverage — no member ground truth needed), and per-run (carbon,
# latency — see grouprec.profiling). This registry is the per-list extension point.
def _novelty(rec_items, k, ctx):
    pop, n_users = ctx["popularity"], max(ctx["n_users"], 1)
    idx = ctx["item_index"]
    vals = [(-np.log2((pop[idx[i]] + 1) / (n_users + 1))) for i in rec_items[:k] if i in idx]
    return float(np.mean(vals)) if vals else 0.0


def _list_coverage(rec_items, k, ctx):
    window = rec_items[:k]
    return len(set(window)) / len(window) if window else 0.0


LIST_METRICS: dict[str, Callable] = {"novelty": _novelty, "list_coverage": _list_coverage}


def register_list_metric(name: str, fn) -> None:
    """Register a custom per-**list** metric ``fn(rec_items, k, ctx)`` where ``ctx``
    has ``popularity`` (train counts), ``item_index``, ``n_users``, ``n_items``.
    For diversity/novelty/coverage that depend on the list, not per-member relevance."""
    LIST_METRICS[name.lower()] = fn


def register_metric(name: str, fn) -> None:
    """Register a custom per-member base metric ``fn(rec_items, gains, relevant, k)``.

    Example (intra-list novelty proxy)::

        gr.eval.register_metric("listlen", lambda rec, g, rel, k: len(rec[:k]) / k)
        evaluate(rec, ..., metrics=["listlen"])
    """
    BASE_METRICS[name.lower()] = fn


def register_aggregation(name: str, fn) -> None:
    """Register a custom group aggregation ``fn(values: np.ndarray) -> float``."""
    GROUP_AGGREGATIONS[name] = fn


__all__ = ["BASE_METRICS", "GROUP_AGGREGATIONS", "LIST_METRICS",
           "register_metric", "register_aggregation", "register_list_metric"]
