"""Evaluation: coupled / decoupled protocols + metrics, with fairness-aware
group aggregations.

* **coupled**   -- score the group list against each member's held-out feedback
  (the RS + aggregator are judged as a pair). Works for any recommender.
* **decoupled** -- treat each member's predicted scores ``r̂`` as ground truth,
  isolating the aggregator. Requires a results-first recommender exposing
  ``.base.score(member)``; profile-first / deep group models support coupled only.

:func:`evaluate` runs the single-shot protocol; :func:`evaluate_sequential` runs the
multi-round protocol with the long-term fairness metrics from the KAIS and Stratigi
papers (``dMAE``, ``groupSatO``, ``groupDisO``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ..aggregators.base import SequentialAggregator
from ..data import Dataset, Groups
from ..split import Split
from ._common import (
    Report,
    coupled_ground_truth,
    decoupled_member,
    parse_list_metric,
    parse_metric,
)
from .grouplevel import evaluate_grouplevel
from .metrics import (
    BASE_METRICS,
    GROUP_AGGREGATIONS,
    LIST_METRICS,
    register_aggregation,
    register_list_metric,
    register_metric,
)
from .sampled import evaluate_sampled
from .sequential import evaluate_sequential


def evaluate(
    rec,
    data: Dataset,
    groups: Groups,
    splits,
    *,
    k: int = 10,
    protocol="coupled",
    metrics: Sequence[str] = ("ndcg", "recall", "hr"),
    group_aggregations: Sequence[str] = ("mean", "min", "minmax"),
    binarize: bool = True,
    rating_threshold: float = 4.0,
    exclude_seen: bool = True,
    list_metrics: Sequence[str] = (),
) -> Report:
    """Single-shot evaluation of a group recommender.

    Parameters
    ----------
    rec : group recommender with ``fit(dataset)`` and
        ``recommend(members, k, *, exclude=None) -> item ids``. For ``decoupled`` it
        must also expose ``.base.score(...)``.
    splits : a :class:`~grouprec.split.Split` or list of them (cross-validation).
    k : default cutoff; per-metric override via ``"ndcg@20"``.
    protocol : ``"coupled"``, ``"decoupled"``, or a list of both.
    metrics : base metric names (optionally ``name@k``).
    group_aggregations : how to collapse per-member scores (the fairness lens).
    """
    if isinstance(splits, Split):
        splits = [splits]
    protocols = [protocol] if isinstance(protocol, str) else list(protocol)
    for p in protocols:
        if p not in ("coupled", "decoupled"):
            raise ValueError(f"unknown protocol {p!r}; use 'coupled' and/or 'decoupled'.")
    specs = [parse_metric(m, k) for m in metrics]
    for a in group_aggregations:
        if a not in GROUP_AGGREGATIONS:
            raise ValueError(f"unknown aggregation {a!r}; available: {sorted(GROUP_AGGREGATIONS)}")
    max_k = max(kk for _, kk in specs)

    acc: dict[tuple, list[float]] = defaultdict(list)

    for split in splits:
        rec.fit(split.train)
        train = split.train
        base = getattr(rec, "base", None)
        decoupled_ok = base is not None and hasattr(base, "score")
        if "decoupled" in protocols and not decoupled_ok:
            raise ValueError(
                "decoupled protocol requires a results-first recommender exposing "
                ".base.score(member); profile-first / deep models support 'coupled' only."
            )
        gt = coupled_ground_truth(split.test, binarize, rating_threshold) \
            if "coupled" in protocols else {}
        agg_obj = getattr(rec, "aggregator", None)

        list_ctx = None
        if list_metrics:
            idx = train.interactions["item"].map(train.item_index).to_numpy()
            list_ctx = {"popularity": np.bincount(idx, minlength=train.n_items).astype(float),
                        "item_index": train.item_index, "n_users": train.n_users,
                        "n_items": train.n_items}

        for group in groups:
            members = list(group)
            if isinstance(agg_obj, SequentialAggregator):
                agg_obj.reset()

            exclude = None
            if exclude_seen:
                seen: set = set()
                for u in members:
                    if u in train.user_index:
                        seen.update(int(x) for x in train.items[train.items_seen_by(u)])
                exclude = seen

            rec_items = [int(x) for x in rec.recommend(members, k=max_k, exclude=exclude)]

            for lm in list_metrics:
                name, kk = parse_list_metric(lm, k)
                acc[("-", name, kk, "list")].append(LIST_METRICS[name](rec_items, kk, list_ctx))

            for proto in protocols:
                member_gt = []
                for u in members:
                    if proto == "coupled":
                        g = gt.get(u)
                        if not g or not g["relevant"]:
                            continue
                        member_gt.append((u, g["gains"], g["relevant"], None))
                    else:
                        member_gt.append((u, None, None, np.asarray(base.score([u]), dtype=float)[0]))
                if not member_gt:
                    continue
                for name, kk in specs:
                    fn = BASE_METRICS[name]
                    vals = []
                    for u, gains, relevant, scores_u in member_gt:
                        if proto == "decoupled":
                            gains, relevant = decoupled_member(scores_u, train.items, kk)
                        vals.append(fn(rec_items, gains, relevant, kk))
                    vals = np.asarray(vals, dtype=float)
                    for a in group_aggregations:
                        acc[(proto, name, kk, a)].append(GROUP_AGGREGATIONS[a](vals))

    records = [
        {"protocol": p, "metric": m, "k": kk, "aggregation": a,
         "value": float(np.mean(v)) if v else float("nan")}
        for (p, m, kk, a), v in sorted(acc.items())
    ]
    return Report(records)


__all__ = ["evaluate", "evaluate_sequential", "evaluate_grouplevel", "evaluate_sampled",
           "Report", "BASE_METRICS", "GROUP_AGGREGATIONS", "LIST_METRICS",
           "register_metric", "register_aggregation", "register_list_metric"]
