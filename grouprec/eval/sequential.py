"""Sequential (multi-round) evaluation with long-term fairness metrics.

A group receives a fresh top-``k`` list each round; already-recommended items are
excluded across rounds, and stateful aggregators carry their fairness state. Beyond
per-session metrics (averaged over rounds), this reports the long-term metrics:

* **dMAE@k** (KAIS, Eq. 2) -- discounted Mean Absolute Error: how far each member's
  *cumulative discounted utility* share ``U_u / sum_u U`` deviates from the equal
  share ``1/|G|``. Lower is fairer. ``U_u`` uses the same ``log2(i+1)`` discount as nDCG.
* **groupSatO** (Stratigi, Eq. 4) -- overall group satisfaction: mean over members of
  ``satO``, where ``sat`` (Eq. 1) is the share of a member's ideal top-k utility a list
  captures and ``satO`` averages it over rounds. Higher is better.
* **groupDisO** (Stratigi, Eq. 6) -- overall group disagreement: ``max_u satO - min_u satO``.
  Lower is fairer.

Relevances come from held-out feedback (coupled) or predicted ``r̂`` (decoupled).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ..aggregators.base import SequentialAggregator
from ..data import Dataset, Groups
from ..split import Split
from ._common import Report, coupled_ground_truth, decoupled_member, parse_metric
from .metrics import BASE_METRICS, GROUP_AGGREGATIONS

LONG_TERM_METRICS = ("dMAE", "groupSatO", "groupDisO")


def _discounted(rels) -> float:
    return float(sum(max(r, 0.0) / np.log2(i + 2) for i, r in enumerate(rels)))


def _ideal_discounted(rel_values, k) -> float:
    return _discounted(sorted((max(r, 0.0) for r in rel_values), reverse=True)[:k])


def _ideal_sum(rel_values, k) -> float:
    return float(sum(sorted((max(r, 0.0) for r in rel_values), reverse=True)[:k]))


def evaluate_sequential(
    rec,
    data: Dataset,
    groups: Groups,
    splits,
    *,
    n_rounds: int = 5,
    k: int = 10,
    protocol="coupled",
    metrics: Sequence[str] = ("ndcg", "ar"),
    group_aggregations: Sequence[str] = ("mean", "min", "minmax"),
    long_term_metrics: Sequence[str] = LONG_TERM_METRICS,
    binarize: bool = True,
    rating_threshold: float = 4.0,
    exclude_seen: bool = True,
) -> Report:
    """Run ``n_rounds`` sessions per group and report per-session + long-term metrics.

    Same recommender contract as :func:`~grouprec.evaluate`. Stateful aggregators are
    reset at the start of each group; recommendation lists are generated **once** per
    group (so the aggregator advances once) and then scored under each protocol.
    Long-term metrics are group-level (reported with aggregation label ``"group"``).
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
    for ltm in long_term_metrics:
        if ltm not in LONG_TERM_METRICS:
            raise ValueError(f"unknown long-term metric {ltm!r}; available: {list(LONG_TERM_METRICS)}")

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

        for group in groups:
            members = list(group)

            # -- generate the round lists ONCE (recommendation is protocol-agnostic) --
            if isinstance(agg_obj, SequentialAggregator):
                agg_obj.reset()
            seen_ids: set = set()
            if exclude_seen:
                for u in members:
                    if u in train.user_index:
                        seen_ids.update(int(x) for x in train.items[train.items_seen_by(u)])
            recommended: set = set()
            round_lists: list[list[int]] = []
            for _ in range(n_rounds):
                lst = [int(x) for x in rec.recommend(members, k=k, exclude=seen_ids | recommended)]
                if not lst:
                    break
                recommended.update(lst)
                round_lists.append(lst)
            if not round_lists:
                continue

            for proto in protocols:
                _score_group(proto, members, round_lists, gt, base, train, specs,
                             group_aggregations, long_term_metrics, k, acc)

    records = [
        {"protocol": p, "metric": m, "k": kk, "aggregation": a,
         "value": float(np.mean(v)) if v else float("nan")}
        for (p, m, kk, a), v in sorted(acc.items())
    ]
    return Report(records)


def _score_group(proto, members, round_lists, gt, base, train, specs,
                 group_aggregations, long_term_metrics, k, acc):
    """Compute per-session + long-term metrics for one group under one protocol."""
    # Build per-member relevance accessors and ideal references.
    rel_of: list = []          # rel_of[i] = function(item_id) -> relevance
    ideal_dcg: list = []
    ideal_sum: list = []
    decoupled_scores: list = []  # for per-session decoupled base metrics
    eval_members = []
    for u in members:
        if proto == "coupled":
            g = gt.get(u)
            if not g or not g["relevant"]:
                continue
            gains = g["gains"]
            rel_of.append(lambda it, _g=gains: _g.get(it, 0.0))
            ideal_dcg.append(_ideal_discounted(gains.values(), k))
            ideal_sum.append(_ideal_sum(gains.values(), k))
            decoupled_scores.append(None)
        else:
            su = np.asarray(base.score([u]), dtype=float)[0]
            idx = train.item_index
            rel_of.append(lambda it, _s=su, _i=idx: max(0.0, _s[_i[it]]) if it in _i else 0.0)
            ideal_dcg.append(_ideal_discounted(su.tolist(), k))
            ideal_sum.append(_ideal_sum(su.tolist(), k))
            decoupled_scores.append(su)
        eval_members.append(u)
    n = len(eval_members)
    if n == 0:
        return

    U = np.zeros(n)          # cumulative discounted utility per member
    sat_sum = np.zeros(n)    # sum over rounds of per-round satisfaction
    rounds = 0

    for lst in round_lists:
        for i in range(n):
            rels = [rel_of[i](it) for it in lst[:k]]
            U[i] += _discounted(rels)
            if ideal_sum[i] > 0:
                sat_sum[i] += sum(max(r, 0.0) for r in rels) / ideal_sum[i]
        rounds += 1

        # per-session base metrics (aggregated across members), one sample per round
        for name, kk in specs:
            fn = BASE_METRICS[name]
            vals = []
            for i, u in enumerate(eval_members):
                if proto == "coupled":
                    g = gt[u]
                    gains, relevant = g["gains"], g["relevant"]
                else:
                    gains, relevant = decoupled_member(decoupled_scores[i], train.items, kk)
                vals.append(fn(lst, gains, relevant, kk))
            vals = np.asarray(vals, dtype=float)
            for a in group_aggregations:
                acc[(proto, name, kk, a)].append(GROUP_AGGREGATIONS[a](vals))

    # -- long-term metrics --
    satO = sat_sum / rounds
    if "groupSatO" in long_term_metrics:
        acc[(proto, "groupSatO", k, "group")].append(float(satO.mean()))
    if "groupDisO" in long_term_metrics:
        acc[(proto, "groupDisO", k, "group")].append(float(satO.max() - satO.min()))
    if "dMAE" in long_term_metrics:
        total = float(U.sum())
        if total > 0 and n > 0:
            dmae = float(np.mean(np.abs(U / total - 1.0 / n)))
        else:
            dmae = 0.0
        acc[(proto, "dMAE", k, "group")].append(dmae)


__all__ = ["evaluate_sequential", "LONG_TERM_METRICS"]
