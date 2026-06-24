"""Sampled group-level ranking evaluation (1 positive vs N sampled negatives).

This is the protocol the deep group-recommendation literature reports on CAMRa2011 /
Mafengwo / Yelp / Douban / Weeplaces (HR@k, nDCG@k over the held-out positive ranked
against ~99 negatives). It works for **any** recommender exposing ``recommend`` --
results-aggregators and profile-aggregation deep models alike -- by restricting the
candidate set, so it is the faithful bridge for reproduction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ..aggregators.base import SequentialAggregator
from ..data import Dataset, Groups
from ._common import Report


def evaluate_sampled(
    rec,
    dataset: Dataset,
    groups: Groups,
    test_instances: Sequence[tuple[int, int, Sequence[int]]],
    *,
    ks: Sequence[int] = (5, 10),
    fit: bool = True,
) -> Report:
    """Rank each test positive against its sampled negatives; report HR@k / nDCG@k.

    Parameters
    ----------
    test_instances : ``(group_index, positive_item_id, [negative_item_ids])`` tuples.
    ks : cutoffs to report.
    fit : whether to ``rec.fit(dataset)`` first (set False if already fit).
    """
    if fit:
        rec.fit(dataset)
    agg_obj = getattr(rec, "aggregator", None)
    acc: dict[tuple, list[float]] = defaultdict(list)
    max_k = max(ks)

    for gi, pos, negs in test_instances:
        members = groups[gi]
        candidates = [int(pos)] + [int(n) for n in negs]
        if isinstance(agg_obj, SequentialAggregator):
            agg_obj.reset()
        # rank within the candidate set (so ranking-aware aggregators rank candidates,
        # not the global item space)
        ranked = [int(x) for x in rec.recommend(members, k=min(len(candidates), max_k),
                                                candidates=candidates)]
        for k in ks:
            topk = ranked[:k]
            if pos in topk:
                rank = topk.index(pos)
                acc[("coupled", "hr", k, "sampled")].append(1.0)
                acc[("coupled", "ndcg", k, "sampled")].append(1.0 / np.log2(rank + 2))
            else:
                acc[("coupled", "hr", k, "sampled")].append(0.0)
                acc[("coupled", "ndcg", k, "sampled")].append(0.0)

    records = [
        {"protocol": p, "metric": m, "k": kk, "aggregation": a,
         "value": float(np.mean(v)) if v else float("nan")}
        for (p, m, kk, a), v in sorted(acc.items())
    ]
    return Report(records)


__all__ = ["evaluate_sampled"]
