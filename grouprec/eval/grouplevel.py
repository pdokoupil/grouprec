"""Group-level (coupled) evaluation -- the bridge between paradigms.

Here the ground truth is the **group's** held-out item(s) (leave-one-out), not each
member's feedback. This is the protocol the deep group-recommendation literature
reports (HR@k / nDCG@k on CAMRa2011 / Mafengwo / Yelp / Douban / Weeplaces), and the
*only* setting where results-aggregators and profile-aggregation deep models are
directly comparable -- both emit a group top-k list scored against the same group
choice. Decoupled is not defined here (deep models have no per-member ``r̂``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ..aggregators.base import SequentialAggregator
from ..data import Dataset, Groups
from ..split import Split
from ._common import Report, parse_metric
from .metrics import BASE_METRICS


def evaluate_grouplevel(
    rec,
    data: Dataset,
    groups: Groups,
    splits,
    group_truth: dict[int, Sequence],
    *,
    k: int = 10,
    metrics: Sequence[str] = ("hr", "ndcg", "recall"),
    exclude_seen: bool = True,
) -> Report:
    """Coupled group-level evaluation against each group's held-out item(s).

    Parameters
    ----------
    group_truth : maps a group's position index (0..len(groups)-1) to its held-out
        item id(s). Groups without an entry are skipped.
    splits : the LOO split(s); the held-out group items must live in ``split.test``
        space (i.e. excluded from training), and are not removed by ``exclude_seen``.
    """
    if isinstance(splits, Split):
        splits = [splits]
    specs = [parse_metric(m, k) for m in metrics]
    max_k = max(kk for _, kk in specs)
    acc: dict[tuple, list[float]] = defaultdict(list)

    for split in splits:
        rec.fit(split.train)
        train = split.train
        agg_obj = getattr(rec, "aggregator", None)
        for gi, group in enumerate(groups):
            relevant = set(int(x) for x in group_truth.get(gi, ()))
            if not relevant:
                continue
            members = list(group)
            if isinstance(agg_obj, SequentialAggregator):
                agg_obj.reset()
            exclude = None
            if exclude_seen:
                seen: set = set()
                for u in members:
                    if u in train.user_index:
                        seen.update(int(x) for x in train.items[train.items_seen_by(u)])
                exclude = seen - relevant  # never exclude the held-out group item(s)
            rec_items = [int(x) for x in rec.recommend(members, k=max_k, exclude=exclude)]
            gains = {it: 1.0 for it in relevant}
            for name, kk in specs:
                acc[("coupled", name, kk, "group")].append(
                    BASE_METRICS[name](rec_items, gains, relevant, kk))

    records = [
        {"protocol": p, "metric": m, "k": kk, "aggregation": a,
         "value": float(np.mean(v)) if v else float("nan")}
        for (p, m, kk, a), v in sorted(acc.items())
    ]
    return Report(records)


__all__ = ["evaluate_grouplevel"]
