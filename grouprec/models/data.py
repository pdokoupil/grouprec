"""Group training data for deep models.

Deep group models need a signal the aggregator pipeline doesn't: **group-item
interactions** (which items each group consumed). This module defines the lightweight
container and a synthetic generator for tests/examples. Real group datasets
(CAMRa2011, Mafengwo, Yelp/Douban/Weeplaces) provide these directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Dataset, Groups
from ..groups import synthetic
from ..split import Split


def normalize_group_interactions(group_interactions, n_groups: int) -> dict[int, list]:
    """Coerce dict / DataFrame(group, item) -> {group_index: [item ids]}."""
    if isinstance(group_interactions, pd.DataFrame):
        out: dict[int, list] = {gi: [] for gi in range(n_groups)}
        for gi, item in zip(group_interactions["group"], group_interactions["item"]):
            out.setdefault(int(gi), []).append(int(item))
        return out
    return {int(gi): [int(x) for x in items] for gi, items in dict(group_interactions).items()}


@dataclass
class GroupTrainData:
    """Bundle for a deep-model benchmark: user-item data + groups + their training
    interactions + held-out group choices (for group-level LOO evaluation)."""

    dataset: Dataset
    groups: Groups
    group_interactions: dict[int, list]   # group index -> training item ids
    group_truth: dict[int, list]          # group index -> held-out item id(s)
    split: Split


def make_synthetic_group_data(
    n_users: int = 80,
    n_items: int = 60,
    n_clusters: int = 4,
    n_groups: int = 30,
    group_size: int = 4,
    items_per_group: int = 8,
    seed: int | None = 0,
) -> GroupTrainData:
    """Clustered users -> similar groups -> each group 'consumes' items its members
    like; one item per group is held out as the group's choice."""
    from ..data import make_blobs_dataset

    data = make_blobs_dataset(n_users=n_users, n_items=n_items, n_clusters=n_clusters,
                              density=1.0, seed=seed)
    groups = synthetic(data, kind="similar", size=group_size, n=n_groups,
                       metric="pearson", sim_high=0.2, seed=seed)
    rng = np.random.default_rng(seed)
    R = data.user_item_matrix(value="rating")

    group_interactions: dict[int, list] = {}
    group_truth: dict[int, list] = {}
    for gi, members in enumerate(groups):
        midx = [data.user_index[u] for u in members]
        liking = R[midx].mean(axis=0)              # items the group collectively likes
        ranked = np.argsort(-liking)
        chosen = ranked[: items_per_group + 1]
        items = [int(data.items[c]) for c in chosen]
        group_truth[gi] = [items[0]]               # hold out the top one
        group_interactions[gi] = items[1:]         # the rest are training signal
    split = Split(train=data, test=data.interactions.iloc[:0])
    return GroupTrainData(data, groups, group_interactions, group_truth, split)


__all__ = ["GroupTrainData", "make_synthetic_group_data", "normalize_group_interactions"]
