"""Parser for the GroupIM dataset format (Weeplaces, and GroupIM's Yelp/Douban dumps).

Files: ``group_users.csv`` (group,user), ``train_gi.csv`` (group,item),
``train_ui.csv`` (user,item), ``test_gi.csv`` (group,item held-out). There is no
negatives file, so we **sample** N negatives per held-out positive (items the group
did not interact with) to build the 1-vs-N test instances.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import Dataset, Groups
from .consrec import GroupBenchmarkData


def _read(path):
    df = pd.read_csv(path)
    return df.astype({c: "int64" for c in df.columns} if df.values.dtype != object
                     else {c: float for c in df.columns}).astype("int64", errors="ignore")


def load_groupim(path, name: str | None = None, *, n_negatives: int = 100,
                 seed: int | None = 0) -> GroupBenchmarkData:
    """Load a GroupIM-format directory (e.g. Weeplaces) into GroupBenchmarkData."""
    root = Path(path)
    name = name or root.name

    gu = pd.read_csv(root / "group_users.csv").astype(float).astype(int)
    train_gi = pd.read_csv(root / "train_gi.csv").astype(float).astype(int)
    train_ui = pd.read_csv(root / "train_ui.csv").astype(float).astype(int)
    test_gi = pd.read_csv(root / "test_gi.csv").astype(float).astype(int)

    group_ids = sorted(gu["group"].unique())
    gid_to_index = {g: i for i, g in enumerate(group_ids)}
    members_by_gid: dict[int, list] = defaultdict(list)
    for g, u in zip(gu["group"], gu["user"]):
        members_by_gid[int(g)].append(int(u))
    members = [np.array(members_by_gid[g], dtype=np.int64) for g in group_ids]
    groups = Groups(members, metadata={"kind": "inferred", "source": name})

    all_items = set(train_gi["item"]) | set(train_ui["item"]) | set(test_gi["item"])
    all_users = set(train_ui["user"]) | {int(u) for m in members for u in m}
    dataset = Dataset(train_ui.rename(columns={"user": "user", "item": "item"}),
                      name=name, users=sorted(all_users), items=sorted(all_items))

    group_interactions: dict[int, list] = {i: [] for i in range(len(group_ids))}
    pos_by_index: dict[int, set] = defaultdict(set)
    for g, it in zip(train_gi["group"], train_gi["item"]):
        gi = gid_to_index.get(int(g))
        if gi is not None:
            group_interactions[gi].append(int(it))
            pos_by_index[gi].add(int(it))

    items_arr = np.array(sorted(all_items))
    rng = np.random.default_rng(seed)
    test_instances: list[tuple[int, int, list[int]]] = []
    for g, it in zip(test_gi["group"], test_gi["item"]):
        gi = gid_to_index.get(int(g))
        if gi is None:
            continue
        forbidden = pos_by_index[gi] | {int(it)}
        target = min(n_negatives, items_arr.size - len(forbidden))
        negs: list[int] = []
        attempts = 0
        max_attempts = 50 * max(target, 1)
        while len(negs) < target and attempts < max_attempts:
            cand = int(items_arr[rng.integers(0, items_arr.size)])
            attempts += 1
            if cand not in forbidden:
                negs.append(cand)
                forbidden.add(cand)
        test_instances.append((gi, int(it), negs))

    return GroupBenchmarkData(dataset, groups, group_interactions, test_instances, name)


__all__ = ["load_groupim"]
