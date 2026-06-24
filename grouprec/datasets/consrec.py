"""Parser for the AGREE / ConsRec group-dataset format (CAMRa2011, Mafengwo).

Files (per dataset dir):
  groupMember.txt          ``gid m1,m2,...``
  groupRatingTrain.txt     ``gid item rating``      (group-item training positives)
  groupRatingTest.txt      ``gid item rating``      (held-out group positives, LOO)
  groupRatingNegative.txt  ``(gid,item) n1 n2 ...``  (99 sampled negatives per test pos)
  userRatingTrain.txt      ``uid item rating``      (user-item training)

The held-out test uses the **sampled ranking protocol** (1 positive vs 99 negatives),
which is what the deep-group literature reports -- see
:func:`grouprec.eval.evaluate_sampled`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import Dataset, Groups
from ..split import Split


@dataclass
class GroupBenchmarkData:
    """Everything needed to benchmark deep models + aggregators on a real dataset."""

    dataset: Dataset                                   # user-item training
    groups: Groups                                     # membership
    group_interactions: dict[int, list]                # group index -> training item ids
    test_instances: list[tuple[int, int, list[int]]]   # (group index, pos item, [neg items])
    name: str

    @property
    def split(self) -> Split:
        return Split(train=self.dataset, test=self.dataset.interactions.iloc[:0])


def _read_triples(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["a", "b", "c"],
                       engine="python")


def load_consrec(path, name: str | None = None) -> GroupBenchmarkData:
    """Load a CAMRa2011/Mafengwo-style directory into :class:`GroupBenchmarkData`."""
    root = Path(path)
    name = name or root.name

    # -- group membership (user ids preserved) --
    members: list[np.ndarray] = []
    gid_to_index: dict[int, int] = {}
    with open(root / "groupMember.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gid_str, _, rest = line.partition(" ")
            gid = int(gid_str)
            gid_to_index[gid] = len(members)
            members.append(np.array([int(x) for x in rest.split(",") if x != ""], dtype=np.int64))
    groups = Groups(members, metadata={"kind": "explicit", "source": name})

    user_train = _read_triples(root / "userRatingTrain.txt")
    group_train = _read_triples(root / "groupRatingTrain.txt")
    group_test = _read_triples(root / "groupRatingTest.txt")

    # -- negatives keyed by (gid, item) --
    negatives: dict[tuple[int, int], list[int]] = {}
    with open(root / "groupRatingNegative.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, rest = line.partition(" ")
            g, i = key.strip("()").split(",")
            negatives[(int(g), int(i))] = [int(x) for x in rest.split()]

    # -- vocabularies spanning every file so train/test/neg item spaces align --
    all_items = set(user_train["b"]) | set(group_train["b"]) | set(group_test["b"])
    for negs in negatives.values():
        all_items.update(negs)
    all_users = set(user_train["a"]) | {int(u) for m in members for u in m}

    dataset = Dataset(
        user_train.rename(columns={"a": "user", "b": "item", "c": "rating"}),
        name=name, users=sorted(all_users), items=sorted(all_items),
    )

    # -- group training interactions --
    group_interactions: dict[int, list] = {i: [] for i in range(len(members))}
    for gid, item in zip(group_train["a"], group_train["b"]):
        gi = gid_to_index.get(int(gid))
        if gi is not None:
            group_interactions[gi].append(int(item))

    # -- test instances (pos + its 99 negatives) --
    test_instances: list[tuple[int, int, list[int]]] = []
    for gid, item in zip(group_test["a"], group_test["b"]):
        gi = gid_to_index.get(int(gid))
        negs = negatives.get((int(gid), int(item)))
        if gi is not None and negs is not None:
            test_instances.append((gi, int(item), negs))

    return GroupBenchmarkData(dataset, groups, group_interactions, test_instances, name)


__all__ = ["GroupBenchmarkData", "load_consrec"]
