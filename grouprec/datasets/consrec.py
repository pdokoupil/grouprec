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


def _read_pairs(path: Path, positives_only: bool = True) -> list[tuple[int, int]]:
    """(a, b) interaction pairs. Files are either ``a b`` (Mafengwo) or ``a b rating``
    (CAMRa, ratings 0-100); for the 3-column form, ``rating == 0`` rows are negatives
    and dropped when ``positives_only`` (matching the original AGREE/ConsRec loaders)."""
    pairs: list[tuple[int, int]] = []
    with open(path) as f:
        for line in f:
            c = line.split()
            if not c:
                continue
            if len(c) > 2:
                if (not positives_only) or int(c[2]) > 0:
                    pairs.append((int(c[0]), int(c[1])))
            else:
                pairs.append((int(c[0]), int(c[1])))
    return pairs


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

    user_train = _read_pairs(root / "userRatingTrain.txt")            # positives only
    group_train = _read_pairs(root / "groupRatingTrain.txt")          # positives only
    group_test = _read_pairs(root / "groupRatingTest.txt", positives_only=False)  # held-out pos

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
    all_items = {b for _, b in user_train} | {b for _, b in group_train} | {b for _, b in group_test}
    for negs in negatives.values():
        all_items.update(negs)
    all_users = {a for a, _ in user_train} | {int(u) for m in members for u in m}

    dataset = Dataset(
        pd.DataFrame(user_train, columns=["user", "item"]),   # implicit (positives)
        name=name, users=sorted(all_users), items=sorted(all_items),
    )

    # -- group training interactions --
    group_interactions: dict[int, list] = {i: [] for i in range(len(members))}
    for gid, item in group_train:
        gi = gid_to_index.get(int(gid))
        if gi is not None:
            group_interactions[gi].append(int(item))

    # -- test instances (pos + its sampled negatives) --
    test_instances: list[tuple[int, int, list[int]]] = []
    for gid, item in group_test:
        gi = gid_to_index.get(int(gid))
        negs = negatives.get((int(gid), int(item)))
        if gi is not None and negs is not None:
            test_instances.append((gi, int(item), negs))

    return GroupBenchmarkData(dataset, groups, group_interactions, test_instances, name)


__all__ = ["GroupBenchmarkData", "load_consrec"]
