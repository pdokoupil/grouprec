"""Train/test splitting.

Each splitter returns :class:`Split` objects holding a training :class:`Dataset`
and a held-out test interactions DataFrame (same columns as the source).

* :func:`random_split` -- random holdout by interaction.
* :func:`crossval`     -- k-fold cross-validation (list of Splits).
* :func:`leave_one_out`-- hold out one interaction per user (latest if timestamps
  are present, else random) -- the protocol the deep-model literature reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Dataset


@dataclass
class Split:
    """A single train/test split."""

    train: Dataset
    test: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Split(train={len(self.train)} ix, test={len(self.test)} ix)"


def _make(parent: Dataset, train_mask: np.ndarray) -> Split:
    df = parent.interactions
    train = Dataset(df.loc[train_mask].reset_index(drop=True), name=parent.name)
    test = df.loc[~train_mask].reset_index(drop=True)
    return Split(train=train, test=test)


def random_split(data: Dataset, test_frac: float = 0.2, seed: int | None = None) -> Split:
    """Hold out a random ``test_frac`` of interactions."""
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1).")
    n = len(data)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx = perm[:n_test]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    return _make(data, mask)


def crossval(data: Dataset, k: int = 5, seed: int | None = None) -> list[Split]:
    """k-fold cross-validation; every interaction is in exactly one test fold."""
    if k < 2:
        raise ValueError("k must be >= 2.")
    n = len(data)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    splits = []
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        splits.append(_make(data, mask))
    return splits


def leave_one_out(data: Dataset, seed: int | None = None) -> Split:
    """Hold out one interaction per user (latest by timestamp if available)."""
    df = data.interactions
    rng = np.random.default_rng(seed)
    test_positions = []
    for _, grp in df.groupby("user", sort=False):
        if data.has_timestamps:
            pos = grp["timestamp"].idxmax()
        else:
            pos = int(rng.choice(grp.index.to_numpy()))
        test_positions.append(pos)
    mask = np.ones(len(df), dtype=bool)
    mask[df.index.get_indexer(test_positions)] = False
    return _make(data, mask)
