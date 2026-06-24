"""Dataset preprocessing: k-core filtering, minimum-count filtering, binarization."""

from __future__ import annotations

from ..data import Dataset


def k_core(dataset: Dataset, k: int = 5, max_iter: int = 100) -> Dataset:
    """Iteratively drop users and items with fewer than ``k`` interactions until the
    remaining graph is a k-core (every user and item has degree >= ``k``)."""
    if k <= 1:
        return dataset
    df = dataset.interactions
    for _ in range(max_iter):
        uc = df["user"].value_counts()
        ic = df["item"].value_counts()
        keep_u = uc.index[uc >= k]
        keep_i = ic.index[ic >= k]
        new = df[df["user"].isin(keep_u) & df["item"].isin(keep_i)]
        if len(new) == len(df):
            break
        df = new
    return Dataset(df.reset_index(drop=True), name=dataset.name)


def filter_min_interactions(dataset: Dataset, *, min_per_user: int = 1,
                            min_per_item: int = 1) -> Dataset:
    """Single-pass filter on minimum interactions per user and per item."""
    df = dataset.interactions
    uc = df["user"].value_counts()
    ic = df["item"].value_counts()
    keep_u = uc.index[uc >= min_per_user]
    keep_i = ic.index[ic >= min_per_item]
    new = df[df["user"].isin(keep_u) & df["item"].isin(keep_i)].reset_index(drop=True)
    return Dataset(new, name=dataset.name)


def binarize(dataset: Dataset, threshold: float = 4.0) -> Dataset:
    """Keep interactions with ``rating >= threshold`` and drop the rating column
    (the standard implicit-feedback conversion). Delegates to ``Dataset.binarize``."""
    return dataset.binarize(threshold=threshold)


__all__ = ["k_core", "filter_min_interactions", "binarize"]
