"""Core data containers: :class:`Dataset` and :class:`Groups`.

A :class:`Dataset` wraps a long-format interactions table and maintains contiguous
user/item index maps so that the numpy aggregators (which work in item-*index*
space, columns ``0..n_items-1`` in ``dataset.items`` order) integrate cleanly with
id-based data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd


class Dataset:
    """A group-recommendation dataset of user-item interactions.

    Parameters
    ----------
    interactions : DataFrame with at least ``user`` and ``item`` columns, optionally
        ``rating`` and ``timestamp``.
    name : optional dataset name (provenance).

    Item columns of any score/rating matrix are ordered by :attr:`items` (sorted
    unique item ids), so aggregator output indices map back via ``dataset.items[idx]``.
    """

    def __init__(self, interactions: pd.DataFrame, *, name: str | None = None,
                 users=None, items=None) -> None:
        missing = {"user", "item"} - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing required column(s): {sorted(missing)}")
        self.interactions = interactions.reset_index(drop=True)
        self.name = name
        # Optional explicit vocabularies let train/test/negative item spaces align
        # (e.g. the AGREE/ConsRec sampled-ranking protocol). Must be supersets.
        self.users = np.sort(np.unique(users)) if users is not None \
            else np.sort(self.interactions["user"].unique())
        self.items = np.sort(np.unique(items)) if items is not None \
            else np.sort(self.interactions["item"].unique())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {it: j for j, it in enumerate(self.items)}

    # -- basic properties ---------------------------------------------------- #
    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def has_ratings(self) -> bool:
        return "rating" in self.interactions.columns

    @property
    def has_timestamps(self) -> bool:
        return "timestamp" in self.interactions.columns

    def __len__(self) -> int:
        return len(self.interactions)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nm = f"{self.name!r}, " if self.name else ""
        return (f"Dataset({nm}{self.n_users} users, {self.n_items} items, "
                f"{len(self)} interactions)")

    # -- constructors -------------------------------------------------------- #
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        *,
        user_col: str = "user",
        item_col: str = "item",
        rating_col: str | None = None,
        timestamp_col: str | None = None,
        name: str | None = None,
    ) -> "Dataset":
        """Build a Dataset from an arbitrary DataFrame by naming its columns."""
        cols = {user_col: "user", item_col: "item"}
        if rating_col is not None:
            cols[rating_col] = "rating"
        if timestamp_col is not None:
            cols[timestamp_col] = "timestamp"
        return cls(df.rename(columns=cols)[list(cols.values())], name=name)

    # -- views --------------------------------------------------------------- #
    def user_item_matrix(self, value: str = "rating", fill: float = 0.0) -> np.ndarray:
        """Dense ``(n_users, n_items)`` matrix in ``users`` x ``items`` order.

        ``value="rating"`` uses the rating column (falling back to binary if absent);
        ``value="binary"`` marks 1 for any interaction.
        """
        ui = self.interactions["user"].map(self.user_index).to_numpy()
        ij = self.interactions["item"].map(self.item_index).to_numpy()
        mat = np.full((self.n_users, self.n_items), float(fill))
        if value == "binary" or not self.has_ratings:
            mat[ui, ij] = 1.0
        elif value == "rating":
            mat[ui, ij] = self.interactions["rating"].to_numpy(dtype=float)
        else:
            raise ValueError("value must be 'rating' or 'binary'.")
        return mat

    def items_seen_by(self, user) -> np.ndarray:
        """Item *indices* the given user id interacted with (for exclusion)."""
        seen = self.interactions.loc[self.interactions["user"] == user, "item"]
        return np.array([self.item_index[i] for i in seen.unique()], dtype=np.int64)

    def binarize(self, threshold: float = 4.0) -> "Dataset":
        """Return a new Dataset keeping only interactions with ``rating >= threshold``
        and dropping the rating column (the standard implicit conversion)."""
        if not self.has_ratings:
            return self
        kept = self.interactions.loc[self.interactions["rating"] >= threshold]
        kept = kept.drop(columns=["rating"]).reset_index(drop=True)
        return Dataset(kept, name=self.name)


@dataclass
class Groups:
    """A collection of groups, each a 1-D array of user ids.

    ``metadata`` records provenance (kind, size, metric, seed, thresholds) so an
    experiment is reproducible.
    """

    members: list[np.ndarray]
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.members)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.members[i]

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(g) for g in self.members], dtype=int)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = self.metadata.get("kind", "?")
        return f"Groups(n={len(self)}, kind={kind!r}, sizes={sorted(set(self.sizes.tolist()))})"


def make_blobs_dataset(
    n_users: int = 60,
    n_items: int = 40,
    n_clusters: int = 3,
    noise: float = 0.4,
    density: float = 1.0,
    seed: int | None = 0,
) -> Dataset:
    """Synthetic clustered-preference dataset for tests/examples.

    Users are assigned round-robin to ``n_clusters`` taste clusters; each cluster has
    a random item-rating profile, and users rate (a ``density`` fraction of) items
    near their cluster profile plus Gaussian ``noise``. Same-cluster users are
    Pearson-correlated, enabling meaningful ``similar`` / ``divergent`` groups.
    """
    rng = np.random.default_rng(seed)
    profiles = rng.uniform(1.0, 5.0, size=(n_clusters, n_items))
    n_rate = max(1, int(round(density * n_items)))
    rows = []
    for u in range(n_users):
        c = u % n_clusters
        items = rng.choice(n_items, size=n_rate, replace=False)
        for it in items:
            r = float(np.clip(profiles[c, it] + rng.normal(0, noise), 1.0, 5.0))
            rows.append((u, it, round(r, 3)))
    df = pd.DataFrame(rows, columns=["user", "item", "rating"])
    return Dataset(df, name="blobs")
