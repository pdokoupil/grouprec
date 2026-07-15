"""Single-user recommender backends.

We do **not** reimplement base recommenders; we accept any object satisfying the
:class:`BaseRecommender` protocol and ship thin adapters (later) plus one trivial,
dependency-free built-in (:class:`Popularity`) so the library works out of the box.

A backend's :meth:`score` returns a ``(n_query_users, n_items)`` matrix whose
columns follow ``dataset.items`` order, so the numpy aggregators consume it directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy import sparse

from ..data import Dataset


@runtime_checkable
class BaseRecommender(Protocol):
    """The interop contract: bring your own recommender (LensKit/implicit/RecBole/...)."""

    def fit(self, dataset: Dataset) -> "BaseRecommender": ...

    def score(self, users, items=None) -> np.ndarray:
        """Per-user item scores, shape ``(len(users), n_items or len(items))``,
        columns in ``dataset.items`` order (or the order of ``items`` if given)."""
        ...


class _FittedMixin:
    dataset_: Dataset

    def _check_fitted(self) -> None:
        if getattr(self, "dataset_", None) is None:
            raise RuntimeError(f"{type(self).__name__} must be fit() before scoring.")

    def _item_cols(self, items):
        if items is None:
            return slice(None)
        return np.array([self.dataset_.item_index[i] for i in items], dtype=np.int64)


class Popularity(_FittedMixin):
    """Global-popularity baseline: every user gets the same item scores.

    ``measure="count"`` ranks by interaction count; ``measure="mean"`` by mean
    rating (falling back to count when ratings are absent).
    """

    def __init__(self, measure: str = "count") -> None:
        if measure not in ("count", "mean"):
            raise ValueError("measure must be 'count' or 'mean'.")
        self.measure = measure
        self.dataset_ = None  # type: ignore[assignment]
        self.popularity_ = None

    def fit(self, dataset: Dataset) -> "Popularity":
        self.dataset_ = dataset
        idx = dataset.interactions["item"].map(dataset.item_index).to_numpy()
        pop = np.zeros(dataset.n_items)
        if self.measure == "mean" and dataset.has_ratings:
            r = dataset.interactions["rating"].to_numpy(dtype=float)
            counts = np.bincount(idx, minlength=dataset.n_items)
            sums = np.bincount(idx, weights=r, minlength=dataset.n_items)
            pop = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        else:
            pop = np.bincount(idx, minlength=dataset.n_items).astype(float)
        self.popularity_ = pop
        return self

    def score(self, users, items=None) -> np.ndarray:
        self._check_fitted()
        cols = self._item_cols(items)
        base = self.popularity_[cols]
        return np.tile(base, (len(list(users)), 1))


class Random(_FittedMixin):
    """Random-score baseline (seeded); useful as a sanity floor."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.dataset_ = None  # type: ignore[assignment]

    def fit(self, dataset: Dataset) -> "Random":
        self.dataset_ = dataset
        return self

    def score(self, users, items=None) -> np.ndarray:
        self._check_fitted()
        n_items = self.dataset_.n_items if items is None else len(list(items))
        rng = np.random.default_rng(self.seed)
        return rng.random((len(list(users)), n_items))


class _UserMatrixMixin(_FittedMixin):
    """Shared scoring for item-similarity models that score ``x_u @ W``."""

    X_: np.ndarray  # user-item interactions; dense or scipy.sparse CSR
    W_: np.ndarray  # item-item weights/similarity (n_items, n_items)

    def score(self, users, items=None) -> np.ndarray:
        self._check_fitted()
        # Gather the requested users' rows in one shot: for a CSR X_ this is a sparse
        # row-slice @ dense W_, which stays sparse on the left and never materialises
        # the full user-item matrix. Unknown users score as an all-zero profile.
        idx = [self.dataset_.user_index.get(u) for u in users]
        pos = [j for j, i in enumerate(idx) if i is not None]
        s = np.zeros((len(idx), self.dataset_.n_items), dtype=float)
        if pos:
            s[pos] = np.asarray(self.X_[[idx[j] for j in pos]] @ self.W_)
        cols = self._item_cols(items)
        return s[:, cols] if items is not None else s

    def score_profile(self, profiles, items=None) -> np.ndarray:
        """Score arbitrary item-interaction profiles (``(n, n_items)``) -- enables
        profile-first / pseudo-user group recommendation. Returns ``profiles @ W``."""
        self._check_fitted()
        s = np.asarray(profiles, dtype=float) @ self.W_
        cols = self._item_cols(items)
        return s[:, cols] if items is not None else s


class EASE(_UserMatrixMixin):
    """EASE^R -- closed-form shallow autoencoder (Steck, WWW'19): a strong,
    dependency-free linear baseline. One matrix inverse, numpy/scipy only.

    Parameters
    ----------
    reg : L2 regularization ``lambda`` on the item-item weights.
    binarize : score on the binary interaction matrix (default) or raw ratings.
    """

    def __init__(self, reg: float = 250.0, binarize: bool = True) -> None:
        self.reg = float(reg)
        self.binarize = binarize
        self.dataset_ = None  # type: ignore[assignment]

    def fit(self, dataset: Dataset) -> "EASE":
        self.dataset_ = dataset
        # Sparse X keeps memory at O(nnz); the Gram matrix is (n_items, n_items) and
        # is dense by nature -- that, not the interactions, is EASE's real size limit.
        X = dataset.user_item_csr(value="binary" if self.binarize else "rating")
        G = np.asarray((X.T @ X).todense())
        diag = np.diag_indices(G.shape[0])
        G[diag] += self.reg
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag] = 0.0
        self.X_, self.W_ = X, B
        return self


class ItemKNN(_UserMatrixMixin):
    """Item-item cosine kNN baseline (numpy/scipy only).

    Parameters
    ----------
    k : neighbors kept per item (``None`` keeps all).
    binarize : cosine over the binary interaction matrix (default) or raw ratings.
    """

    def __init__(self, k: int | None = 20, binarize: bool = True) -> None:
        self.k = k
        self.binarize = binarize
        self.dataset_ = None  # type: ignore[assignment]

    def fit(self, dataset: Dataset) -> "ItemKNN":
        self.dataset_ = dataset
        X = dataset.user_item_csr(value="binary" if self.binarize else "rating")
        norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        Xn = X @ sparse.diags(1.0 / np.where(norm > 0, norm, 1.0))
        S = np.asarray((Xn.T @ Xn).todense())
        np.fill_diagonal(S, 0.0)
        if self.k is not None and self.k < S.shape[0]:
            # keep only the top-k neighbours per item (zero the rest)
            keep = np.argsort(-S, axis=1)[:, : self.k]
            mask = np.zeros_like(S, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            S = np.where(mask, S, 0.0)
        self.X_, self.W_ = X, S
        return self


from .adapters import (  # noqa: E402  (re-export adapters; placed last to avoid cycles)
    ImplicitRecommender,
    LensKitRecommender,
    RecBoleRecommender,
    implicit_als,
    implicit_bpr,
    lenskit,
    recbole,
)

__all__ = [
    "BaseRecommender",
    "Popularity",
    "Random",
    "EASE",
    "ItemKNN",
    "ImplicitRecommender",
    "LensKitRecommender",
    "RecBoleRecommender",
    "implicit_als",
    "implicit_bpr",
    "lenskit",
    "recbole",
]
