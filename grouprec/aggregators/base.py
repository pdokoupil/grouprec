"""Core aggregator abstractions.

A group recommender in *results-first* mode takes a per-member score matrix and
emits a single ranked group list. Every aggregator here therefore operates on a
dense ``scores`` array of shape ``(n_members, n_items)`` and returns an array of
selected **column indices** (item positions) in selection order. Mapping those
indices back to dataset item ids is the pipeline's job, not the aggregator's.

This keeps the numeric core dependency-light (numpy only) and trivially testable,
while the higher-level :class:`~grouprec.GroupRecommender` wrappers bind an
aggregator to a base recommender.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Literal

import numpy as np

Paradigm = Literal["results", "profile"]

ArrayLike = "np.ndarray | object"


def as_score_matrix(scores) -> np.ndarray:
    """Coerce ``scores`` to a 2-D float ``(n_members, n_items)`` numpy array."""
    rm = np.asarray(scores, dtype=float)
    if rm.ndim != 2:
        raise ValueError(
            f"scores must be 2-D (n_members, n_items); got shape {rm.shape}."
        )
    return rm


def available_mask(n_items: int, exclude: Iterable[int] | None) -> np.ndarray:
    """Boolean mask of selectable items, with ``exclude`` indices turned off."""
    mask = np.ones(n_items, dtype=bool)
    if exclude is not None:
        idx = np.asarray(list(exclude), dtype=int)
        if idx.size:
            mask[idx] = False
    return mask


def top_k_indices(item_scores: np.ndarray, k: int, available: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``k`` highest-scoring items, ties broken by ascending index.

    This reproduces the pandas ``groupby(item).agg(...).sort_values(ascending=False)``
    convention used by the reference implementations: a *stable* descending sort,
    which (because items are laid out in ascending-id column order) breaks ties by
    the smaller item id.
    """
    scores = np.asarray(item_scores, dtype=float)
    if available is not None:
        scores = np.where(available, scores, -np.inf)
        budget = int(min(k, int(available.sum())))
    else:
        budget = int(min(k, scores.shape[0]))
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    # stable argsort on the negated scores -> descending, ties by ascending index.
    order = np.argsort(-scores, kind="stable")[:budget]
    return order.astype(np.int64)


class Aggregator(ABC):
    """Base class for results-first group aggregators.

    Subclasses implement :meth:`aggregate`. The :attr:`paradigm` attribute lets the
    pipeline and benchmarks group/filter aggregators by how they consume data --
    the rift-bridging capability of the library.
    """

    paradigm: Paradigm = "results"

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"AVG"`` -- used in result tables."""

    @abstractmethod
    def aggregate(self, scores, k: int, *, exclude: Iterable[int] | None = None) -> np.ndarray:
        """Return the top-``k`` item indices for one group.

        Parameters
        ----------
        scores : array-like, shape (n_members, n_items)
            Per-member predicted scores for every candidate item.
        k : int
            Number of items to return.
        exclude : iterable of int, optional
            Item indices that must never be selected (e.g. already-seen items).
        """

    #: ``True`` for score-reduction aggregators that expose a static per-item group
    #: utility via :meth:`score_items`; ``False`` for selection / greedy / sequential
    #: aggregators whose output is only an ordering. Lets callers (and
    #: :meth:`~grouprec.GroupRecommender.group_scores`) tell the two apart.
    produces_item_scores: bool = False

    def score_items(self, scores, *, exclude: Iterable[int] | None = None) -> np.ndarray:
        """Per-item aggregated group utility, shape ``(n_items,)``.

        Defined only for **score-reduction** aggregators (the per-item value the
        ranking sorts on -- mean, weighted mean, min, ...). Selection-based
        aggregators (FAI and the greedy/fairness/sequential families) build a list
        incrementally and have no such static per-item score, so they leave this
        raising; rank them with :meth:`aggregate` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} ({self.name}) is selection-based and exposes no "
            "per-item group score; use aggregate()/GroupRecommender.recommend() for an "
            "ordering.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class SequentialAggregator(Aggregator):
    """Base class for aggregators that carry fairness state across sessions.

    Create one instance per group, call :meth:`aggregate` once per session (state
    accumulates), and :meth:`reset` to start a fresh group. Stateless aggregators
    should subclass :class:`Aggregator` instead.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> "SequentialAggregator":
        """Forget all cross-session state."""
        return self
