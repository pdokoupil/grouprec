"""Adapters to external single-user recommender frameworks.

These are the *interop* layer: instead of reimplementing base recommenders, we wrap
the established ones so users inherit their full model zoos.

* ``implicit_als`` / ``implicit_bpr`` -- the ``implicit`` package (ALS/BPR). ALS is
  what the GFAR / Coupled-Decoupled papers used.
* ``lenskit(scorer)`` -- any LensKit scorer component (ImplicitMF, BiasedMF, ItemKNN,
  EASE, SLIM, UserKNN, ...).
* ``recbole(model, dataset)`` -- a trained RecBole model (experimental).

Each returns an object satisfying :class:`~grouprec.backends.BaseRecommender`:
``fit(dataset)`` then ``score(users, items=None) -> (n_users, n_items)`` in
``dataset.items`` order.
"""

from __future__ import annotations

import numpy as np

from ..data import Dataset
from ._optional import optional_import


def _fill_nan_rowmin(s: np.ndarray) -> np.ndarray:
    """Replace NaN (e.g. items a kNN won't score) with each row's finite min, so
    such items rank last rather than breaking aggregation."""
    out = s.copy()
    for r in range(out.shape[0]):
        row = out[r]
        nan = np.isnan(row)
        if nan.any():
            finite = row[~nan]
            row[nan] = finite.min() if finite.size else 0.0
    return out


class _FittedBase:
    dataset_: Dataset | None

    def _check_fitted(self) -> None:
        if getattr(self, "dataset_", None) is None:
            raise RuntimeError(f"{type(self).__name__} must be fit() before scoring.")

    def _select(self, s: np.ndarray, items) -> np.ndarray:
        if items is None:
            return s
        cols = np.array([self.dataset_.item_index[i] for i in items], dtype=np.int64)
        return s[:, cols]


# --------------------------------------------------------------------------- #
# implicit (ALS / BPR)
# --------------------------------------------------------------------------- #
class ImplicitRecommender(_FittedBase):
    """Adapter over the ``implicit`` package's matrix-factorization models."""

    cite_key = "implicit"  # we can't resolve the transitive algorithm; cite the framework

    def __init__(self, kind: str = "als", *, use_ratings: bool = False,
                 alpha: float = 1.0, **model_kwargs) -> None:
        if kind not in ("als", "bpr"):
            raise ValueError("kind must be 'als' or 'bpr'.")
        self.kind = kind
        self.use_ratings = use_ratings
        self.alpha = alpha
        self.model_kwargs = model_kwargs
        self.dataset_ = None
        self.model_ = None

    def fit(self, dataset: Dataset) -> "ImplicitRecommender":
        implicit = optional_import("implicit", "implicit", "The implicit backend")
        from scipy.sparse import csr_matrix

        self.dataset_ = dataset
        ui = dataset.interactions["user"].map(dataset.user_index).to_numpy()
        ij = dataset.interactions["item"].map(dataset.item_index).to_numpy()
        if self.use_ratings and dataset.has_ratings:
            vals = dataset.interactions["rating"].to_numpy(dtype=float)
        else:
            vals = np.ones(len(ui))
        confidence = 1.0 + self.alpha * vals  # implicit-feedback confidence weighting
        mat = csr_matrix((confidence, (ui, ij)),
                         shape=(dataset.n_users, dataset.n_items)).astype("float32")
        if self.kind == "als":
            self.model_ = implicit.als.AlternatingLeastSquares(**self.model_kwargs)
        else:
            self.model_ = implicit.bpr.BayesianPersonalizedRanking(**self.model_kwargs)
        self.model_.fit(mat, show_progress=False)
        return self

    def score(self, users, items=None) -> np.ndarray:
        self._check_fitted()
        uf = np.asarray(self.model_.user_factors)
        itf = np.asarray(self.model_.item_factors)
        rows = []
        for u in users:
            ui = self.dataset_.user_index.get(u)
            rows.append(uf[ui] @ itf.T if ui is not None else np.zeros(self.dataset_.n_items))
        return self._select(np.vstack(rows), items)


def implicit_als(*, use_ratings: bool = False, alpha: float = 1.0, **kwargs) -> ImplicitRecommender:
    """ALS backend (e.g. ``implicit_als(factors=64, regularization=0.05)``)."""
    return ImplicitRecommender("als", use_ratings=use_ratings, alpha=alpha, **kwargs)


def implicit_bpr(*, use_ratings: bool = False, alpha: float = 1.0, **kwargs) -> ImplicitRecommender:
    """BPR backend (e.g. ``implicit_bpr(factors=64)``)."""
    return ImplicitRecommender("bpr", use_ratings=use_ratings, alpha=alpha, **kwargs)


# --------------------------------------------------------------------------- #
# LensKit
# --------------------------------------------------------------------------- #
class LensKitRecommender(_FittedBase):
    """Adapter wrapping any LensKit scorer component (LensKit >= 2025)."""

    cite_key = "lenskit"  # cite the framework (transitive algorithm not resolved)

    def __init__(self, scorer) -> None:
        self.scorer = scorer
        self.dataset_ = None
        self._lk_ds = None

    def fit(self, dataset: Dataset) -> "LensKitRecommender":
        lkdata = optional_import("lenskit.data", "lenskit", "The lenskit backend")
        self.dataset_ = dataset
        df = dataset.interactions.rename(columns={"user": "user_id", "item": "item_id"})
        cols = ["user_id", "item_id"] + (["rating"] if dataset.has_ratings else [])
        self._lk_ds = lkdata.from_interactions_df(df[cols])
        trained = self.scorer.train(self._lk_ds)
        if trained is not None:  # some components return the trained instance
            self.scorer = trained
        return self

    def score(self, users, items=None) -> np.ndarray:
        self._check_fitted()
        from lenskit.data import ItemList, RecQuery

        target = self.dataset_.items if items is None else np.asarray(list(items))
        cand = ItemList(item_ids=target)
        rows = []
        for u in users:
            try:
                hist = self._lk_ds.user_row(u)
            except Exception:
                hist = None
            out = self.scorer(RecQuery(user_id=u, history_items=hist), cand)
            ids = np.asarray(out.ids())
            sc = np.asarray(out.scores(), dtype=float)
            lookup = dict(zip(ids.tolist(), sc.tolist()))
            rows.append(np.array([lookup.get(int(it), np.nan) for it in target]))
        return _fill_nan_rowmin(np.vstack(rows))


def lenskit(scorer) -> LensKitRecommender:
    """Wrap a LensKit scorer, e.g. ``gr.backends.lenskit(ImplicitMFScorer(features=64))``."""
    return LensKitRecommender(scorer)


# --------------------------------------------------------------------------- #
# RecBole (experimental)
# --------------------------------------------------------------------------- #
class RecBoleRecommender(_FittedBase):
    """Adapter wrapping a **trained** RecBole model (experimental).

    Pass the ``model`` and RecBole ``dataset`` (e.g. from
    ``recbole.quick_start.load_data_and_model``). Scoring uses the model's
    ``full_sort_predict`` and maps RecBole's internal ids to this library's item
    space. ``fit(dataset)`` only records the target item ordering (RecBole trains
    its own model separately).
    """

    cite_key = "recbole"  # cite the framework (transitive algorithm not resolved)

    def __init__(self, model, recbole_dataset) -> None:
        self.model = model
        self.recbole_dataset = recbole_dataset
        self.dataset_ = None

    def fit(self, dataset: Dataset) -> "RecBoleRecommender":
        self.dataset_ = dataset
        return self

    def score(self, users, items=None) -> np.ndarray:
        self._check_fitted()
        optional_import("recbole", "recbole", "The recbole backend")
        import torch
        from recbole.data.interaction import Interaction

        rb = self.recbole_dataset
        uid_field, iid_field = rb.uid_field, rb.iid_field
        device = next(self.model.parameters()).device

        internal_u = [rb.token2id(uid_field, str(u)) for u in users]
        inter = Interaction({uid_field: torch.tensor(internal_u)}).to(device)
        with torch.no_grad():
            scores = self.model.full_sort_predict(inter).view(len(internal_u), -1)
        scores = scores.detach().cpu().numpy()

        # map RecBole internal item ids -> this library's item columns
        n = self.dataset_.n_items
        out = np.zeros((len(users), n))
        for it, col in self.dataset_.item_index.items():
            iid = rb.token2id(iid_field, str(it))
            if 0 <= iid < scores.shape[1]:
                out[:, col] = scores[:, iid]
        return self._select(out, items)


def recbole(model, recbole_dataset) -> RecBoleRecommender:
    """Wrap a trained RecBole model (experimental)."""
    return RecBoleRecommender(model, recbole_dataset)


__all__ = [
    "ImplicitRecommender",
    "LensKitRecommender",
    "RecBoleRecommender",
    "implicit_als",
    "implicit_bpr",
    "lenskit",
    "recbole",
]
