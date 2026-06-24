"""Tests for backends: native EASE/ItemKNN and the implicit/lenskit/recbole adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grouprec as gr
from grouprec import Dataset, GroupRecommender, make_blobs_dataset
from grouprec.aggregators import AverageAggregator
from grouprec import backends as B


def cooccur_dataset():
    """Items 0 and 1 always co-occur; query user 100 has only item 0 -> a good
    item-based model should rank item 1 first among the unseen."""
    rows = [(u, 0) for u in range(20)] + [(u, 1) for u in range(20)]
    rows += [(u, 2) for u in range(20, 30)] + [(u, 3) for u in range(30, 40)]
    rows.append((100, 0))
    df = pd.DataFrame(rows, columns=["user", "item"])
    df["rating"] = 5.0
    return Dataset(df)


# --------------------------------------------------------------------------- #
# native EASE / ItemKNN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Model", [B.EASE, B.ItemKNN])
def test_native_models_shape_and_protocol(Model):
    d = make_blobs_dataset(n_users=30, n_items=25, density=0.6, seed=0)
    model = Model().fit(d)
    assert isinstance(model, B.BaseRecommender)
    s = model.score(d.users[:3])
    assert s.shape == (3, d.n_items)
    assert np.isfinite(s).all()


@pytest.mark.parametrize("Model", [B.EASE, B.ItemKNN])
def test_native_models_learn_cooccurrence(Model):
    d = cooccur_dataset()
    model = Model().fit(d)
    row = model.score([100])[0].copy()
    row[d.item_index[0]] = -np.inf  # exclude the seen item
    assert d.items[int(np.argmax(row))] == 1  # the co-occurring partner


def test_ease_zero_diagonal():
    d = cooccur_dataset()
    ease = B.EASE().fit(d)
    assert np.allclose(np.diag(ease.W_), 0.0)


def test_itemknn_topk_truncation():
    d = make_blobs_dataset(n_users=30, n_items=20, density=0.7, seed=1)
    knn = B.ItemKNN(k=3).fit(d)
    # each item keeps at most k non-zero neighbours
    assert (np.count_nonzero(knn.W_, axis=1) <= 3).all()


def test_native_model_in_pipeline_and_eval():
    data = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=0)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=10, seed=0)
    split = gr.split.random_split(data, test_frac=0.2, seed=0)
    rec = GroupRecommender(B.EASE(), AverageAggregator())
    rep = gr.evaluate(rec, data, groups, split, k=10,
                      protocol=["coupled", "decoupled"], metrics=["ndcg", "recall"])
    assert (rep.to_frame()["value"].between(0, 1) | rep.to_frame()["value"].isna()).all()


# --------------------------------------------------------------------------- #
# implicit adapter
# --------------------------------------------------------------------------- #
def test_implicit_als_adapter():
    pytest.importorskip("implicit")
    d = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=0)
    rec = B.implicit_als(factors=16, iterations=3, regularization=0.1)
    rec.fit(d)
    s = rec.score(d.users[:5])
    assert s.shape == (5, d.n_items)
    assert np.isfinite(s).all()


def test_implicit_in_group_pipeline():
    pytest.importorskip("implicit")
    data = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=1)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=8, seed=0)
    rec = GroupRecommender(B.implicit_als(factors=16, iterations=3), AverageAggregator()).fit(data)
    out = rec.recommend(groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10


def test_implicit_bad_kind():
    with pytest.raises(ValueError):
        B.ImplicitRecommender(kind="bogus")


# --------------------------------------------------------------------------- #
# lenskit adapter
# --------------------------------------------------------------------------- #
def test_lenskit_itemknn_adapter():
    pytest.importorskip("lenskit")
    from lenskit.knn import ItemKNNScorer
    d = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=0)
    rec = B.lenskit(ItemKNNScorer(k=10)).fit(d)
    s = rec.score(d.users[:4])
    assert s.shape == (4, d.n_items)
    assert np.isfinite(s).all()  # NaN (unseen-by-knn) filled


def test_lenskit_implicitmf_in_pipeline():
    pytest.importorskip("lenskit")
    from lenskit.als import ImplicitMFScorer
    data = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=2)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=6, seed=0)
    rec = GroupRecommender(B.lenskit(ImplicitMFScorer(features=16)), AverageAggregator()).fit(data)
    out = rec.recommend(groups[0], k=8)
    assert len(out) == 8 and len(set(out.tolist())) == 8


# --------------------------------------------------------------------------- #
# recbole adapter (experimental; only the friendly-error path without recbole)
# --------------------------------------------------------------------------- #
def test_recbole_friendly_error_when_missing():
    if pytest.importorskip:  # placeholder to keep structure clear
        pass
    try:
        import recbole  # noqa: F401
        pytest.skip("recbole installed; skipping missing-dependency test")
    except ImportError:
        pass
    d = make_blobs_dataset(n_users=10, n_items=8, density=0.8, seed=0)
    rec = B.recbole(object(), object()).fit(d)
    with pytest.raises(ImportError, match="recbole"):
        rec.score([d.users[0]])
