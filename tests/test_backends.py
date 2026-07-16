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


def test_implicit_als_scores_are_the_models_own():
    """The adapter must return implicit's factor dot-products, not fabricated numbers."""
    pytest.importorskip("implicit")
    d = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=0)
    rec = B.implicit_als(factors=16, iterations=3, regularization=0.1).fit(d)
    users = list(d.users[:5])
    s = rec.score(users)
    uf = np.asarray(rec.model_.user_factors)
    itf = np.asarray(rec.model_.item_factors)
    for r, u in enumerate(users):
        assert np.allclose(s[r], uf[d.user_index[u]] @ itf.T, atol=1e-4)


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


def test_lenskit_itemknn_learns_cooccurrence():
    """A real item-kNN ranks item 1 (which always co-occurs with the seen item 0)
    first among the unseen -- the same structural check the native models pass."""
    pytest.importorskip("lenskit")
    from lenskit.knn import ItemKNNScorer
    d = cooccur_dataset()
    base = B.lenskit(ItemKNNScorer(k=10)).fit(d)
    row = base.score([100])[0].copy()
    row[d.item_index[0]] = -np.inf  # exclude the seen item
    assert d.items[int(np.argmax(row))] == 1


def test_lenskit_score_block_is_the_scorers_own_output():
    """The adapter must assemble LensKit's own per-query scores into the dense block,
    not fabricate them: every item LensKit actually scored must match cell-for-cell."""
    pytest.importorskip("lenskit")
    from lenskit.knn import ItemKNNScorer
    from lenskit.data import ItemList, RecQuery
    d = cooccur_dataset()
    base = B.lenskit(ItemKNNScorer(k=10)).fit(d)
    users = list(d.users[:3])
    s = base.score(users)
    for r, u in enumerate(users):
        try:
            hist = base._lk_ds.user_row(u)
        except Exception:
            hist = None
        out = base.scorer(RecQuery(user_id=u, history_items=hist), ItemList(item_ids=d.items))
        lk = dict(zip(np.asarray(out.ids()).tolist(), np.asarray(out.scores(), float).tolist()))
        for it in d.items:
            v = lk.get(int(it), np.nan)
            if not np.isnan(v):
                assert np.isclose(s[r][d.item_index[it]], v, atol=1e-6)


def test_lenskit_implicitmf_in_pipeline():
    pytest.importorskip("lenskit")
    from lenskit.als import ImplicitMFScorer
    data = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=2)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=6, seed=0)
    rec = GroupRecommender(B.lenskit(ImplicitMFScorer(features=16)), AverageAggregator()).fit(data)
    out = rec.recommend(groups[0], k=8)
    assert len(out) == 8 and len(set(out.tolist())) == 8


def test_lenskit_userknn_in_pipeline():
    pytest.importorskip("lenskit")
    from lenskit.knn import UserKNNScorer
    data = make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=2)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=6, seed=0)
    rec = GroupRecommender(B.lenskit(UserKNNScorer(k=10)), AverageAggregator()).fit(data)
    out = rec.recommend(groups[0], k=8)
    assert len(out) == 8 and len(set(out.tolist())) == 8


# --------------------------------------------------------------------------- #
# recbole adapter (experimental)
# --------------------------------------------------------------------------- #
def test_recbole_friendly_error_when_missing():
    try:
        import recbole  # noqa: F401
        pytest.skip("recbole installed; skipping missing-dependency test")
    except ImportError:
        pass
    d = make_blobs_dataset(n_users=10, n_items=8, density=0.8, seed=0)
    rec = B.recbole(object(), object()).fit(d)
    with pytest.raises(ImportError, match="recbole"):
        rec.score([d.users[0]])


def _train_tiny_recbole_bpr(tmp_path):
    """Train a real 1-epoch BPR on a tiny, locally-written RecBole atomic dataset
    (no download). Returns (model, recbole_dataset) or raises if the installed
    RecBole cannot run here (e.g. RecBole 1.2.x is incompatible with NumPy >= 2)."""
    import pandas as pd
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.general_recommender.bpr import BPR
    from recbole.trainer import Trainer
    from recbole.utils import init_seed

    name = "rbtiny"
    folder = tmp_path / name
    folder.mkdir()
    rng = np.random.default_rng(0)
    rows = [(u, int(it), 5.0, 0.0)
            for u in range(1, 31)
            for it in rng.choice(range(1, 21), size=6, replace=False)]
    pd.DataFrame(rows, columns=["user_id:token", "item_id:token",
                                "rating:float", "timestamp:float"]) \
      .to_csv(folder / f"{name}.inter", sep="\t", index=False)

    cfg = Config(model="BPR", dataset=name, config_dict={
        "data_path": str(tmp_path), "checkpoint_dir": str(tmp_path / "ckpt"),
        "epochs": 1, "embedding_size": 8, "train_batch_size": 256,
        "device": "cpu", "show_progress": False,
        "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "RO",
                      "group_by": "user", "mode": "full"},
    })
    init_seed(cfg["seed"], cfg["reproducibility"])
    ds = create_dataset(cfg)
    train_data, valid_data, _ = data_preparation(cfg, ds)
    model = BPR(cfg, train_data.dataset).to("cpu")
    Trainer(cfg, model).fit(train_data, valid_data, show_progress=False, verbose=False)
    return model, ds


def test_recbole_bpr_scores_are_the_models_own(tmp_path):
    """End-to-end: a genuinely trained RecBole BPR, wrapped by the adapter, scores
    every item and does so faithfully -- the adapter cell for (user, item) equals the
    model's own ``full_sort_predict`` at RecBole's internal id for that item.

    Skips (rather than fails) where the installed RecBole cannot run -- notably the
    RecBole 1.2.x / NumPy >= 2 incompatibility -- so it exercises the real path on a
    compatible stack and stays quiet elsewhere."""
    pytest.importorskip("recbole")
    pytest.importorskip("torch")
    import pandas as pd
    import torch
    from grouprec import Dataset

    try:
        model, ds = _train_tiny_recbole_bpr(tmp_path)
    except Exception as e:  # numpy-2 incompatibility, GPU-only op, etc.
        pytest.skip(f"recbole cannot run in this environment: {type(e).__name__}: {e}")

    # a grouprec Dataset over exactly the ids RecBole trained on
    itoks = [t for t in ds.field2id_token[ds.iid_field] if t != "[PAD]"]
    utoks = [t for t in ds.field2id_token[ds.uid_field] if t != "[PAD]"]
    df = pd.DataFrame([(u, i) for u in utoks for i in itoks], columns=["user", "item"])
    df["rating"] = 1.0
    data = Dataset(df)

    base = B.recbole(model, ds).fit(data)
    users = utoks[:4]
    s = base.score(users)
    assert s.shape == (4, data.n_items) and np.isfinite(s).all()
    assert float((s == 0).mean()) < 0.05          # the id map populated the block

    from recbole.data.interaction import Interaction
    u0 = users[0]
    iu = ds.token2id(ds.uid_field, str(u0))
    with torch.no_grad():
        direct = model.full_sort_predict(
            Interaction({ds.uid_field: torch.tensor([iu])})
        ).view(1, -1).cpu().numpy()[0]
    it0 = data.items[0]
    iid0 = ds.token2id(ds.iid_field, str(it0))
    assert np.isclose(s[0][data.item_index[it0]], direct[iid0], atol=1e-5)
