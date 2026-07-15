"""Sparse interaction matrices + lazy user-user similarity.

The point of these tests is *equivalence*: going sparse/lazy must not move a single
number, only the memory bill.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import grouprec as gr
from grouprec.data import Dataset
from grouprec.backends import EASE, ItemKNN
from grouprec.groups import LazySimilarity, similarity_matrix, build_predicate_group


@pytest.fixture(scope="module")
def data():
    return gr.make_blobs_dataset(n_users=120, n_items=90, seed=0)


@pytest.fixture(scope="module")
def generic_data():
    """A dataset whose item-item similarities are generically distinct.

    ``make_blobs_dataset`` is clustered to the point that every *binary* item column is
    identical, so every cosine similarity is exactly 1.0. Any top-k over that is an
    arbitrary choice among ~n_items ties, and a 1-ulp difference between two computation
    paths (e.g. Accelerate vs OpenBLAS) silently selects different neighbours. Anything
    that asserts on *which* neighbours were kept needs data without that degeneracy.
    """
    rng = np.random.default_rng(0)
    n_users, n_items = 80, 40
    rows = [(u, i, float(rng.integers(1, 6)))
            for u in range(n_users) for i in range(n_items) if rng.random() < 0.3]
    return Dataset(pd.DataFrame(rows, columns=["user", "item", "rating"]), name="generic")


# --------------------------------------------------------------------------- #
# Dataset.user_item_csr
# --------------------------------------------------------------------------- #
def test_csr_matches_dense(data):
    for value in ("rating", "binary"):
        dense = data.user_item_matrix(value=value)
        csr = data.user_item_csr(value=value)
        assert sparse.issparse(csr)
        assert csr.shape == dense.shape
        np.testing.assert_allclose(csr.toarray(), dense)


def test_csr_is_cached(data):
    assert data.user_item_csr("binary") is data.user_item_csr("binary")


def test_csr_rejects_bad_value(data):
    with pytest.raises(ValueError):
        data.user_item_csr(value="nope")


# --------------------------------------------------------------------------- #
# Backends: sparse fit must not change the fitted weights
# --------------------------------------------------------------------------- #
def test_ease_weights_match_dense_formulation(data):
    X = data.user_item_matrix(value="binary")
    G = X.T @ X
    d = np.diag_indices(G.shape[0])
    G[d] += 200.0
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[d] = 0.0
    fitted = EASE(reg=200.0).fit(data)
    np.testing.assert_allclose(np.asarray(fitted.W_), B, atol=1e-10)


def test_ease_score_matches_dense(data):
    users = list(data.users[:10])
    fitted = EASE(reg=200.0).fit(data)
    X = data.user_item_matrix(value="binary")
    expected = np.vstack([X[data.user_index[u]] @ np.asarray(fitted.W_) for u in users])
    np.testing.assert_allclose(fitted.score(users), expected, atol=1e-10)


def test_itemknn_similarities_match_dense_formulation(data):
    """The similarity matrix itself, before top-k, must match the dense formulation.

    Asserted on the blobs data *without* top-k: which of N tied neighbours survives
    selection is arbitrary, but the similarities themselves are not.
    """
    fitted = ItemKNN(k=None).fit(data)
    X = data.user_item_matrix(value="binary")
    norm = np.linalg.norm(X, axis=0)
    Xn = X / np.where(norm > 0, norm, 1.0)
    S = Xn.T @ Xn
    np.fill_diagonal(S, 0.0)
    np.testing.assert_allclose(np.asarray(fitted.W_), S, atol=1e-10)


@pytest.mark.parametrize("fixture", ["data", "generic_data"])
def test_itemknn_topk_keeps_the_k_largest_similarities(fixture, request):
    """Top-k keeps the k largest similarities per row, matching the dense formulation.

    Compared by the *values* kept rather than by which indices survived. When neighbours
    tie at the k-th place, the choice between them is arbitrary: a 1-ulp difference
    between two BLAS implementations picks the other one, so an index-based assertion is
    not portable (it fails ~2/3 of the time under random ulp jitter, which is how this
    surfaced on the Linux CI runners). The kept similarities are well-defined; the
    identity of a tied winner is not.
    """
    dataset = request.getfixturevalue(fixture)
    W = np.asarray(ItemKNN(k=5).fit(dataset).W_)
    X = dataset.user_item_matrix(value="binary")
    norm = np.linalg.norm(X, axis=0)
    Xn = X / np.where(norm > 0, norm, 1.0)
    S = Xn.T @ Xn
    np.fill_diagonal(S, 0.0)
    assert ((W != 0.0).sum(axis=1) <= 5).all()          # never keeps more than k
    for i in range(W.shape[0]):
        # cosine over binary profiles is >= 0, so the k largest of W are the kept ones
        np.testing.assert_allclose(np.sort(W[i])[::-1][:5],
                                   np.sort(S[i])[::-1][:5], atol=1e-10)


def test_score_handles_unknown_users(data):
    fitted = EASE(reg=200.0).fit(data)
    s = fitted.score([data.users[0], "__nobody__"])
    assert s.shape == (2, data.n_items)
    assert np.all(s[1] == 0.0)          # unknown user -> all-zero profile


# --------------------------------------------------------------------------- #
# LazySimilarity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["pearson", "cosine", "jaccard"])
def test_lazy_rows_match_dense(data, metric):
    dense = similarity_matrix(data, metric, lazy=False)
    lazy = similarity_matrix(data, metric, lazy=True)
    assert isinstance(lazy, LazySimilarity)
    assert lazy.shape == dense.shape
    for i in range(0, data.n_users, 7):
        a, b = dense[i], lazy[i]
        assert np.array_equal(np.isnan(a), np.isnan(b))
        m = ~np.isnan(a)
        np.testing.assert_allclose(a[m], b[m], atol=1e-9)


def test_lazy_diagonal_is_nan(data):
    lazy = similarity_matrix(data, "pearson", lazy=True)
    assert np.isnan(lazy[3][3])


def test_auto_stays_dense_for_small_data(data):
    assert isinstance(similarity_matrix(data, "pearson"), np.ndarray)


def test_auto_goes_lazy_past_the_budget(data):
    got = similarity_matrix(data, "pearson", max_dense_gib=1e-12)
    assert isinstance(got, LazySimilarity)


def test_lru_evicts_and_counts(data):
    lazy = similarity_matrix(data, "pearson", lazy=True, cache_rows=2)
    lazy[0], lazy[1], lazy[0]                      # 2 misses, 1 hit
    assert lazy.cache_stats()["hits"] == 1
    lazy[2], lazy[3]                               # evicts row 0 and row 1
    assert lazy.cache_stats()["cached_rows"] == 2
    assert 0 not in lazy._cache


def test_lazy_rejects_non_row_access(data):
    lazy = similarity_matrix(data, "pearson", lazy=True)
    with pytest.raises(TypeError):
        lazy[0:2]


def test_lazy_true_rejects_callable_metric(data):
    with pytest.raises(ValueError):
        similarity_matrix(data, lambda d: np.zeros((d.n_users, d.n_users)), lazy=True)


def test_builders_accept_lazy_unmodified(data):
    """The whole point: the group builders take LazySimilarity with no changes."""
    lazy = similarity_matrix(data, "pearson", lazy=True)
    rng = np.random.default_rng(0)
    g = build_predicate_group(lazy, 3, lambda r: r >= 0.0, rng, max_tries=200)
    assert g is not None and len(g) == 3
    assert lazy.cache_stats()["rows_computed"] < data.n_users   # never materialised


def test_synthetic_groups_identical_dense_vs_lazy(data):
    """Same seed + same rows => same groups, whichever path built the matrix."""
    a = gr.groups.synthetic(data, kind="similar", size=3, n=5, seed=7, sim_high=0.0)
    b = gr.groups.synthetic(data, kind="similar", size=3, n=5, seed=7, sim_high=0.0,
                            max_dense_gib=1e-12)
    assert [list(x) for x in a] == [list(x) for x in b]
