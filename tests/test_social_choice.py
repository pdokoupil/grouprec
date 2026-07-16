"""Validate the vectorized social-choice aggregators against the reference pandas
implementations from ``aggregators_new.py``.

The reference functions are copied here (lightly adapted) as an executable oracle,
so the tests assert *bit-for-bit selection-order equality* between the numpy core
and the original code on randomized inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from grouprec.aggregators import (
    normalize_mgains,
    AdditiveAggregator,
    AverageAggregator,
    WeightedAverageAggregator,
    LeastMiseryAggregator,
    MultiplicativeAggregator,
    MostPleasureAggregator,
    AVGNoMiseryAggregator,
    BordaCountAggregator,
    FAIAggregator,
    get,
)


def test_weighted_average_uniform_equals_average():
    rng = np.random.default_rng(0)
    rm = rng.random((3, 20))
    a = AverageAggregator().aggregate(rm, 20)
    b = WeightedAverageAggregator().aggregate(rm, 20)
    c = WeightedAverageAggregator(member_weights=[2.0, 2.0, 2.0]).aggregate(rm, 20)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)            # any uniform scale == plain mean
    assert get("wAVG") is not None         # registered in the factory


def test_weighted_average_weights_shift_ranking():
    # member 0 alone prefers item 0; member 1 alone prefers item 1
    rm = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    top0 = WeightedAverageAggregator(member_weights=[0.9, 0.1]).aggregate(rm, 1)[0]
    top1 = WeightedAverageAggregator(member_weights=[0.1, 0.9]).aggregate(rm, 1)[0]
    assert top0 == 0 and top1 == 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def matrix_to_df(rm: np.ndarray) -> pd.DataFrame:
    """Long-format (user, item, predicted_rating) df; item id == column index."""
    n_members, n_items = rm.shape
    rows = []
    for u in range(n_members):
        for i in range(n_items):
            rows.append((u, i, rm[u, i]))
    return pd.DataFrame(rows, columns=["user", "item", "predicted_rating"])


def random_matrix(rng, n_members, n_items, low=0.5, high=5.0):
    return rng.uniform(low, high, size=(n_members, n_items))


# --------------------------------------------------------------------------- #
# reference oracles (ported from aggregators_new.py)
# --------------------------------------------------------------------------- #
def ref_simple(df, k, how):
    agg = {"sum": df.groupby("item").sum,
           "mean": df.groupby("item").mean,
           "min": df.groupby("item").min,
           "prod": df.groupby("item").prod,
           "max": df.groupby("item").max}[how]()
    agg = agg.sort_values(by="predicted_rating", ascending=False).reset_index()
    return list(agg.head(k)["item"])


def ref_avgnm(df, k, threshold):
    allowed = df.groupby("item", as_index=False).min()
    allowed = allowed.loc[allowed["predicted_rating"] > threshold]
    allowed = allowed["item"].tolist()
    if len(allowed) == 0:
        return []
    ordered = df.groupby("item").mean()
    ordered = ordered.sort_values(by="predicted_rating", ascending=False).reset_index()
    collected = ordered[ordered["item"].isin(allowed)]
    return list(collected.head(k)["item"])


def ref_bdc(df, k):
    local = df.copy()
    local["borda_score"] = 0.0
    for uid in df["user"].unique():
        per_user = local.loc[local.user == uid]
        score = rankdata(per_user["predicted_rating"].values, method="min")
        local.loc[per_user.index, "borda_score"] = score
    agg = local.groupby("item").sum()
    # stable sort so ties break by ascending item id -- matching the library's
    # documented tie-break convention (pandas defaults to an unstable quicksort).
    agg = agg.sort_values(by="borda_score", ascending=False, kind="stable").reset_index()
    return list(agg.head(k)["item"])


def ref_fai(df, k):
    selected = []
    users = df["user"].unique()
    for i in range(int(k)):
        ui = i % len(users)
        cur = df.loc[df["user"] == users[ui]]
        cur = cur.sort_values(by="predicted_rating", ascending=False).reset_index()
        cur = cur.loc[~cur["item"].isin(selected)]
        selected.append(cur["item"].iloc[0])
    return selected


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
SHAPES = [(2, 10), (3, 25), (4, 50), (5, 7), (8, 100)]
KS = [1, 5, 20]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize(
    "agg,how",
    [
        (AdditiveAggregator(), "sum"),
        (AverageAggregator(), "mean"),
        (LeastMiseryAggregator(), "min"),
        (MultiplicativeAggregator(), "prod"),
        (MostPleasureAggregator(), "max"),
    ],
)
def test_simple_aggregators_match_reference(shape, k, agg, how):
    rng = np.random.default_rng(hash((shape, k, how)) % (2**32))
    rm = random_matrix(rng, *shape)
    k = min(k, shape[1])
    got = list(agg.aggregate(rm, k))
    expected = ref_simple(matrix_to_df(rm), k, how)
    assert got == expected


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("k", KS)
def test_avgnm_matches_reference(shape, k):
    rng = np.random.default_rng(hash((shape, k, "avgnm")) % (2**32))
    rm = random_matrix(rng, *shape)
    k = min(k, shape[1])
    threshold = 2.0
    got = list(AVGNoMiseryAggregator(threshold=threshold).aggregate(rm, k))
    expected = ref_avgnm(matrix_to_df(rm), k, threshold)
    assert got == expected


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("k", KS)
def test_bdc_matches_reference(shape, k):
    rng = np.random.default_rng(hash((shape, k, "bdc")) % (2**32))
    rm = random_matrix(rng, *shape)
    k = min(k, shape[1])
    got = list(BordaCountAggregator().aggregate(rm, k))
    expected = ref_bdc(matrix_to_df(rm), k)
    assert got == expected


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("k", KS)
def test_fai_matches_reference(shape, k):
    rng = np.random.default_rng(hash((shape, k, "fai")) % (2**32))
    rm = random_matrix(rng, *shape)
    k = min(k, shape[1])
    got = list(FAIAggregator(start=0).aggregate(rm, k))
    expected = ref_fai(matrix_to_df(rm), k)
    assert got == expected


# --------------------------------------------------------------------------- #
# behavioural / invariant tests
# --------------------------------------------------------------------------- #
def test_returns_no_duplicates_and_correct_length():
    rng = np.random.default_rng(0)
    rm = random_matrix(rng, 4, 30)
    for agg in [
        AdditiveAggregator(), AverageAggregator(), LeastMiseryAggregator(),
        MultiplicativeAggregator(), MostPleasureAggregator(),
        BordaCountAggregator(), FAIAggregator(),
    ]:
        out = agg.aggregate(rm, 10)
        assert len(out) == 10
        assert len(set(out.tolist())) == 10


def test_exclude_is_respected():
    rng = np.random.default_rng(1)
    rm = random_matrix(rng, 3, 20)
    exclude = {0, 1, 2, 3, 4}
    for agg in [
        AdditiveAggregator(), AverageAggregator(), LeastMiseryAggregator(),
        MostPleasureAggregator(), BordaCountAggregator(), FAIAggregator(),
    ]:
        out = set(agg.aggregate(rm, 10, exclude=exclude).tolist())
        assert out.isdisjoint(exclude)


def test_k_larger_than_items_returns_all():
    rm = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    out = AverageAggregator().aggregate(rm, 99)
    assert sorted(out.tolist()) == [0, 1, 2]


def test_avg_least_misery_disagree_on_outlier():
    # item 0: high average but a low-scoring member; item 1: both lukewarm.
    rm = np.array([[5.0, 3.0], [2.0, 3.0]])
    assert AverageAggregator().aggregate(rm, 1)[0] == 0  # mean 3.5 > 3.0
    assert LeastMiseryAggregator().aggregate(rm, 1)[0] == 1  # min 2.0 < 3.0 -> safe item


def test_paradigm_attribute():
    assert AverageAggregator().paradigm == "results"


def test_score_items_matches_ranking_for_reductions():
    rng = np.random.default_rng(1)
    rm = rng.random((3, 25))
    cases = {
        AdditiveAggregator(): rm.sum(0),
        AverageAggregator(): rm.mean(0),
        WeightedAverageAggregator(member_weights=[0.6, 0.3, 0.1]): (
            np.array([0.6, 0.3, 0.1])[:, None] * rm).sum(0),
        LeastMiseryAggregator(): rm.min(0),
        MultiplicativeAggregator(): rm.prod(0),
        MostPleasureAggregator(): rm.max(0),
    }
    for agg, expected in cases.items():
        util = agg.score_items(rm)
        assert agg.produces_item_scores is True
        assert util.shape == (25,)
        np.testing.assert_allclose(util, expected)
        # the ranking the pipeline sorts on is exactly score_items (descending, stable)
        np.testing.assert_array_equal(
            agg.aggregate(rm, 25), np.argsort(-util, kind="stable"))


def test_score_items_honors_exclude():
    rm = np.random.default_rng(2).random((2, 10))
    util = AverageAggregator().score_items(rm, exclude=[3, 7])
    assert util[3] == -np.inf and util[7] == -np.inf
    assert np.isfinite(util[[0, 1, 2, 4, 5, 6, 8, 9]]).all()


def test_selection_based_aggregators_have_no_item_scores():
    rm = np.random.default_rng(3).random((3, 12))
    for agg in [FAIAggregator(), get("EPFuzzDA"), get("GFAR")]:
        assert agg.produces_item_scores is False
        with pytest.raises(NotImplementedError):
            agg.score_items(rm)


# --------------------------------------------------------------------------- #
# normalize_mgains (public: the fairness aggregators assume commensurable scores)
# --------------------------------------------------------------------------- #
def test_normalize_mgains_is_public_and_row_independent():
    import grouprec as gr

    assert gr.aggregators.normalize_mgains is normalize_mgains
    assert "none" in gr.aggregators.NORMALIZE_METHODS

    rm = np.array([[1.0, 2.0, 3.0], [10.0, 30.0, 20.0]])       # different per-member scales
    out = normalize_mgains(rm, "minmax")
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0], [0.0, 1.0, 0.5]])

    # every method rescales each row on its own, and leaves the ranking within a row intact
    for method in gr.aggregators.NORMALIZE_METHODS:
        got = normalize_mgains(rm, method)
        assert got.shape == rm.shape
        for src, dst in zip(rm, got):
            assert (np.argsort(src) == np.argsort(dst)).all(), method


def test_normalize_mgains_none_is_a_noop_and_unknown_raises():
    rm = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert normalize_mgains(rm, None) is rm and normalize_mgains(rm, "none") is rm
    with pytest.raises(ValueError, match="unknown normalize method"):
        normalize_mgains(rm, "nope")


def test_normalize_mgains_handles_a_flat_row():
    out = normalize_mgains(np.array([[5.0, 5.0, 5.0]]), "minmax")
    np.testing.assert_array_equal(out, [[0.0, 0.0, 0.0]])       # no divide-by-zero
