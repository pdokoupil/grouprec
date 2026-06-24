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
    AdditiveAggregator,
    AverageAggregator,
    LeastMiseryAggregator,
    MultiplicativeAggregator,
    MostPleasureAggregator,
    AVGNoMiseryAggregator,
    BordaCountAggregator,
    FAIAggregator,
)


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
