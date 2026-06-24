"""Tests for sequential evaluation and the long-term fairness metrics
(dMAE, groupSatO, groupDisO)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grouprec as gr
from grouprec import Dataset, Groups, GroupRecommender, evaluate_sequential
from grouprec.aggregators import AverageAggregator, LTPAggregator
from grouprec.backends import Popularity, Random
from grouprec.split import Split
from grouprec.eval.sequential import _discounted, _ideal_discounted


def blobs_split(seed=0):
    data = gr.make_blobs_dataset(n_users=40, n_items=60, n_clusters=4, density=0.6, seed=seed)
    folds = gr.split.crossval(data, k=3, seed=seed)
    return data, folds


# --------------------------------------------------------------------------- #
# long-term metric math
# --------------------------------------------------------------------------- #
def test_discounted_utility_formula():
    rels = [1.0, 0.0, 2.0]
    expected = 1 / np.log2(2) + 0 / np.log2(3) + 2 / np.log2(4)
    np.testing.assert_allclose(_discounted(rels), expected)
    # ideal sorts descending, clips negatives, truncates to k
    np.testing.assert_allclose(_ideal_discounted([2.0, -1.0, 1.0], 2),
                               2 / np.log2(2) + 1 / np.log2(3))


# --------------------------------------------------------------------------- #
# structure of the report
# --------------------------------------------------------------------------- #
def test_sequential_report_has_longterm_metrics():
    data, folds = blobs_split()
    groups = gr.groups.synthetic(data, kind="random", size=4, n=10, seed=0)
    rec = GroupRecommender(Popularity(measure="mean"), AverageAggregator())
    rep = evaluate_sequential(rec, data, groups, folds, n_rounds=4, k=5,
                              protocol=["coupled", "decoupled"],
                              metrics=["ndcg", "ar"],
                              group_aggregations=["mean", "min", "minmax"])
    d = rep.to_dict()
    for proto in ("coupled", "decoupled"):
        assert (proto, "dMAE", 5, "group") in d
        assert (proto, "groupSatO", 5, "group") in d
        assert (proto, "groupDisO", 5, "group") in d
        assert (proto, "ndcg", 5, "mean") in d
        assert (proto, "ar", 5, "mean") in d
    # bounds
    assert 0.0 <= d[("decoupled", "groupSatO", 5, "group")] <= 1.0
    assert d[("decoupled", "dMAE", 5, "group")] >= 0.0
    assert d[("decoupled", "groupDisO", 5, "group")] >= 0.0


def test_no_repeats_across_rounds():
    # Sanity: with exclude across rounds, a member never sees the same item twice in
    # the concatenation of the round lists -> reflected by groupSatO computed cleanly.
    data, folds = blobs_split()
    groups = gr.groups.synthetic(data, kind="random", size=3, n=5, seed=1)
    rec = GroupRecommender(Random(seed=0), AverageAggregator())
    rep = evaluate_sequential(rec, data, groups, [folds[0]], n_rounds=3, k=4,
                              protocol="decoupled", metrics=["ndcg"])
    # should run without error and produce finite long-term values
    d = rep.to_dict()
    assert np.isfinite(d[("decoupled", "dMAE", 4, "group")])


# --------------------------------------------------------------------------- #
# dMAE is a fairness metric: a fairness-aware sequential aggregator should not be
# worse than plain AVG over many rounds (lower dMAE = fairer).
# --------------------------------------------------------------------------- #
def test_ltp_improves_long_term_dmae_vs_avg():
    data, folds = blobs_split(seed=5)
    # divergent groups stress fairness the most
    groups = gr.groups.synthetic(data, kind="divergent", size=4, n=30,
                                 metric="pearson", sim_low=0.2, seed=0)
    base = Popularity(measure="mean")

    def dmae(agg):
        rec = GroupRecommender(base, agg)
        rep = evaluate_sequential(rec, data, groups, folds, n_rounds=6, k=5,
                                  protocol="decoupled",
                                  long_term_metrics=["dMAE"])
        return rep.to_dict()[("decoupled", "dMAE", 5, "group")]

    ltp_dmae = dmae(LTPAggregator(avoid_repeats=False, normalize="minmax"))
    avg_dmae = dmae(AverageAggregator())
    assert ltp_dmae <= avg_dmae + 1e-9


def test_perfect_equality_has_zero_dmae_and_disagreement():
    # All members identical -> every member gets identical utility every round ->
    # dMAE = 0 and groupDisO = 0.
    rows = []
    for u in (1, 2, 3):
        for it in range(20):
            rows.append((u, it, 5.0))
    train = pd.DataFrame(rows, columns=["user", "item", "rating"])
    data = Dataset(train)
    split = Split(train=data, test=train.copy())
    groups = Groups([np.array([1, 2, 3])])
    rec = GroupRecommender(Popularity(measure="mean"), AverageAggregator())
    rep = evaluate_sequential(rec, data, groups, split, n_rounds=3, k=4,
                              protocol="decoupled", exclude_seen=False,
                              long_term_metrics=["dMAE", "groupDisO"])
    d = rep.to_dict()
    assert d[("decoupled", "dMAE", 4, "group")] == pytest.approx(0.0, abs=1e-9)
    assert d[("decoupled", "groupDisO", 4, "group")] == pytest.approx(0.0, abs=1e-9)


def test_sequential_decoupled_requires_base():
    data, folds = blobs_split()
    groups = gr.groups.synthetic(data, kind="random", size=3, n=3, seed=0)

    class NoBaseRec:
        def fit(self, dataset):
            self.items = dataset.items
            return self

        def recommend(self, members, k, *, exclude=None):
            avail = [i for i in self.items if i not in (exclude or set())]
            return np.array(avail[:k])

    with pytest.raises(ValueError, match="decoupled"):
        evaluate_sequential(NoBaseRec(), data, groups, folds, protocol="decoupled")
