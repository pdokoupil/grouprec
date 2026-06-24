"""Tests for evaluation: per-member metrics, group aggregations, and the
coupled/decoupled evaluate() driver on controlled data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grouprec as gr
from grouprec import Dataset, Groups, GroupRecommender, evaluate
from grouprec.aggregators import AverageAggregator
from grouprec.backends import Popularity
from grouprec.split import Split
from grouprec.eval.metrics import BASE_METRICS, GROUP_AGGREGATIONS


# --------------------------------------------------------------------------- #
# per-member base metrics
# --------------------------------------------------------------------------- #
def test_base_metrics_hand_values():
    rec = [10, 20, 30, 40]
    relevant = {20, 40}
    gains = {20: 1.0, 40: 1.0}
    k = 4
    assert BASE_METRICS["recall"](rec, gains, relevant, k) == 1.0
    assert BASE_METRICS["precision"](rec, gains, relevant, k) == 0.5
    assert BASE_METRICS["hr"](rec, gains, relevant, k) == 1.0
    assert BASE_METRICS["brecall"](rec, gains, relevant, k) == 1.0
    np.testing.assert_allclose(BASE_METRICS["dfh"](rec, gains, relevant, k), 1 / np.log2(3))
    np.testing.assert_allclose(BASE_METRICS["mrr"](rec, gains, relevant, k), np.mean([1 / 2, 1 / 4]))
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    np.testing.assert_allclose(BASE_METRICS["ndcg"](rec, gains, relevant, k), dcg / idcg)


def test_metrics_empty_relevant_are_zero():
    for name in ["ndcg", "recall", "hr", "dfh", "mrr", "brecall"]:
        assert BASE_METRICS[name]([1, 2], {}, set(), 2) == 0.0


def test_metric_respects_cutoff():
    rec = [1, 2, 3, 4]
    relevant = {4}
    assert BASE_METRICS["hr"](rec, {}, relevant, 2) == 0.0  # item 4 beyond k=2
    assert BASE_METRICS["hr"](rec, {}, relevant, 4) == 1.0


# --------------------------------------------------------------------------- #
# group aggregations (the fairness lens)
# --------------------------------------------------------------------------- #
def test_group_aggregations_hand_values():
    v = np.array([0.0, 0.5, 1.0])
    assert GROUP_AGGREGATIONS["mean"](v) == 0.5
    assert GROUP_AGGREGATIONS["min"](v) == 0.0
    assert GROUP_AGGREGATIONS["max"](v) == 1.0
    assert GROUP_AGGREGATIONS["minmax"](v) == 0.0
    np.testing.assert_allclose(GROUP_AGGREGATIONS["std"](v), np.std(v))
    np.testing.assert_allclose(GROUP_AGGREGATIONS["jain"](v), 0.6)
    np.testing.assert_allclose(GROUP_AGGREGATIONS["zero"](v), 1 / 3)


def test_jain_and_minmax_perfect_equality():
    v = np.array([0.7, 0.7, 0.7])
    assert GROUP_AGGREGATIONS["jain"](v) == pytest.approx(1.0)
    assert GROUP_AGGREGATIONS["minmax"](v) == pytest.approx(1.0)


def test_zero_aggregation_is_zrecall():
    v = np.array([0.0, 0.3, 0.0, 0.9])
    assert GROUP_AGGREGATIONS["zero"](v) == 0.5  # half the members got nothing


# --------------------------------------------------------------------------- #
# controlled evaluate() fixtures
# --------------------------------------------------------------------------- #
def controlled_split_and_data():
    # train: item1 most popular (3), item2 (2), item3 (1), item4 (1)
    train = pd.DataFrame(
        [(10, 1), (11, 1), (12, 1), (10, 2), (11, 2), (10, 3), (10, 4)],
        columns=["user", "item"],
    )
    train["rating"] = 5.0
    # test: user1 likes item1, user2 likes item2
    test = pd.DataFrame([(1, 1, 5.0), (2, 2, 5.0)], columns=["user", "item", "rating"])
    full = pd.concat([train, test], ignore_index=True)
    return Split(train=Dataset(train), test=test), Dataset(full)


def test_evaluate_coupled_known_values():
    split, data = controlled_split_and_data()
    groups = Groups([np.array([1, 2])])
    rec = GroupRecommender(Popularity(measure="count"), AverageAggregator())
    rep = evaluate(rec, data, groups, split, k=2, protocol="coupled",
                   metrics=["ndcg", "recall", "hr"],
                   group_aggregations=["mean", "min", "minmax"],
                   exclude_seen=False)
    d = rep.to_dict()
    # both members' relevant item is recommended within top-2 -> perfect recall/hr
    assert d[("coupled", "recall", 2, "mean")] == pytest.approx(1.0)
    assert d[("coupled", "hr", 2, "mean")] == pytest.approx(1.0)
    # nDCG: item1@rank0 -> 1.0 ; item2@rank1 -> 1/log2(3)
    assert d[("coupled", "ndcg", 2, "min")] == pytest.approx(1 / np.log2(3))
    assert d[("coupled", "ndcg", 2, "mean")] == pytest.approx((1.0 + 1 / np.log2(3)) / 2)


def test_evaluate_decoupled_isolates_aggregator():
    split, data = controlled_split_and_data()
    groups = Groups([np.array([1, 2])])
    rec = GroupRecommender(Popularity(measure="count"), AverageAggregator())
    rep = evaluate(rec, data, groups, split, k=2, protocol="decoupled",
                   metrics=["ndcg", "recall"], group_aggregations=["mean"],
                   exclude_seen=False)
    d = rep.to_dict()
    # popularity gives identical scores to all members; AVG reproduces that exact
    # ranking -> the aggregator perfectly matches each member's ideal top-2.
    assert d[("decoupled", "ndcg", 2, "mean")] == pytest.approx(1.0)
    assert d[("decoupled", "recall", 2, "mean")] == pytest.approx(1.0)


def test_evaluate_both_protocols_at_once():
    split, data = controlled_split_and_data()
    groups = Groups([np.array([1, 2])])
    rec = GroupRecommender(Popularity(), AverageAggregator())
    rep = evaluate(rec, data, groups, split, k=2, protocol=["coupled", "decoupled"],
                   metrics=["ndcg"], group_aggregations=["mean"], exclude_seen=False)
    protos = {r["protocol"] for r in rep.records}
    assert protos == {"coupled", "decoupled"}


def test_decoupled_requires_base():
    split, data = controlled_split_and_data()
    groups = Groups([np.array([1, 2])])

    class NoBaseRec:  # profile-first / deep-model stand-in: no .base.score
        def fit(self, dataset):
            self.items = dataset.items
            return self

        def recommend(self, members, k, *, exclude=None):
            return self.items[:k]

    with pytest.raises(ValueError, match="decoupled"):
        evaluate(NoBaseRec(), data, groups, split, protocol="decoupled")


def test_coupled_works_without_base():
    split, data = controlled_split_and_data()
    groups = Groups([np.array([1, 2])])

    class NoBaseRec:
        def fit(self, dataset):
            self.items = dataset.items
            return self

        def recommend(self, members, k, *, exclude=None):
            return self.items[:k]  # always [1, 2, ...]

    rep = evaluate(NoBaseRec(), data, groups, split, k=2, protocol="coupled",
                   metrics=["recall"], group_aggregations=["mean"], exclude_seen=False)
    assert rep.to_dict()[("coupled", "recall", 2, "mean")] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# integration / report
# --------------------------------------------------------------------------- #
def test_evaluate_crossval_and_report_shapes():
    data = gr.make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=0)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=15, seed=0)
    folds = gr.split.crossval(data, k=3, seed=0)
    rec = GroupRecommender(Popularity(measure="mean"), AverageAggregator())
    rep = evaluate(rec, data, groups, folds, k=10,
                   protocol=["coupled", "decoupled"],
                   metrics=["ndcg@10", "recall@5", "hr@10"],
                   group_aggregations=["mean", "min", "minmax", "jain", "zero"])
    df = rep.to_frame()
    assert set(df["protocol"]) == {"coupled", "decoupled"}
    assert set(df["k"]) == {5, 10}
    assert (df["value"].between(0, 1) | df["value"].isna()).all()
    piv = rep.pivot()
    assert "coupled" in piv.columns and "decoupled" in piv.columns


def test_unknown_metric_and_aggregation_raise():
    split, data = controlled_split_and_data()
    groups = Groups([np.array([1, 2])])
    rec = GroupRecommender(Popularity(), AverageAggregator())
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(rec, data, groups, split, metrics=["bogus"])
    with pytest.raises(ValueError, match="unknown aggregation"):
        evaluate(rec, data, groups, split, metrics=["ndcg"], group_aggregations=["bogus"])
