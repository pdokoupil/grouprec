"""Tests for the benchmark runner and group-level (bridge) evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grouprec as gr
from grouprec import Dataset, GroupRecommender, Groups, benchmark
from grouprec.bench import BenchmarkTask
from grouprec.aggregators import AverageAggregator, get
from grouprec.backends import Popularity
from grouprec.eval import evaluate_grouplevel
from grouprec.split import Split, random_split


def member_task(name, seed):
    data = gr.make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=seed)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=10, seed=0)
    split = random_split(data, test_frac=0.2, seed=0)
    return BenchmarkTask(name=name, data=data, groups=groups, splits=split)


def agg_rec(name):
    return GroupRecommender(Popularity(measure="mean"), get(name))


# --------------------------------------------------------------------------- #
# member-level benchmark grid
# --------------------------------------------------------------------------- #
def test_benchmark_grid_shapes_and_csv(tmp_path):
    tasks = [member_task("ml-ish", 0), member_task("kgrec-ish", 1)]
    recs = {"AVG": agg_rec("AVG"), "GFAR": agg_rec("GFAR"), "LMS": agg_rec("LMS")}
    res = benchmark(recs, tasks, protocols=["coupled", "decoupled"],
                    metrics=["ndcg@10", "recall@10"], group_aggregations=["mean", "min"])
    df = res.to_frame()
    assert set(df["dataset"]) == {"ml-ish", "kgrec-ish"}
    assert set(df["recommender"]) == {"AVG", "GFAR", "LMS"}
    assert set(df["protocol"]) == {"coupled", "decoupled"}
    # CSV round-trips
    p = tmp_path / "lb.csv"
    res.to_csv(p)
    assert pd.read_csv(p).shape[0] == df.shape[0]


def test_benchmark_leaderboard_and_best():
    tasks = [member_task("d1", 0)]
    recs = {"AVG": agg_rec("AVG"), "GFAR": agg_rec("GFAR")}
    res = benchmark(recs, tasks, protocols=["coupled"], metrics=["ndcg@10"],
                    group_aggregations=["mean"])
    lb = res.leaderboard("ndcg", k=10, aggregation="mean", protocol="coupled")
    assert set(lb.index) == {"AVG", "GFAR"} and "d1" in lb.columns
    best = res.best("ndcg", aggregation="mean", protocol="coupled")
    assert list(best.index)[0] in {"AVG", "GFAR"}


def test_decoupled_autoskipped_for_baseless_recommender():
    class NoBaseRec:
        paradigm = "profile"

        def fit(self, dataset):
            self.items = dataset.items
            return self

        def recommend(self, members, k, *, exclude=None):
            avail = [int(i) for i in self.items if i not in (exclude or set())]
            return np.array(avail[:k])

    tasks = [member_task("d1", 0)]
    recs = {"deepish": NoBaseRec(), "AVG": agg_rec("AVG")}
    res = benchmark(recs, tasks, protocols=["coupled", "decoupled"], metrics=["ndcg@10"])
    df = res.to_frame()
    # base-less recommender: coupled only; aggregator: both
    assert set(df[df.recommender == "deepish"]["protocol"]) == {"coupled"}
    assert set(df[df.recommender == "AVG"]["protocol"]) == {"coupled", "decoupled"}


def test_benchmark_on_error_skip():
    class Boom:
        paradigm = "results"

        def fit(self, dataset):
            return self

        def recommend(self, *a, **k):
            raise RuntimeError("boom")

    tasks = [member_task("d1", 0)]
    res = benchmark({"boom": Boom(), "AVG": agg_rec("AVG")}, tasks,
                    protocols=["coupled"], metrics=["ndcg@10"], on_error="skip")
    assert set(res.to_frame()["recommender"]) == {"AVG"}
    with pytest.raises(RuntimeError):
        benchmark({"boom": Boom()}, tasks, protocols=["coupled"],
                  metrics=["ndcg@10"], on_error="raise")


def test_benchmark_factories_get_fresh_instances():
    tasks = [member_task("d1", 0)]
    res = benchmark({"AVG": lambda: agg_rec("AVG")}, tasks,
                    protocols=["coupled"], metrics=["ndcg@10"])
    assert "AVG" in set(res.to_frame()["recommender"])


def test_benchmark_sequential_long_term():
    tasks = [member_task("d1", 0)]
    tasks[0].sequential = True
    tasks[0].n_rounds = 3
    res = benchmark({"AVG": agg_rec("AVG")}, tasks, protocols=["coupled"],
                    metrics=["ndcg"], group_aggregations=["mean"])
    assert "dMAE" in set(res.to_frame()["metric"])


# --------------------------------------------------------------------------- #
# group-level (bridge) evaluation
# --------------------------------------------------------------------------- #
def grouplevel_fixture():
    # item 100 is globally popular (rated by users 0..9) but NOT by the group
    # members (50, 51, who rated items 1, 2). The group's held-out choice is 100.
    rows = [(u, 100) for u in range(10)] + [(u, 200) for u in range(5)]
    rows += [(50, 1), (51, 2)]
    df = pd.DataFrame(rows, columns=["user", "item"])
    data = Dataset(df)
    split = Split(train=data, test=df.iloc[:0])
    groups = Groups([np.array([50, 51])])
    return data, groups, split


def test_grouplevel_eval_hits_group_choice():
    data, groups, split = grouplevel_fixture()
    rec = GroupRecommender(Popularity(measure="count"), AverageAggregator())
    rep = evaluate_grouplevel(rec, data, groups, split, group_truth={0: {100}},
                              k=2, metrics=["hr", "ndcg"])
    d = rep.to_dict()
    assert d[("coupled", "hr", 2, "group")] == pytest.approx(1.0)
    assert d[("coupled", "ndcg", 2, "group")] == pytest.approx(1.0)  # 100 ranked first


def test_grouplevel_via_benchmark_task():
    data, groups, split = grouplevel_fixture()
    task = BenchmarkTask(name="bridge", data=data, groups=groups, splits=split,
                         level="group", group_truth={0: {100}})
    res = benchmark({"AVG": GroupRecommender(Popularity(), AverageAggregator())},
                    [task], metrics=["hr", "ndcg"], k=2)
    df = res.to_frame()
    assert (df["protocol"] == "coupled").all()  # group-level is coupled-only
    assert df[(df.metric == "hr")]["value"].iloc[0] == pytest.approx(1.0)


def test_grouplevel_missing_truth_raises():
    data, groups, split = grouplevel_fixture()
    task = BenchmarkTask(name="bad", data=data, groups=groups, splits=split, level="group")
    res = benchmark({"AVG": GroupRecommender(Popularity(), AverageAggregator())},
                    [task], metrics=["hr"], on_error="skip")
    assert res.to_frame().empty  # skipped due to missing group_truth
