"""Tests for the ConsRec-format parser and the sampled (1+N) ranking evaluation,
on a tiny synthetic fixture in the same file format."""

from __future__ import annotations

import numpy as np
import pytest

import grouprec as gr
from grouprec import GroupRecommender, evaluate_sampled
from grouprec.aggregators import AverageAggregator
from grouprec.backends import Popularity
from grouprec.datasets import load_consrec


def write_consrec(root):
    # 2 groups, items 0..9; group 0 = users {1,2}, group 1 = users {3,4}
    (root / "groupMember.txt").write_text("0 1,2\n1 3,4\n")
    # group-item train positives
    (root / "groupRatingTrain.txt").write_text("0 0 1\n0 1 1\n1 5 1\n1 6 1\n")
    # user-item train (gives the base RS popularity signal: item 0 and 5 popular)
    (root / "userRatingTrain.txt").write_text(
        "1 0 1\n2 0 1\n3 5 1\n4 5 1\n1 1 1\n3 6 1\n")
    # held-out group positives (LOO)
    (root / "groupRatingTest.txt").write_text("0 0 1\n1 5 1\n")
    # negatives keyed by (group,item): pos vs a few negatives
    (root / "groupRatingNegative.txt").write_text(
        "(0,0) 7 8 9\n(1,5) 2 3 4\n")


def test_load_consrec_parses_structure(tmp_path):
    write_consrec(tmp_path)
    gd = load_consrec(tmp_path, "toy")
    assert len(gd.groups) == 2
    assert list(gd.groups[0]) == [1, 2]
    assert gd.group_interactions[0] == [0, 1]
    assert len(gd.test_instances) == 2
    gi, pos, negs = gd.test_instances[0]
    assert gi == 0 and pos == 0 and negs == [7, 8, 9]
    # item/user vocab spans all files
    assert {0, 1, 5, 6, 7, 8, 9} <= set(gd.dataset.items.tolist())


def test_sampled_eval_ranks_positive(tmp_path):
    write_consrec(tmp_path)
    gd = load_consrec(tmp_path, "toy")
    # popularity makes the held-out positives (0 and 5) the top of their candidate sets
    rec = GroupRecommender(Popularity(measure="count"), AverageAggregator())
    rep = evaluate_sampled(rec, gd.dataset, gd.groups, gd.test_instances, ks=(1, 2))
    d = rep.to_dict()
    assert d[("coupled", "hr", 2, "sampled")] == pytest.approx(1.0)
    assert d[("coupled", "ndcg", 2, "sampled")] > 0.0


def test_sampled_via_benchmark_bridges_paradigms(tmp_path):
    write_consrec(tmp_path)
    gd = load_consrec(tmp_path, "toy")
    from grouprec import benchmark
    from grouprec.bench import BenchmarkTask

    task = BenchmarkTask("toy", gd.dataset, gd.groups, gd.split,
                         level="sampled", test_instances=gd.test_instances)
    res = benchmark({"pop": GroupRecommender(Popularity(), AverageAggregator())},
                    [task], sampled_ks=(1, 5), silent=True)
    df = res.to_frame()
    assert set(df["protocol"]) == {"coupled"}
    assert set(df["aggregation"]) == {"sampled"}
    assert {"hr", "ndcg"} <= set(df["metric"])


def test_sampled_requires_test_instances_in_benchmark(tmp_path):
    write_consrec(tmp_path)
    gd = load_consrec(tmp_path, "toy")
    from grouprec import benchmark
    from grouprec.bench import BenchmarkTask

    task = BenchmarkTask("toy", gd.dataset, gd.groups, gd.split, level="sampled")
    res = benchmark({"pop": GroupRecommender(Popularity(), AverageAggregator())},
                    [task], silent=True)
    assert res.to_frame().empty  # skipped: no test_instances


def test_load_groupim_format(tmp_path):
    from grouprec.datasets import load_groupim
    (tmp_path / "group_users.csv").write_text("group,user\n0,1.0\n0,2.0\n1,3.0\n1,4.0\n")
    (tmp_path / "train_gi.csv").write_text("group,item\n0,10\n0,11\n1,20\n")
    (tmp_path / "train_ui.csv").write_text(
        "user,item\n1,10\n2,10\n3,20\n4,20\n1,11\n1,30\n2,31\n3,32\n4,33\n")
    (tmp_path / "test_gi.csv").write_text("group,item\n0,11\n1,21\n")
    gd = load_groupim(tmp_path, "wp", n_negatives=3, seed=0)
    assert len(gd.groups) == 2 and list(gd.groups[0]) == [1, 2]
    assert gd.group_interactions[0] == [10, 11]
    gi, pos, negs = gd.test_instances[0]
    assert gi == 0 and pos == 11 and len(negs) == 3
    assert pos not in negs  # negatives exclude the positive


def test_load_yin_format(tmp_path):
    from grouprec.datasets import load_yin
    sub = tmp_path / "yelp_la"
    sub.mkdir()
    (sub / "groupid_users.dat").write_text("0\tA,B\n1\tC,D\n")
    (sub / "groupid_events.dat").write_text("0\tx,y,z\n1\tp,q\n")
    (sub / "user_events.dat").write_text("A\tx,y\nB\tx\nC\tp\nD\tq,p\n")
    gd = load_yin(tmp_path, "yelp", n_negatives=2, seed=0)
    assert len(gd.groups) == 2 and list(gd.groups[0]) == ["A", "B"]
    # one event held out per group; rest are training interactions
    assert len(gd.test_instances) == 2
    gi, pos, negs = gd.test_instances[0]
    assert pos in {"x", "y", "z"} and pos not in gd.group_interactions[gi]
    assert len(negs) <= 2 and pos not in negs


def test_dataset_explicit_vocab():
    import pandas as pd
    from grouprec import Dataset
    df = pd.DataFrame([(1, 10), (2, 20)], columns=["user", "item"])
    d = Dataset(df, items=[10, 20, 30, 40], users=[1, 2, 3])
    assert d.n_items == 4 and d.n_users == 3
    assert d.item_index[40] == 3
    # absent items present in vocab -> zero columns in the matrix
    assert d.user_item_matrix(value="binary").shape == (3, 4)
