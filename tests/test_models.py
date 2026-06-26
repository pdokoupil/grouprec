"""Smoke + integration tests for the deep group models (requires torch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from grouprec import GroupRecommender, benchmark
from grouprec.bench import BenchmarkTask
from grouprec.aggregators import AverageAggregator
from grouprec.backends import Popularity
from grouprec.eval import evaluate_grouplevel
from grouprec.models import (
    AGREE, AlignGroup, ConsRec, GroupIM, HyperGroup, NCFGroup, make_synthetic_group_data,
)


def data_fixture():
    return make_synthetic_group_data(n_users=80, n_items=60, n_groups=30,
                                     group_size=4, items_per_group=8, seed=0)


@pytest.mark.parametrize("Model", [NCFGroup, AGREE])
def test_fit_and_recommend(Model):
    gd = data_fixture()
    model = Model(gd.groups, gd.group_interactions, factors=16, epochs=3, seed=0)
    model.fit(gd.dataset)
    assert model.paradigm == "profile"
    out = model.recommend(gd.groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    assert set(out.tolist()) <= set(gd.dataset.items.tolist())


@pytest.mark.parametrize("Model", [NCFGroup, AGREE])
def test_recommend_respects_exclude(Model):
    gd = data_fixture()
    model = Model(gd.groups, gd.group_interactions, factors=16, epochs=2, seed=0).fit(gd.dataset)
    exclude = set(int(x) for x in gd.dataset.items[:5])
    out = set(model.recommend(gd.groups[0], k=10, exclude=exclude).tolist())
    assert out.isdisjoint(exclude)


def test_cold_members_do_not_crash():
    gd = data_fixture()
    model = NCFGroup(gd.groups, gd.group_interactions, factors=8, epochs=1, seed=0).fit(gd.dataset)
    out = model.recommend(np.array([999999, 888888]), k=5)  # unknown users
    assert len(out) == 5


@pytest.mark.parametrize("Model", [NCFGroup, AGREE])
def test_models_learn_above_random(Model):
    gd = data_fixture()
    model = Model(gd.groups, gd.group_interactions, factors=32, epochs=40, seed=0).fit(gd.dataset)
    rep = evaluate_grouplevel(model, gd.dataset, gd.groups, gd.split, gd.group_truth,
                              k=10, metrics=["hr", "ndcg"])
    hr = rep.to_dict()[("coupled", "hr", 10, "group")]
    random_hr = 10 / gd.dataset.n_items
    assert hr > random_hr  # learned signal beats random


def test_groupim_fit_recommend_and_candidates():
    gd = data_fixture()
    m = GroupIM(gd.groups, gd.group_interactions, embedding_dim=16,
                epochs=3, pretrain_epochs=2, seed=0).fit(gd.dataset)
    assert m.paradigm == "profile"
    out = m.recommend(gd.groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    # candidate-restricted ranking (sampled protocol path)
    cands = [int(x) for x in gd.dataset.items[:20]]
    ranked = m.recommend(gd.groups[0], k=5, candidates=cands)
    assert len(ranked) == 5 and set(ranked.tolist()) <= set(cands)


def test_groupim_learns_above_random():
    gd = data_fixture()
    m = GroupIM(gd.groups, gd.group_interactions, embedding_dim=32,
                epochs=30, pretrain_epochs=10, seed=0).fit(gd.dataset)
    rep = evaluate_grouplevel(m, gd.dataset, gd.groups, gd.split, gd.group_truth,
                              k=10, metrics=["hr"])
    assert rep.to_dict()[("coupled", "hr", 10, "group")] > 10 / gd.dataset.n_items


def test_consrec_fit_recommend_and_learns():
    gd = data_fixture()
    m = ConsRec(gd.groups, gd.group_interactions, emb_dim=16, layers=2, epochs=3, seed=0)
    m.fit(gd.dataset)
    assert m.paradigm == "profile"
    # transductive: recommend maps the member set back to its group index
    out = m.recommend(gd.groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    cands = [int(x) for x in gd.dataset.items[:20]]
    ranked = m.recommend(gd.groups[0], k=5, candidates=cands)
    assert set(ranked.tolist()) <= set(cands)


def test_consrec_learns_above_random():
    gd = data_fixture()
    m = ConsRec(gd.groups, gd.group_interactions, emb_dim=32, layers=2, epochs=30, seed=0).fit(gd.dataset)
    rep = evaluate_grouplevel(m, gd.dataset, gd.groups, gd.split, gd.group_truth, k=10, metrics=["hr"])
    assert rep.to_dict()[("coupled", "hr", 10, "group")] > 10 / gd.dataset.n_items


def test_hypergroup_fit_recommend_and_learns():
    gd = data_fixture()
    m = HyperGroup(gd.groups, gd.group_interactions, emb_dim=16, layers=2, epochs=3, seed=0)
    m.fit(gd.dataset)
    assert m.paradigm == "profile"
    out = m.recommend(gd.groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    cands = [int(x) for x in gd.dataset.items[:20]]
    assert set(m.recommend(gd.groups[0], k=5, candidates=cands).tolist()) <= set(cands)


def test_hypergroup_learns_above_random():
    gd = data_fixture()
    m = HyperGroup(gd.groups, gd.group_interactions, emb_dim=32, layers=2, epochs=30, seed=0).fit(gd.dataset)
    rep = evaluate_grouplevel(m, gd.dataset, gd.groups, gd.split, gd.group_truth, k=10, metrics=["hr"])
    assert rep.to_dict()[("coupled", "hr", 10, "group")] > 10 / gd.dataset.n_items


def test_aligngroup_fit_recommend_and_learns():
    gd = data_fixture()
    m = AlignGroup(gd.groups, gd.group_interactions, emb_dim=16, layers=2, epochs=3,
                   batch_size=256, seed=0)
    m.fit(gd.dataset)
    assert m.paradigm == "profile"
    out = m.recommend(gd.groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    cands = [int(x) for x in gd.dataset.items[:20]]
    assert set(m.recommend(gd.groups[0], k=5, candidates=cands).tolist()) <= set(cands)


def test_aligngroup_learns_above_random():
    gd = data_fixture()
    m = AlignGroup(gd.groups, gd.group_interactions, emb_dim=32, layers=2, epochs=25,
                   batch_size=256, seed=0).fit(gd.dataset)
    rep = evaluate_grouplevel(m, gd.dataset, gd.groups, gd.split, gd.group_truth, k=10, metrics=["hr"])
    assert rep.to_dict()[("coupled", "hr", 10, "group")] > 10 / gd.dataset.n_items


def test_bridge_deep_and_aggregator_one_leaderboard():
    gd = data_fixture()
    task = BenchmarkTask(name="bridge", data=gd.dataset, groups=gd.groups,
                         splits=gd.split, level="group", group_truth=gd.group_truth)
    recs = {
        "AVG": GroupRecommender(Popularity(measure="mean"), AverageAggregator()),
        "AGREE": AGREE(gd.groups, gd.group_interactions, factors=16, epochs=5, seed=0),
    }
    res = benchmark(recs, [task], metrics=["hr", "ndcg"], k=10, silent=True)
    df = res.to_frame()
    # both paradigms share one coupled leaderboard; decoupled never appears
    assert set(df["recommender"]) == {"AVG", "AGREE"}
    assert set(df["protocol"]) == {"coupled"}
    assert set(df["paradigm"]) == {"results", "profile"}
