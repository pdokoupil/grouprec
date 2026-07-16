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
    AGREE, AlignGroup, ConsRec, GroupIM, HHGR, HyperGroup, NCFGroup, make_synthetic_group_data,
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


def test_groupim_group_scores_and_member_weights():
    gd = data_fixture()
    m = GroupIM(gd.groups, gd.group_interactions, embedding_dim=16,
                epochs=3, pretrain_epochs=2, seed=0).fit(gd.dataset)
    members = gd.groups[0]
    cands = [int(x) for x in gd.dataset.items[:20]]
    s = m.group_scores(members, cands)
    assert s.shape == (len(cands),)
    # uniform member_weights reproduce the native (None) scoring exactly
    s_uniform = m.group_scores(members, cands, member_weights=[1.0] * len(members))
    assert np.allclose(s, s_uniform, atol=1e-5)
    # return_attention gives one pooling weight per member, summing to ~1
    _, att_uniform = m.group_scores(members, cands, member_weights=[1.0] * len(members),
                                    return_attention=True)
    heavy = [5.0] + [1.0] * (len(members) - 1)
    _, att_heavy = m.group_scores(members, cands, member_weights=heavy, return_attention=True)
    assert att_heavy.shape == (len(members),) and abs(float(att_heavy.sum()) - 1.0) < 1e-4
    # boosting member 0 raises its pooling weight (the steering actually takes effect)
    assert att_heavy[0] > att_uniform[0]
    # recommend(member_weights=...) routes through group_scores (steerable)
    r = m.recommend(members, k=5, candidates=cands, member_weights=[5.0] + [1.0] * (len(members) - 1))
    assert len(r) == 5 and set(r.tolist()) <= set(cands)


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


@pytest.mark.parametrize("Model", [NCFGroup, AGREE, ConsRec, HyperGroup, AlignGroup])
def test_group_scores_uniform_interface(Model):
    """Every deep model exposes group_scores(members, items=None) with consistent shape,
    and ranking via group_scores matches recommend()."""
    gd = data_fixture()
    m = Model(gd.groups, gd.group_interactions, epochs=2, seed=0).fit(gd.dataset)
    members = gd.groups[0]
    full = m.group_scores(members)
    assert full.shape == (gd.dataset.n_items,)
    cands = [int(x) for x in gd.dataset.items[:15]]
    sub = m.group_scores(members, cands)
    assert sub.shape == (len(cands),)
    # candidate ranking by group_scores == recommend(candidates=...)
    by_scores = np.asarray(cands)[np.argsort(-sub, kind="stable")[:5]]
    np.testing.assert_array_equal(by_scores, m.recommend(members, k=5, candidates=cands))


@pytest.mark.parametrize("Model,supported", [
    (NCFGroup, True), (AGREE, True), (GroupIM, True),
    (ConsRec, False), (HyperGroup, False), (AlignGroup, False), (HHGR, False),
])
def test_supports_member_weights_flag(Model, supported):
    assert Model.supports_member_weights is supported


@pytest.mark.parametrize("Model", [NCFGroup, AGREE])
def test_member_weights_steer_pooling(Model):
    """For member-pooling models, boosting one member raises its pooling weight and
    moves the scores; uniform weights reproduce the native model."""
    gd = data_fixture()
    m = Model(gd.groups, gd.group_interactions, factors=16, epochs=3, seed=0).fit(gd.dataset)
    members = gd.groups[0]
    cands = [int(x) for x in gd.dataset.items[:20]]
    base = m.group_scores(members, cands)
    uniform = m.group_scores(members, cands, member_weights=[1.0] * len(members))
    np.testing.assert_allclose(base, uniform, atol=1e-5)   # uniform == native
    _, att_u = m.group_scores(members, cands, member_weights=[1.0] * len(members),
                              return_attention=True)
    heavy = [5.0] + [1.0] * (len(members) - 1)
    steered, att_h = m.group_scores(members, cands, member_weights=heavy, return_attention=True)
    assert att_h.shape == (len(members),) and abs(float(att_h.sum()) - 1.0) < 1e-4
    assert att_h[0] > att_u[0]                              # member 0 pulled up
    assert not np.allclose(steered, base)                  # steering changes the scores


@pytest.mark.parametrize("Model", [ConsRec, HyperGroup, AlignGroup])
def test_transductive_models_reject_member_weights(Model):
    gd = data_fixture()
    m = Model(gd.groups, gd.group_interactions, epochs=2, seed=0).fit(gd.dataset)
    members = gd.groups[0]
    with pytest.raises(NotImplementedError):
        m.group_scores(members, member_weights=[1.0, 1.0, 1.0, 1.0])
    with pytest.raises(NotImplementedError):
        m.group_scores(members, return_attention=True)
    with pytest.raises(NotImplementedError):
        m.recommend(members, k=5, member_weights=[1.0, 1.0, 1.0, 1.0])


def test_group_recommender_member_and_group_scores():
    """GroupRecommender.member_scores == base RS output; group_scores == aggregated
    per-item utility for score-based aggregators; selection-based ones raise."""
    from grouprec.aggregators import WeightedAverageAggregator, EPFuzzDAAggregator
    from grouprec.backends import EASE
    gd = data_fixture()
    cands = [int(x) for x in gd.dataset.items[:20]]
    members = gd.groups[0]

    rec = GroupRecommender(EASE(reg=50.0),
                           WeightedAverageAggregator(member_weights=[0.6, 0.3, 0.1, 0.0]),
                           normalize="minmax").fit(gd.dataset)
    ms = rec.member_scores(members, items=cands)
    assert ms.shape == (len(members), len(cands))
    gs = rec.group_scores(members, items=cands)
    assert gs.shape == (len(cands),)
    # group_scores is the weighted mean of member_scores (the ranking criterion)
    w = np.array([0.6, 0.3, 0.1, 0.0]); w = w / w.sum()
    np.testing.assert_allclose(gs, (w[:, None] * ms).sum(0), atol=1e-6)
    # ranking by group_scores == recommend
    by_scores = np.asarray(cands)[np.argsort(-gs, kind="stable")[:5]]
    np.testing.assert_array_equal(by_scores, rec.recommend(members, k=5, candidates=cands))

    # selection-based aggregator: no per-item group score
    rec2 = GroupRecommender(EASE(reg=50.0), EPFuzzDAAggregator(), normalize="minmax").fit(gd.dataset)
    with pytest.raises(NotImplementedError):
        rec2.group_scores(members, items=cands)


def test_hhgr_fit_recommend_and_candidates():
    gd = data_fixture()
    m = HHGR(gd.groups, gd.group_interactions, emb_dim=16, epochs=2, group_epochs=3, seed=0)
    m.fit(gd.dataset)
    assert m.paradigm == "profile"
    out = m.recommend(gd.groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    cands = [int(x) for x in gd.dataset.items[:20]]
    ranked = m.recommend(gd.groups[0], k=5, candidates=cands)
    assert len(ranked) == 5 and set(ranked.tolist()) <= set(cands)
    # ranking by group_scores agrees with recommend(candidates=...)
    sub = m.group_scores(gd.groups[0], cands)
    np.testing.assert_array_equal(np.asarray(cands)[np.argsort(-sub, kind="stable")[:5]], ranked)
    # transductive: per-member weighting is not defined
    with pytest.raises(NotImplementedError):
        m.group_scores(gd.groups[0], member_weights=[1.0] * len(gd.groups[0]))


def test_hhgr_double_scale_views_and_group_graph():
    """The two scales must actually differ (coarse drops users, fine drops hyperedges),
    otherwise the self-supervised signal is vacuous."""
    from grouprec.models.hhgr import _corrupt_views, _user_group_incidence, _group_graph
    members = [np.array([0, 1, 2]), np.array([2, 3, 4]), np.array([4, 5, 0]), np.array([9])]
    H = _user_group_incidence(members, n_users=10, n_groups=4)
    assert H.shape == (10, 4) and H.sum() == 10        # one entry per membership
    fine, coarse = _corrupt_views(H, np.random.default_rng(0), coarse_frac=0.5, fine_frac=0.5)
    # both views are subsets of the original incidence, and are not identical to each other
    assert (fine.toarray() <= H.toarray()).all()
    assert (coarse.toarray() <= H.toarray()).all()
    assert not np.array_equal(fine.toarray(), coarse.toarray())
    # Groups 0,1,2 pairwise overlap (a triangle); group 3 is disjoint. The reference builds
    # H_gg = (d @ d.T) * d, i.e. the weight is the number of *common neighbour groups*, masked
    # to directly-overlapping pairs -- so an overlapping pair only scores if it also shares a
    # third group, and identical/disjoint groups score zero.
    gg = _group_graph(members, 4).toarray()
    assert gg[0, 1] > 0 and gg[1, 0] > 0
    assert gg[0, 3] == 0 and gg[3, 0] == 0
    assert (np.diag(gg) == 0).all()


def test_hhgr_learns_above_random():
    gd = data_fixture()
    m = HHGR(gd.groups, gd.group_interactions, emb_dim=32, epochs=5, group_epochs=30,
             seed=0).fit(gd.dataset)
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
