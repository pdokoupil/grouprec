"""Tests for the fairness / greedy aggregators: GFAR, GreedyLM (=GLM), PAR,
SPGreedy. GFAR is checked against the worked example in the RecSys'20 paper;
GreedyLM against a port of the original ``aggregators_new.py`` greedy.
"""

from __future__ import annotations

import numpy as np
import pytest

from grouprec.aggregators import (
    GFARAggregator,
    GreedyLMAggregator,
    PARAggregator,
    SPGreedyAggregator,
    GreedyScalarizationAggregator,
)


# --------------------------------------------------------------------------- #
# GFAR
# --------------------------------------------------------------------------- #
def test_gfar_paper_worked_example():
    # Reconstructs Table 1/2 of Kaya et al. (RecSys'20): items i1..i4 (cols 0..3),
    # N=3. Expected output top-3 = {i1, i4, i2} = [0, 3, 1].
    rm = np.array([
        [5.0, 4.0, 3.0, 0.0],   # u1: top3 i1>i2>i3
        [5.0, 4.0, 3.0, 0.0],   # u2: same
        [3.0, 0.0, 4.0, 5.0],   # u3: top3 i4>i3>i1
    ])
    out = GFARAggregator(relevant_max_items=3).aggregate(rm, 3)
    assert list(out) == [0, 3, 1]


def oracle_gfar(rm, k, N):
    from scipy.stats import rankdata
    m, n = rm.shape
    N = min(N, n)
    p_rel = np.zeros((m, n))
    for u in range(m):
        top = np.argsort(-rm[u], kind="stable")[:N]
        borda = rankdata(rm[u, top], method="max") - 1.0
        s = borda.sum()
        if s > 0:
            p_rel[u, top] = borda / s
    avail = np.ones(n, bool)
    prob_not = np.ones(m)
    chosen = []
    for _ in range(min(k, n)):
        mg = (p_rel * prob_not[:, None]).sum(axis=0)
        mg = np.where(avail, mg, -np.inf)
        b = int(np.argmax(mg))
        chosen.append(b)
        avail[b] = False
        prob_not = prob_not * (1 - p_rel[:, b])
    return chosen


@pytest.mark.parametrize("shape", [(2, 15), (3, 30), (5, 50)])
@pytest.mark.parametrize("k", [5, 10])
def test_gfar_matches_oracle(shape, k):
    rng = np.random.default_rng(hash((shape, k, "gfar")) % 2**32)
    rm = rng.uniform(0.5, 5.0, size=shape)
    out = list(GFARAggregator(relevant_max_items=20).aggregate(rm, k))
    assert out == oracle_gfar(rm, k, 20)


# --------------------------------------------------------------------------- #
# GreedyLM (== Xiao greedy, least-misery, proportional utility, lam=0.5)
# --------------------------------------------------------------------------- #
def oracle_greedylm_original(rm, k, lam=0.5):
    """Port of GreedyLMAggregator.generate_group_recommendations_for_group_fastest."""
    m, n = rm.shape
    items = np.arange(n)
    rec = []
    k_max = min(n, k)
    top_k_sums = np.sort(rm, axis=1)[:, -k_max:].sum(axis=1)
    while len(rec) < min(k, n):
        rec_idx = np.array(rec, dtype=int)
        pre = rm[:, rec_idx].sum(axis=1) if rec_idx.size else np.zeros(m)
        cand_sum = pre[:, None] + rm[:, items]
        indiv = cand_sum / top_k_sums[:, None]
        sw = indiv.mean(axis=0)
        f = indiv.min(axis=0)
        scores = lam * sw + (1 - lam) * f
        best = int(np.argmax(scores))
        rec.append(int(items[best]))
        items = np.delete(items, best)
    return rec


@pytest.mark.parametrize("shape", [(2, 12), (3, 25), (4, 40)])
@pytest.mark.parametrize("k", [5, 10])
def test_greedylm_matches_original(shape, k):
    rng = np.random.default_rng(hash((shape, k, "glm")) % 2**32)
    rm = rng.uniform(0.5, 5.0, size=shape)
    out = list(GreedyLMAggregator().aggregate(rm, k))
    assert out == oracle_greedylm_original(rm, k)


# --------------------------------------------------------------------------- #
# Generic greedy oracle for PAR (variance fairness)
# --------------------------------------------------------------------------- #
def oracle_greedy(rm, k, fairness, lam, utility="proportional"):
    m, n = rm.shape
    if utility == "proportional":
        kk = min(n, k)
        denom = np.sort(rm, axis=1)[:, -kk:].sum(axis=1)
        denom[denom == 0] = 1.0
    else:
        denom = np.ones(m)
    avail = np.ones(n, bool)
    cum = np.zeros(m)
    chosen = []
    for _ in range(min(k, n)):
        cand = (cum[:, None] + rm) / denom[:, None]
        sw = cand.mean(axis=0)
        if fairness == "least_misery":
            f = cand.min(axis=0)
        elif fairness == "variance":
            f = 1 - cand.var(axis=0)
        else:
            raise ValueError(fairness)
        score = np.where(avail, lam * sw + (1 - lam) * f, -np.inf)
        b = int(np.argmax(score))
        chosen.append(b)
        avail[b] = False
        cum = cum + rm[:, b]
    return chosen


@pytest.mark.parametrize("shape", [(2, 12), (3, 25), (4, 40)])
@pytest.mark.parametrize("k", [5, 10])
def test_par_matches_greedy_oracle(shape, k):
    rng = np.random.default_rng(hash((shape, k, "par")) % 2**32)
    rm = rng.uniform(0.5, 5.0, size=shape)
    out = list(PARAggregator().aggregate(rm, k))
    assert out == oracle_greedy(rm, k, "variance", lam=0.8)


def test_greedylm_consistent_with_generic_greedy():
    rng = np.random.default_rng(99)
    rm = rng.uniform(0.5, 5.0, size=(4, 30))
    assert list(GreedyLMAggregator().aggregate(rm, 8)) == oracle_greedy(rm, 8, "least_misery", 0.5)


def test_greedy_lam_one_is_average_ranking():
    # lam=1 -> pure social welfare; with proportional utility the first pick is the
    # item with highest mean proportional contribution.
    rng = np.random.default_rng(11)
    rm = rng.uniform(0.5, 5.0, size=(3, 20))
    agg = GreedyScalarizationAggregator(fairness="least_misery", lam=1.0, utility="average")
    # average utility + lam=1: marginal SW gain of item i is its mean -> ranking by mean
    out = list(agg.aggregate(rm, 5))
    expected = list(np.argsort(-rm.mean(axis=0), kind="stable")[:5])
    assert out == expected


# --------------------------------------------------------------------------- #
# SPGreedy
# --------------------------------------------------------------------------- #
def test_spgreedy_set_cover_example():
    # n_like=1 (delta=0.25, n=4). u0,u1 like item0; u2 likes item1.
    rm = np.array([
        [9.0, 1.0, 1.0, 1.0],
        [9.0, 1.0, 1.0, 1.0],
        [1.0, 9.0, 1.0, 1.0],
    ])
    out = SPGreedyAggregator(delta=0.25).aggregate(rm, 2)
    assert list(out) == [0, 1]  # cover {u0,u1} then {u2}


def oracle_spgreedy(rm, k, delta):
    m, n = rm.shape
    n_like = max(1, int(round(delta * n)))
    likes = np.zeros((m, n), bool)
    for u in range(m):
        likes[u, np.argsort(-rm[u], kind="stable")[:n_like]] = True
    mean_rel = rm.mean(axis=0)
    avail = np.ones(n, bool)
    covered = np.zeros(m, bool)
    chosen = []
    for _ in range(min(k, n)):
        gain = np.where(avail, (likes & (~covered)[:, None]).sum(axis=0).astype(float), -1.0)
        cand = np.flatnonzero(gain == gain.max())
        b = int(cand[np.argmax(mean_rel[cand])])
        chosen.append(b)
        avail[b] = False
        covered |= likes[:, b]
    return chosen


@pytest.mark.parametrize("shape", [(3, 20), (4, 30), (5, 40)])
@pytest.mark.parametrize("delta", [0.1, 0.25])
def test_spgreedy_matches_oracle(shape, delta):
    rng = np.random.default_rng(hash((shape, delta)) % 2**32)
    rm = rng.uniform(0.5, 5.0, size=shape)
    out = list(SPGreedyAggregator(delta=delta).aggregate(rm, 8))
    assert out == oracle_spgreedy(rm, 8, delta)


# --------------------------------------------------------------------------- #
# shared invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agg", [
    GFARAggregator(), GreedyLMAggregator(), PARAggregator(), SPGreedyAggregator(),
])
def test_no_duplicates_length_and_exclude(agg):
    rng = np.random.default_rng(7)
    rm = rng.uniform(0.5, 5.0, size=(4, 30))
    out = agg.aggregate(rm, 10)
    assert len(out) == 10 and len(set(out.tolist())) == 10
    exclude = {0, 1, 2, 3}
    out2 = set(agg.aggregate(rm, 10, exclude=exclude).tolist())
    assert out2.isdisjoint(exclude)
