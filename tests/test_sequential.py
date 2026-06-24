"""Validate RLProp / LTP / EP-FuzzDA(+Weighted) / PeriodicFAI against oracles
ported from ``aggregators_new.py`` (and the ltp_prospective semantics for the
session-averaged LTP state).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grouprec.aggregators import (
    RLPropAggregator,
    LTPAggregator,
    PeriodicFAIAggregator,
    EPFuzzDAAggregator,
    EPFuzzDAWeightedAggregator,
)
from grouprec.aggregators import SDAAAggregator, SIAAAggregator, AverageAggregator
from grouprec.aggregators.sequential import _satisfaction

NEG_INF = int(-10e6)


def matrix_to_df(rm):
    rows = [(u, i, rm[u, i]) for u in range(rm.shape[0]) for i in range(rm.shape[1])]
    return pd.DataFrame(rows, columns=["user", "item", "predicted_rating"])


# --------------------------------------------------------------------------- #
# oracles
# --------------------------------------------------------------------------- #
def oracle_rlprop(rm, k):
    """Port of RLPropAggregator.fast_impl_v3_without_history (with_norm=False)."""
    rm = rm.copy().astype(float)
    m, n = rm.shape
    weights = np.ones(m) / m
    TOT = 0.0
    gm = np.zeros(m)
    seen = np.ones(n)
    pos = rm >= 0.0
    neg = rm < 0.0
    top = []
    for _ in range(k):
        gain = np.zeros_like(rm)
        tots = np.maximum(TOT, TOT + rm.sum(axis=0))
        remainder = tots * weights[:, None] - gm[:, None]
        gain[pos] = np.maximum(0, np.minimum(rm, remainder)[pos])
        gain[neg] = np.minimum(0, (rm - remainder)[neg])
        raw = gain.sum(axis=0)
        scores = raw * seen + NEG_INF * (1 - seen)
        i_best = int(scores.argmax())
        seen[i_best] = 0
        gm = gm + rm[:, i_best]
        TOT = float(np.clip(gm, 0.0, None).sum())
        top.append(i_best)
    return top


def oracle_ltp(sessions, k, gamma, beta, tie_breaking):
    """Session-averaged stateful RLProp == ltp_prospective semantics."""
    results = []
    gm = None
    n_sessions = 0
    for rm in sessions:
        rm = rm.copy().astype(float)
        m, n = rm.shape
        weights = np.ones(m) / m
        if gm is None:
            gm = np.zeros(m)
        sess = max(n_sessions, 1)
        TOT = float((gm / sess).sum())
        seen = np.ones(n)
        pos = rm >= 0.0
        neg = rm < 0.0
        top = []
        for position in range(k):
            gain = np.zeros_like(rm)
            tots = np.maximum(TOT, TOT + rm.sum(axis=0))
            remainder = tots * weights[:, None] - (gm / sess)[:, None]
            gain[pos] = np.maximum(0, np.minimum(rm, remainder)[pos])
            gain[neg] = np.minimum(0, (rm - remainder)[neg])
            if tie_breaking != 0.0:
                tie = remainder <= rm
                gain[tie] = gain[tie] + tie_breaking * (rm - remainder)[tie]
            scores = np.where(seen > 0, gain.sum(axis=0), -np.inf)
            i_best = int(scores.argmax())
            seen[i_best] = 0
            position_factor = np.exp(-beta * position)
            gm = gamma * gm + position_factor * rm[:, i_best]
            TOT = float(np.clip(gm / sess, 0.0, None).sum())
            top.append(i_best)
        n_sessions += 1
        results.append(top)
    return results


def oracle_epfuzzda(rm, k, member_weights=None):
    """Port of EPFuzzDAAggregator.ep_fuzzdhondt_algorithm (returns selection, awarded)."""
    df = matrix_to_df(rm)
    members = df.user.unique()
    m = len(members)
    if member_weights is None:
        member_weights = [1.0 / m] * m
    mw = pd.DataFrame(pd.Series(member_weights, index=members))
    cu = pd.pivot_table(df, values="predicted_rating", index="item", columns="user", fill_value=0.0)
    csu = pd.DataFrame(cu.sum(axis="columns"))
    tua = pd.Series(np.zeros(m), index=members)
    total = 0.0
    selected = []
    for _ in range(k):
        ptu = csu + total
        allowed = pd.DataFrame(np.dot(ptu.values, mw.T.values), columns=mw.T.columns, index=ptu.index)
        unf = allowed.subtract(tua, axis="columns")
        unf[unf < 0] = 0
        cur = pd.concat([unf, cu]).groupby(level=0).min()
        crel = cur.sum(axis="columns")
        crel = crel.loc[~crel.index.isin(selected)]
        iid = crel.index[crel.argmax()]
        selected.append(iid)
        winner = cu.loc[iid, :]
        tua = tua + winner
        total += winner.values.sum()
    return selected, tua.values


# --------------------------------------------------------------------------- #
# RLProp single-session
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 10), (3, 25), (5, 40), (4, 8)])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_rlprop_matches_oracle_positive(shape, k):
    rng = np.random.default_rng(hash((shape, k, "rl")) % 2**32)
    rm = rng.uniform(0.5, 5.0, size=shape)
    k = min(k, shape[1])
    got = list(RLPropAggregator().aggregate(rm, k))
    assert got == oracle_rlprop(rm, k)


@pytest.mark.parametrize("shape", [(3, 20), (4, 30)])
def test_rlprop_matches_oracle_with_negatives(shape):
    rng = np.random.default_rng(hash((shape, "rlneg")) % 2**32)
    rm = rng.uniform(-2.0, 5.0, size=shape)
    got = list(RLPropAggregator().aggregate(rm, 10))
    assert got == oracle_rlprop(rm, 10)


# --------------------------------------------------------------------------- #
# LTP cross-session
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gamma,beta,tb", [(1.0, 0.0, 0.0), (0.9, 0.1, 0.0),
                                           (1.0, 0.0, 0.5), (0.8, 0.2, -0.3)])
def test_ltp_matches_oracle_over_sessions(gamma, beta, tb):
    rng = np.random.default_rng(hash((gamma, beta, tb)) % 2**32)
    sessions = [rng.uniform(0.5, 5.0, size=(4, 30)) for _ in range(5)]
    k = 6
    expected = oracle_ltp(sessions, k, gamma, beta, tb)
    agg = LTPAggregator(gamma=gamma, beta=beta, tie_breaking=tb, avoid_repeats=False)
    got = [list(agg.aggregate(rm, k)) for rm in sessions]
    assert got == expected


def test_ltp_first_session_equals_rlprop():
    rng = np.random.default_rng(7)
    rm = rng.uniform(0.5, 5.0, size=(5, 40))
    ltp = LTPAggregator(gamma=1.0, beta=0.0, tie_breaking=0.0, avoid_repeats=False)
    assert list(ltp.aggregate(rm, 15)) == list(RLPropAggregator().aggregate(rm, 15))


def test_ltp_avoid_repeats_across_sessions():
    rng = np.random.default_rng(8)
    sessions = [rng.uniform(0.5, 5.0, size=(3, 20)) for _ in range(3)]
    agg = LTPAggregator(avoid_repeats=True)
    seen = set()
    for rm in sessions:
        out = set(agg.aggregate(rm, 5).tolist())
        assert out.isdisjoint(seen)
        seen |= out


def test_ltp_reset_restores_first_session():
    rng = np.random.default_rng(9)
    rm = rng.uniform(0.5, 5.0, size=(4, 25))
    agg = LTPAggregator(avoid_repeats=False)
    first = list(agg.aggregate(rm, 10))
    agg.aggregate(rm, 10)  # advance state
    agg.reset()
    assert list(agg.aggregate(rm, 10)) == first


def test_ltp_improves_long_run_fairness_vs_avg():
    # Two members with conflicting tastes; over many sessions LTP should equalize
    # cumulative per-member utility better than plain AVG.
    rng = np.random.default_rng(123)
    n_items = 50
    sessions = []
    for _ in range(20):
        a = rng.uniform(0, 1, n_items)
        b = rng.uniform(0, 1, n_items)
        sessions.append(np.vstack([a, 5 * b]))  # member 1 has larger scale
    ltp = LTPAggregator(avoid_repeats=False, normalize="minmax")
    from grouprec.aggregators import AverageAggregator
    avg = AverageAggregator()

    def imbalance(agg, stateful):
        gains = np.zeros(2)
        for rm in sessions:
            sel = agg.aggregate(rm, 5)
            gains += rm[:, sel].sum(axis=1)
        # normalize to shares
        shares = gains / gains.sum()
        return abs(shares[0] - shares[1])

    ltp_imb = imbalance(ltp, True)
    avg_imb = imbalance(avg, False)
    assert ltp_imb <= avg_imb


# --------------------------------------------------------------------------- #
# EP-FuzzDA
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 10), (3, 25), (5, 40), (4, 8)])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_epfuzzda_matches_oracle(shape, k):
    rng = np.random.default_rng(hash((shape, k, "ep")) % 2**32)
    rm = rng.uniform(0.5, 5.0, size=shape)
    k = min(k, shape[1])
    got = list(EPFuzzDAAggregator().aggregate(rm, k))
    expected, _ = oracle_epfuzzda(rm, k)
    assert got == expected


def test_epfuzzda_custom_weights_match_oracle():
    rng = np.random.default_rng(55)
    rm = rng.uniform(0.5, 5.0, size=(3, 30))
    w = [0.6, 0.3, 0.1]
    got = list(EPFuzzDAAggregator(member_weights=w).aggregate(rm, 10))
    expected, _ = oracle_epfuzzda(rm, 10, w)
    assert got == expected


# --------------------------------------------------------------------------- #
# EPFuzzDAWeighted (sequential)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_all", [False, True])
def test_epfuzzda_weighted_matches_oracle(use_all):
    rng = np.random.default_rng(hash(("epw", use_all)) % 2**32)
    sessions = [rng.uniform(0.5, 5.0, size=(4, 30)) for _ in range(4)]
    k = 6

    # oracle
    gains_so_far = np.zeros(4)
    iter_ = 0
    expected = []
    for rm in sessions:
        m = 4
        if iter_ == 0:
            w = None
        else:
            if use_all:
                should = np.full(m, iter_ + 1.0) / m
            else:
                should = np.ones(m) / m
            w = list(np.maximum(0.0, should - gains_so_far))
        sel, awarded = oracle_epfuzzda(rm, k, w)
        total = awarded.sum()
        share = awarded / total if total > 0 else np.zeros(m)
        gains_so_far = gains_so_far + share
        iter_ += 1
        expected.append(list(sel))

    agg = EPFuzzDAWeightedAggregator(use_all_iter_weights=use_all)
    got = [list(agg.aggregate(rm, k)) for rm in sessions]
    assert got == expected


# --------------------------------------------------------------------------- #
# PeriodicFAI
# --------------------------------------------------------------------------- #
def test_periodicfai_pointer_persists_across_sessions():
    rng = np.random.default_rng(33)
    sessions = [rng.uniform(0.5, 5.0, size=(3, 20)) for _ in range(4)]
    k = 5
    agg = PeriodicFAIAggregator()

    total_items = 0
    for rm in sessions:
        m, n = rm.shape
        df = matrix_to_df(rm)
        users = df.user.unique()
        ui = total_items % len(users)
        selected = []
        for i in range(k):
            cur = df.loc[df.user == users[ui]].sort_values(
                "predicted_rating", ascending=False, kind="stable")
            cur = cur.loc[~cur.item.isin(selected)]
            selected.append(int(cur.item.iloc[0]))
            ui = (ui + 1) % len(users)
        total_items += k
        assert list(agg.aggregate(rm, k)) == selected


# --------------------------------------------------------------------------- #
# SDAA / SIAA (Stratigi et al.) -- verified against direct formula oracles
# --------------------------------------------------------------------------- #
def oracle_satisfaction(rm, selected, k):
    """Eq. 1 satisfaction, written out plainly."""
    m, n = rm.shape
    kk = min(n, k)
    sat = np.zeros(m)
    for u in range(m):
        ideal = np.sort(rm[u])[-kk:].sum()
        got = sum(rm[u, i] for i in selected)
        sat[u] = got / ideal if ideal > 0 else 0.0
    return sat


def test_satisfaction_helper_matches_oracle():
    rng = np.random.default_rng(1)
    rm = rng.uniform(0.5, 5.0, size=(4, 20))
    sel = np.array([3, 7, 1, 15, 9])
    np.testing.assert_allclose(_satisfaction(rm, sel, 5), oracle_satisfaction(rm, sel, 5))


def test_sdaa_first_round_is_average():
    rng = np.random.default_rng(2)
    rm = rng.uniform(0.5, 5.0, size=(4, 30))
    assert list(SDAAAggregator().aggregate(rm, 10)) == list(AverageAggregator().aggregate(rm, 10))


def test_siaa_first_round_is_average():
    rng = np.random.default_rng(3)
    rm = rng.uniform(0.5, 5.0, size=(4, 30))
    assert list(SIAAAggregator(b=0.5).aggregate(rm, 10)) == list(AverageAggregator().aggregate(rm, 10))


def test_sdaa_matches_formula_oracle_over_sessions():
    rng = np.random.default_rng(4)
    sessions = [rng.uniform(0.5, 5.0, size=(4, 25)) for _ in range(5)]
    k = 6
    agg = SDAAAggregator(avoid_repeats=False)
    sat_prev = None
    expected = []
    for rm in sessions:
        avg = rm.mean(axis=0)
        if sat_prev is None:
            score = avg
        else:
            alpha = sat_prev.max() - sat_prev.min()
            least = int(np.argmin(sat_prev))
            score = (1 - alpha) * avg + alpha * rm[least]
        sel = list(np.argsort(-score, kind="stable")[:k])
        expected.append(sel)
        sat_prev = oracle_satisfaction(rm, sel, k)
    got = [list(agg.aggregate(rm, k)) for rm in sessions]
    assert got == expected


def test_siaa_matches_formula_oracle_over_sessions():
    rng = np.random.default_rng(5)
    sessions = [rng.uniform(0.5, 5.0, size=(4, 25)) for _ in range(5)]
    k, b = 6, 0.4
    agg = SIAAAggregator(b=b, avoid_repeats=False)
    satO_sum = None
    rounds = 0
    sat_prev = None
    expected = []
    for rm in sessions:
        m = rm.shape[0]
        if rounds == 0:
            w = np.full(m, 1 - b)
        else:
            satO = satO_sum / rounds
            user_dis = sat_prev.max() - sat_prev
            w = (1 - b) * (1 - satO) + b * user_dis
        score = w @ rm
        sel = list(np.argsort(-score, kind="stable")[:k])
        expected.append(sel)
        sat_now = oracle_satisfaction(rm, sel, k)
        satO_sum = sat_now if satO_sum is None else satO_sum + sat_now
        rounds += 1
        sat_prev = sat_now
    got = [list(agg.aggregate(rm, k)) for rm in sessions]
    assert got == expected


def test_sdaa_siaa_reset_and_avoid_repeats():
    rng = np.random.default_rng(6)
    sessions = [rng.uniform(0.5, 5.0, size=(3, 20)) for _ in range(3)]
    for agg in [SDAAAggregator(avoid_repeats=True), SIAAAggregator(avoid_repeats=True)]:
        seen = set()
        for rm in sessions:
            out = set(agg.aggregate(rm, 5).tolist())
            assert out.isdisjoint(seen)
            seen |= out
        # reset clears state -> first session reproducible
        agg.reset()
        first_again = agg.aggregate(sessions[0], 5)
        agg.reset()
        assert list(first_again) == list(agg.aggregate(sessions[0], 5))
