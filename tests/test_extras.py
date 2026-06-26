"""Tests for custom metrics, reproducibility, citations, carbon, leaderboard, Pareto."""

from __future__ import annotations

import numpy as np
import pandas as pd

import grouprec as gr
from grouprec import GroupRecommender, benchmark
from grouprec.bench import BenchmarkTask
from grouprec.bench.leaderboard import LeaderboardStore
from grouprec.bench.viz import pareto_front
from grouprec.aggregators import AverageAggregator
from grouprec.backends import Popularity
from grouprec.split import random_split


def _task(seed=0):
    data = gr.make_blobs_dataset(n_users=40, n_items=30, density=0.6, seed=seed)
    groups = gr.groups.synthetic(data, kind="random", size=4, n=10, seed=0)
    return BenchmarkTask("d", data, groups, random_split(data, seed=0)), data, groups


# --- custom metrics -------------------------------------------------------- #
def test_register_custom_metric_and_aggregation():
    gr.eval.register_metric("coverage", lambda rec, g, rel, k: len(set(rec[:k])) / k)
    gr.eval.register_aggregation("p90", lambda v: float(np.percentile(v, 90)))
    task, data, groups = _task()
    rec = GroupRecommender(Popularity(), AverageAggregator())
    rep = gr.evaluate(rec, data, groups, task.splits, protocol="coupled",
                      metrics=["coverage"], group_aggregations=["mean", "p90"])
    d = rep.to_dict()
    assert 0.0 <= d[("coupled", "coverage", 10, "mean")] <= 1.0  # custom metric ran
    assert ("coupled", "coverage", 10, "p90") in d              # custom aggregation ran


# --- reproducibility ------------------------------------------------------- #
def test_experiment_snippet_and_roundtrip(tmp_path):
    gr.set_seed(123)
    exp = gr.Experiment("camra-bridge", seed=42, base="EASE", aggregators=["GFAR", "AVG"])
    snip = exp.snippet()
    assert "gr.set_seed(42)" in snip and "GFAR" in snip
    assert "python" in exp.env
    p = tmp_path / "exp.json"
    exp.save(p)
    loaded = gr.Experiment.load(p)
    assert loaded.seed == 42 and loaded.params["base"] == "EASE"


# --- citations ------------------------------------------------------------- #
def test_citations_present_for_all_models():
    for name in ["GFAR", "EPFuzzDA", "LTP", "AGREE", "GroupIM", "ConsRec", "SDAA", "PAR"]:
        assert "@" in gr.cite(name)
    assert len(gr.references.all()) >= 12


# --- carbon ---------------------------------------------------------------- #
def test_track_emissions_estimates():
    with gr.track_emissions(power_kw=0.05) as em:
        sum(i * i for i in range(10000))
    assert em.seconds > 0 and em.kg_co2e >= 0.0


# --- dynamic leaderboard --------------------------------------------------- #
def test_leaderboard_store_accumulates(tmp_path):
    task, data, groups = _task()
    res = benchmark({"AVG": GroupRecommender(Popularity(), AverageAggregator())},
                    [task], protocols=["coupled"], metrics=["ndcg@10"], silent=True)
    store = LeaderboardStore(tmp_path / "lb.csv")
    store.add(res, tag="run1").add(res, tag="run2")
    df = store.load()
    assert len(df) == 2 * len(res.to_frame())
    best = store.best("ndcg", k=10, protocol="coupled")
    assert "AVG" in set(best["recommender"])


# --- pareto ---------------------------------------------------------------- #
def test_pareto_front_basic():
    # (0.9,0.1) and (0.1,0.9) are non-dominated; (0.5,0.5) dominated? not by either.
    pts = [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (0.2, 0.2)]
    front = set(pareto_front(pts, maximize=(True, True)))
    assert 0 in front and 1 in front and 2 in front  # (0.2,0.2) dominated by (0.5,0.5)
    assert 3 not in front


# --- experiment git/env + static html ------------------------------------- #
def test_experiment_records_git_and_env():
    from grouprec.experiment import git_info
    exp = gr.Experiment("e", seed=1, cite=["GFAR"])
    assert "available" in exp.git                 # git_info ran (repo or not)
    assert isinstance(git_info(), dict)
    assert "@" in exp.citations()["GFAR"]
    assert exp.env.get("cpu_count") is not None


def test_render_static_html_leaderboard(tmp_path):
    from grouprec.bench.leaderboard import render_html
    task, data, groups = _task()
    res = benchmark({"AVG": GroupRecommender(Popularity(), AverageAggregator())},
                    [task], protocols=["coupled"], metrics=["ndcg@10"], silent=True)
    out = tmp_path / "lb.html"
    render_html(res, out, metric="ndcg", k=10, protocol="coupled")
    html = out.read_text()
    assert "<table" in html and "grouprec leaderboard" in html


# --- profile-first + list metrics ----------------------------------------- #
def test_profile_first_recommender_and_list_metrics():
    from grouprec import ProfileGroupRecommender
    from grouprec.backends import EASE
    task, data, groups = _task()
    pf = ProfileGroupRecommender(EASE(), merge="average")
    assert pf.paradigm == "profile"
    rep = gr.evaluate(pf, data, groups, task.splits, protocol="coupled",
                      metrics=["ndcg@10"], list_metrics=["novelty", "list_coverage"])
    aggs = set(rep.to_frame()["aggregation"])
    assert "list" in aggs
    d = rep.to_dict()
    assert ("-", "novelty", 10, "list") in d and d[("-", "novelty", 10, "list")] >= 0.0


def test_register_custom_list_metric():
    from grouprec.backends import EASE
    from grouprec import ProfileGroupRecommender
    gr.eval.register_list_metric("half", lambda rec, k, ctx: 0.5)
    task, data, groups = _task()
    rep = gr.evaluate(ProfileGroupRecommender(EASE()), data, groups, task.splits, k=5,
                      protocol="coupled", metrics=["ndcg"], list_metrics=["half"])
    assert rep.to_dict()[("-", "half", 5, "list")] == 0.5


def test_profile_first_requires_score_profile():
    from grouprec import ProfileGroupRecommender
    from grouprec.backends import Popularity
    import pytest as _pt
    task, data, groups = _task()
    with _pt.raises(TypeError):
        ProfileGroupRecommender(Popularity()).fit(data)  # Popularity has no score_profile


def test_experiment_context_manager_writes_run_folder(tmp_path):
    import json
    rd = tmp_path / "run1"
    with gr.Experiment("demo", seed=7, dir=rd, cite=["GFAR"]) as exp:
        exp.log(hr=0.62)
        exp.attach("leaderboard", pd.DataFrame({"rec": ["AVG"], "ndcg": [0.5]}))
    assert (rd / "config.json").exists() and (rd / "env.json").exists()
    assert (rd / "results.json").exists() and (rd / "leaderboard.csv").exists()
    assert (rd / "citations.bib").read_text().strip()          # GFAR bibtex written
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["seed"] == 7 and "entry_script" in cfg and "cwd" in cfg
    assert json.loads((rd / "results.json").read_text())["hr"] == 0.62
    # round-trip from the folder
    loaded = gr.Experiment.load(rd)
    assert loaded.seed == 7


def test_experiment_context_manager_finalizes_on_error(tmp_path):
    import json
    rd = tmp_path / "boom"
    try:
        with gr.Experiment("boom", dir=rd):
            raise ValueError("kaboom")
    except ValueError:
        pass
    assert (rd / "config.json").exists()
    assert "kaboom" in json.loads((rd / "results.json").read_text())["error"]
