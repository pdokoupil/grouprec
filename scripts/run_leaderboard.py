#!/usr/bin/env python
"""Run a benchmark and (re)generate a leaderboard: CSV + accumulating store + static
HTML page (for GitHub Pages).

    python scripts/run_leaderboard.py                       # default demo (no heavy deps)
    python scripts/run_leaderboard.py --out results/lb.csv --html docs/leaderboard.html

The default config uses the dependency-free EASE backend on a synthetic dataset, so it
runs anywhere. Edit `build_tasks` / `build_recommenders` to benchmark your own
datasets/recommenders, or import and call `run(...)` from your own experiment script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import grouprec as gr
from grouprec import GroupRecommender, benchmark
from grouprec.bench import BenchmarkTask
from grouprec.bench.leaderboard import LeaderboardStore, render_html
from grouprec.backends import EASE


def build_tasks(seed: int = 0):
    data = gr.make_blobs_dataset(n_users=120, n_items=80, n_clusters=4, density=0.5, seed=seed)
    groups = gr.groups.synthetic(data, kind="divergent", size=4, n=40, metric="pearson",
                                 sim_low=0.2, seed=seed)
    folds = gr.split.crossval(data, k=3, seed=seed)
    return [BenchmarkTask("blobs", data, groups, folds)]


def build_recommenders():
    return {name: GroupRecommender(EASE(reg=100.0), gr.aggregators.get(name), normalize="minmax")
            for name in ["AVG", "LMS", "GFAR", "GreedyLM", "EPFuzzDA", "RLProp", "LTP"]}


def run(out_csv: str, html: str | None, store_csv: str | None, tag: str) -> None:
    res = benchmark(build_recommenders(), build_tasks(),
                    protocols=["coupled", "decoupled"], metrics=["ndcg@10", "recall@10"],
                    group_aggregations=["mean", "min", "minmax"], silent=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv)
    print(f"wrote {out_csv} ({len(res.to_frame())} rows)")
    if store_csv:
        LeaderboardStore(store_csv).add(res, tag=tag)
        print(f"appended to leaderboard store {store_csv} (tag={tag})")
    if html:
        render_html(res, html, metric="ndcg", k=10, aggregation="min", protocol="decoupled",
                    meta="ndcg@10 · min (fairness) · decoupled — grouprec demo")
        print(f"wrote static leaderboard {html}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/leaderboard.csv")
    p.add_argument("--html", default=None, help="static HTML leaderboard path (GitHub Pages)")
    p.add_argument("--store", default=None, help="accumulating LeaderboardStore CSV path")
    p.add_argument("--tag", default="demo")
    a = p.parse_args()
    run(a.out, a.html, a.store, a.tag)


if __name__ == "__main__":
    main()
