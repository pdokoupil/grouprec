"""Benchmark runner: one call -> a leaderboard across datasets x recommenders x
protocols, into a tidy CSV.

It drives every applicable (task, recommender, protocol) combination and **skips
invalid ones gracefully** -- e.g. decoupled is only run for recommenders exposing a
base RS (``.base.score``), so profile-first / deep models are automatically limited
to coupled. Group-level tasks (the bridge regime) run coupled-only group LOO, letting
results-aggregators and deep models share one leaderboard.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from ..profiling import track_emissions

from ..data import Dataset, Groups
from ..eval import evaluate, evaluate_grouplevel, evaluate_sampled, evaluate_sequential
from ..eval.sequential import LONG_TERM_METRICS


@dataclass
class BenchmarkTask:
    """One dataset + group configuration to benchmark recommenders on.

    level="member" -> per-member coupled/decoupled (Regime A, synthetic groups on
    ML/KGRec/Last.fm). level="group" -> coupled group-level LOO (Regime B, real/
    inferred groups on CAMRa/Mafengwo/Yelp/Douban/Weeplaces; bridges the rift).
    """

    name: str
    data: Dataset
    groups: Groups
    splits: object                      # Split | list[Split]
    level: str = "member"               # "member" | "group" | "sampled"
    group_truth: dict | None = None     # required when level == "group"
    test_instances: list | None = None  # required when level == "sampled" (1+N ranking)
    sequential: bool = False
    n_rounds: int = 5


def _has_base(rec) -> bool:
    base = getattr(rec, "base", None)
    return base is not None and hasattr(base, "score")


def benchmark(
    recommenders: dict,
    tasks: Sequence[BenchmarkTask],
    *,
    protocols: Sequence[str] = ("coupled", "decoupled"),
    metrics: Sequence[str] = ("ndcg", "recall", "hr"),
    group_aggregations: Sequence[str] = ("mean", "min", "minmax"),
    k: int = 10,
    sampled_ks: Sequence[int] = (5, 10),
    long_term_metrics: Sequence[str] = LONG_TERM_METRICS,
    on_error: str = "skip",
    silent: bool = False,
    track_carbon: bool = False,
) -> "BenchmarkResult":
    """Run the grid and collect a long-format leaderboard.

    Parameters
    ----------
    recommenders : ``{name: recommender}`` where each value is a recommender instance
        or a zero-arg factory returning one (factories get a fresh model per task).
    tasks : the datasets/group configs to evaluate on.
    protocols : requested protocols; ``decoupled`` is auto-dropped for recommenders
        without a base RS, and ignored entirely for ``level="group"`` tasks.
    on_error : ``"skip"`` (warn and continue) or ``"raise"``.
    silent : suppress the warnings emitted when protocols/recommenders are dropped.
    """

    def _warn(msg: str) -> None:
        if not silent:
            warnings.warn(msg, stacklevel=2)

    records: list[dict] = []
    for task in tasks:
        requested_decoupled = "decoupled" in protocols
        for rname, spec in recommenders.items():
            rec = spec() if (callable(spec) and not hasattr(spec, "recommend")) else spec
            paradigm = getattr(rec, "paradigm", "results")
            def _run():
                if task.level == "sampled":
                    if task.test_instances is None:
                        raise ValueError(f"task {task.name!r} is sampled but has no test_instances")
                    if requested_decoupled:
                        _warn(f"task {task.name!r} is sampled group-level (coupled-only); "
                              f"ignoring 'decoupled' for {rname!r}.")
                    return evaluate_sampled(rec, task.data, task.groups, task.test_instances,
                                            ks=sampled_ks)
                if task.level == "group":
                    if task.group_truth is None:
                        raise ValueError(f"task {task.name!r} is group-level but has no group_truth")
                    if requested_decoupled:
                        _warn(f"task {task.name!r} is group-level (coupled-only); "
                              f"ignoring 'decoupled' for {rname!r}.")
                    return evaluate_grouplevel(rec, task.data, task.groups, task.splits,
                                               task.group_truth, k=k, metrics=metrics)
                protos = [p for p in protocols if p == "coupled" or _has_base(rec)]
                if requested_decoupled and "decoupled" not in protos:
                    _warn(f"{rname!r} on {task.name!r}: no base RS (.base.score); "
                          f"dropping 'decoupled' protocol (coupled only).")
                if not protos:
                    _warn(f"{rname!r} on {task.name!r}: no applicable protocol; skipped.")
                    return None
                if task.sequential:
                    return evaluate_sequential(
                        rec, task.data, task.groups, task.splits, n_rounds=task.n_rounds,
                        k=k, protocol=protos, metrics=metrics,
                        group_aggregations=group_aggregations,
                        long_term_metrics=long_term_metrics)
                return evaluate(rec, task.data, task.groups, task.splits, k=k,
                                protocol=protos, metrics=metrics,
                                group_aggregations=group_aggregations)

            em = None
            try:
                cm = track_emissions() if track_carbon else nullcontext()
                with cm as em:
                    rep = _run()
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                _warn(f"skipped {rname!r} on {task.name!r}: {exc}")
                continue
            if rep is None:
                continue
            for r in rep.records:
                records.append({"dataset": task.name, "recommender": rname,
                                "paradigm": paradigm, **r})
            if track_carbon and em is not None:
                records.append({"dataset": task.name, "recommender": rname, "paradigm": paradigm,
                                "protocol": "-", "metric": "carbon_kg", "k": 0,
                                "aggregation": "run", "value": em.kg_co2e})
    return BenchmarkResult(records)


@dataclass
class BenchmarkResult:
    """Tidy leaderboard with CSV export and pivoting helpers."""

    records: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        cols = ["dataset", "recommender", "paradigm", "protocol", "metric", "k",
                "aggregation", "value"]
        return pd.DataFrame(self.records, columns=cols)

    def to_csv(self, path, **kwargs) -> None:
        self.to_frame().to_csv(path, index=False, **kwargs)

    def leaderboard(self, metric: str, *, k: int | None = None, aggregation: str = "mean",
                    protocol: str = "coupled") -> pd.DataFrame:
        """recommenders x datasets table for one (metric, k, aggregation, protocol)."""
        df = self.to_frame()
        sel = (df["metric"] == metric) & (df["aggregation"] == aggregation) & (df["protocol"] == protocol)
        if k is not None:
            sel &= df["k"] == k
        return df[sel].pivot_table(index="recommender", columns="dataset", values="value")

    def best(self, metric: str, *, aggregation: str = "mean", protocol: str = "coupled",
             ascending: bool = False) -> pd.DataFrame:
        df = self.to_frame()
        sel = (df["metric"] == metric) & (df["aggregation"] == aggregation) & (df["protocol"] == protocol)
        return (df[sel].groupby("recommender")["value"].mean()
                .sort_values(ascending=ascending).to_frame("value"))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BenchmarkResult({len(self.records)} records)"


__all__ = ["benchmark", "BenchmarkTask", "BenchmarkResult"]
