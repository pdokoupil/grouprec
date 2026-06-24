"""Shared evaluation helpers: the Report container, metric-spec parsing, and
ground-truth builders used by both single-shot and sequential evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import BASE_METRICS, LIST_METRICS


@dataclass
class Report:
    """Tidy evaluation results: one record per (protocol, metric, k, aggregation)."""

    records: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["protocol", "metric", "k", "aggregation", "value"])

    def to_dict(self) -> dict:
        return {(r["protocol"], r["metric"], r["k"], r["aggregation"]): r["value"]
                for r in self.records}

    def pivot(self) -> pd.DataFrame:
        df = self.to_frame()
        df["name"] = df["metric"] + "@" + df["k"].astype(str) + "." + df["aggregation"]
        return df.pivot_table(index="name", columns="protocol", values="value")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Report({len(self.records)} records)\n{self.to_frame().to_string(index=False)}"


def parse_metric(spec: str, default_k: int) -> tuple[str, int]:
    name, _, ks = spec.partition("@")
    name = name.lower()
    if name not in BASE_METRICS:
        raise ValueError(f"unknown metric {name!r}; available: {sorted(BASE_METRICS)}")
    return name, (int(ks) if ks else default_k)


def parse_list_metric(spec: str, default_k: int) -> tuple[str, int]:
    name, _, ks = spec.partition("@")
    name = name.lower()
    if name not in LIST_METRICS:
        raise ValueError(f"unknown list metric {name!r}; available: {sorted(LIST_METRICS)}")
    return name, (int(ks) if ks else default_k)


def coupled_ground_truth(test: pd.DataFrame, binarize: bool, threshold: float) -> dict:
    """Per-user {gains: {item: gain}, relevant: set(items)} from held-out feedback."""
    has_r = "rating" in test.columns
    gt: dict = {}
    for user, grp in test.groupby("user", sort=False):
        items = grp["item"].to_numpy()
        if has_r:
            ratings = grp["rating"].to_numpy(dtype=float)
            relevant = set(int(it) for it in items[ratings >= threshold])
            if binarize:
                gains = {int(it): 1.0 for it in items[ratings >= threshold]}
            else:
                gains = {int(it): float(r) for it, r in zip(items, ratings)}
        else:  # implicit feedback: every test item is relevant with gain 1
            relevant = set(int(it) for it in items)
            gains = {int(it): 1.0 for it in items}
        gt[user] = {"gains": gains, "relevant": relevant}
    return gt


def decoupled_member(scores_u: np.ndarray, item_ids: np.ndarray, k: int) -> tuple[dict, set]:
    """Per-member ground truth from predicted scores: gains = max(0, r̂) for all
    items, relevant = the member's own ideal top-k."""
    gains = {int(it): float(max(0.0, s)) for it, s in zip(item_ids, scores_u)}
    ideal = item_ids[np.argsort(-scores_u, kind="stable")[:k]]
    return gains, set(int(it) for it in ideal)


__all__ = ["Report", "parse_metric", "coupled_ground_truth", "decoupled_member"]
