"""Leaderboard plotting (optional ``[viz]`` extra: matplotlib)."""

from __future__ import annotations


def plot_leaderboard(result, metric: str, *, k: int | None = None,
                     aggregation: str = "mean", protocol: str = "coupled", ax=None):
    """Bar chart of one metric across recommenders (grouped by dataset).

    Requires matplotlib (``pip install grouprec[viz]``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_leaderboard requires the 'viz' extra: "
                          "pip install grouprec[viz]") from exc

    table = result.leaderboard(metric, k=k, aggregation=aggregation, protocol=protocol)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(table)), 4))
    table.plot.bar(ax=ax)
    ax.set_ylabel(f"{metric}" + (f"@{k}" if k else "") + f" ({aggregation}, {protocol})")
    ax.set_title("grouprec leaderboard")
    ax.legend(title="dataset", fontsize="small")
    return ax


def pareto_front(points, *, maximize=(True, True)):
    """Indices of the non-dominated points in a list of (x, y) tuples.

    ``maximize`` says whether higher is better on each axis (e.g. relevance up,
    fairness up). Useful for the utility-vs-fairness trade-off.
    """
    import numpy as np
    pts = np.asarray(points, dtype=float)
    sign = np.array([1.0 if m else -1.0 for m in maximize])
    s = pts * sign
    keep = []
    for i in range(len(s)):
        dominated = False
        for j in range(len(s)):
            if j != i and np.all(s[j] >= s[i]) and np.any(s[j] > s[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def plot_pareto(result, x_metric, y_metric, *, k=None, x_agg="mean", y_agg="min",
                protocol="coupled", dataset=None, maximize=(True, True), ax=None):
    """Scatter recommenders in (x_metric, y_metric) space with the Pareto front drawn
    -- e.g. relevance (ndcg.mean) vs fairness (ndcg.min). Requires the ``viz`` extra.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_pareto requires the 'viz' extra: pip install grouprec[viz]") from exc

    df = result.to_frame()
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    if k is not None:
        df = df[df["k"] == k]

    def series(metric, agg):
        s = df[(df.metric == metric) & (df.aggregation == agg) & (df.protocol == protocol)]
        return s.groupby("recommender")["value"].mean()

    xs, ys = series(x_metric, x_agg), series(y_metric, y_agg)
    names = sorted(set(xs.index) & set(ys.index))
    pts = [(xs[n], ys[n]) for n in names]
    front = set(pareto_front(pts, maximize=maximize))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for i, n in enumerate(names):
        on = i in front
        ax.scatter(*pts[i], c="crimson" if on else "gray", s=70 if on else 40, zorder=3 if on else 2)
        ax.annotate(n, pts[i], fontsize=8, xytext=(4, 4), textcoords="offset points")
    fp = sorted([pts[i] for i in front])
    if fp:
        ax.plot([p[0] for p in fp], [p[1] for p in fp], "--", c="crimson", alpha=0.6)
    ax.set_xlabel(f"{x_metric}.{x_agg}"); ax.set_ylabel(f"{y_metric}.{y_agg}")
    ax.set_title(f"Pareto front ({protocol})")
    return ax


__all__ = ["plot_leaderboard", "pareto_front", "plot_pareto"]
