"""Synthetic group formation (scenario 1).

``synthetic(data, kind=..., size=..., n=..., metric=...)`` builds groups by
iterative member addition under a similarity predicate, returning a typed
:class:`~grouprec.data.Groups` with full provenance.

Kinds (cf. the Coupled/Decoupled and group-formation literature):

* ``random``    -- uniformly random members.
* ``similar``   -- every pairwise similarity ``>= sim_high`` (default Pearson >= 0.3).
* ``divergent`` / ``dissimilar`` -- every pairwise similarity ``<= sim_low`` (<= 0.1).
* ``outlier``   -- a similar core of ``size-1`` plus one member divergent to all of them.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset, Groups


def similarity_matrix(data: Dataset, metric="pearson") -> np.ndarray:
    """User-user similarity matrix (``n_users x n_users``); diagonal is ``nan``.

    ``metric`` may be:

    * a **string** built-in: ``"pearson"`` / ``"cosine"`` (on the rating matrix) or
      ``"jaccard"`` (on the binary interaction sets);
    * a **callable** ``f(data) -> (n_users, n_users) array`` for a custom metric or
      custom features (e.g. compute similarity over side-information embeddings);
    * a **precomputed** ``(n_users, n_users)`` array (rows aligned to ``data.users``).

    Pairs with undefined similarity (e.g. zero-variance rows for Pearson) are ``nan``
    and therefore never satisfy a similarity predicate.
    """
    if callable(metric):
        s = np.asarray(metric(data), dtype=float)
    elif not isinstance(metric, str):
        s = np.asarray(metric, dtype=float)  # precomputed matrix
    elif metric == "jaccard":
        b = data.user_item_matrix(value="binary") > 0
        inter = b @ b.T.astype(float)
        sizes = b.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(union > 0, inter / union, np.nan)
    else:
        m = data.user_item_matrix(value="rating")
        if metric == "pearson":
            with np.errstate(invalid="ignore"):
                s = np.corrcoef(m)
        elif metric == "cosine":
            norm = np.linalg.norm(m, axis=1)
            denom = np.outer(norm, norm)
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(denom > 0, (m @ m.T) / denom, np.nan)
        else:
            raise ValueError(f"unknown metric {metric!r}; use pearson, cosine, or jaccard.")
    s = np.array(s, dtype=float)
    if s.shape != (data.n_users, data.n_users):
        raise ValueError(
            f"similarity matrix must be ({data.n_users}, {data.n_users}); got {s.shape}.")
    np.fill_diagonal(s, np.nan)
    return s


def build_predicate_group(sim, size, predicate, rng, max_tries=1000):
    """Grow a group of ``size`` users where each added member satisfies ``predicate``
    against *all* current members. Returns user indices or ``None`` if it stalls.

    Public building block for custom ``kind`` callables: ``predicate(row)`` is a boolean
    mask over a similarity row (e.g. ``lambda r: r >= 0.3`` for "similar")."""
    n = sim.shape[0]
    for _ in range(max_tries):
        seed_user = int(rng.integers(n))
        chosen = [seed_user]
        ok = True
        while len(chosen) < size:
            # candidates satisfying the predicate w.r.t. every current member
            cand_mask = np.ones(n, dtype=bool)
            cand_mask[chosen] = False
            for member in chosen:
                cand_mask &= predicate(sim[member])
            cands = np.flatnonzero(cand_mask)
            if cands.size == 0:
                ok = False
                break
            chosen.append(int(rng.choice(cands)))
        if ok:
            return chosen
    return None


def synthetic(
    data: Dataset,
    kind: str = "similar",
    size: int = 4,
    n: int = 100,
    metric="pearson",
    seed: int | None = None,
    sim_high: float = 0.3,
    sim_low: float = 0.1,
    max_tries: int = 1000,
) -> Groups:
    """Generate up to ``n`` synthetic groups of ``size`` members.

    Returns a :class:`~grouprec.data.Groups`; raises if not a single group of the
    requested kind/size can be formed from the data.
    """
    if size < 2:
        raise ValueError("group size must be >= 2.")
    rng = np.random.default_rng(seed)
    kind_name = kind if isinstance(kind, str) else getattr(kind, "__name__", "custom")

    if kind == "random":
        members = [np.sort(data.users[rng.choice(data.n_users, size=size, replace=False)])
                   for _ in range(n)]
        return Groups(members, metadata=_meta(kind_name, size, n, metric, seed, sim_high, sim_low))

    sim = similarity_matrix(data, metric)
    high = lambda row: row >= sim_high  # noqa: E731
    low = lambda row: row <= sim_low    # noqa: E731

    # A "kind" is a builder ``f(sim, size, rng) -> list[int] | None`` (member indices, or
    # None if it stalls). Custom kinds (e.g. a 2+2 clustered group) plug in here exactly as a
    # custom ``metric`` does -- see ``build_predicate_group`` / ``build_outlier_group`` and docs.
    if callable(kind):
        builder = lambda: kind(sim, size, rng)                          # noqa: E731
    elif kind == "similar":
        builder = lambda: build_predicate_group(sim, size, high, rng, max_tries)      # noqa: E731
    elif kind in ("divergent", "dissimilar"):
        builder = lambda: build_predicate_group(sim, size, low, rng, max_tries)       # noqa: E731
    elif kind == "outlier":
        builder = lambda: build_outlier_group(sim, size, high, low, rng, max_tries)   # noqa: E731
    else:
        raise ValueError(f"unknown kind {kind!r}; use a builtin name or a callable builder.")

    members: list[np.ndarray] = []
    for _ in range(n):
        idx = builder()
        if idx is None:
            break
        members.append(np.sort(data.users[np.asarray(idx)]))

    if not members:
        raise RuntimeError(
            f"could not form any '{kind_name}' group of size {size} "
            f"(metric={metric}, sim_high={sim_high}, sim_low={sim_low}); "
            "try a different metric, thresholds, or builder."
        )
    return Groups(members, metadata=_meta(kind_name, size, n, metric, seed, sim_high, sim_low))


def build_outlier_group(sim, size, high, low, rng, max_tries=1000):
    """A similar core of ``size-1`` plus one member divergent to all of them.
    Public building block for custom ``kind`` callables."""
    for _ in range(max_tries):
        core = build_predicate_group(sim, size - 1, high, rng, max_tries=50)
        if core is None:
            continue
        cand_mask = np.ones(sim.shape[0], dtype=bool)
        cand_mask[core] = False
        for member in core:
            cand_mask &= low(sim[member])
        cands = np.flatnonzero(cand_mask)
        if cands.size:
            return core + [int(rng.choice(cands))]
    return None


def _meta(kind, size, n, metric, seed, sim_high, sim_low) -> dict:
    return {
        "kind": kind, "size": size, "n_requested": n,
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        "seed": seed, "sim_high": sim_high, "sim_low": sim_low,
    }
