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


def _similarity_among(data: Dataset, user_ids, metric) -> np.ndarray:
    """Pairwise user-user similarity among *just* ``user_ids`` (diagonal = ``nan``).

    Pearson/cosine/jaccard are pairwise-independent, so we compute them directly on the
    members' rows (exact, and cheap even for huge user bases); callable/precomputed
    metrics fall back to the full :func:`similarity_matrix` and are subset."""
    idx = np.array([data.user_index[u] for u in user_ids], dtype=int)
    if idx.size <= 1:
        return np.full((idx.size, idx.size), np.nan)
    if not isinstance(metric, str):                                  # callable / precomputed
        S = similarity_matrix(data, metric)[np.ix_(idx, idx)]
    elif metric == "jaccard":
        b = (data.user_item_matrix(value="binary") > 0)[idx].astype(float)
        inter = b @ b.T
        sizes = b.sum(1)
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(union > 0, inter / union, np.nan)
    elif metric in ("pearson", "cosine"):
        m = data.user_item_matrix(value="rating")[idx]
        if metric == "pearson":
            with np.errstate(invalid="ignore"):
                S = np.corrcoef(m)
        else:
            norm = np.linalg.norm(m, axis=1)
            denom = np.outer(norm, norm)
            with np.errstate(divide="ignore", invalid="ignore"):
                S = np.where(denom > 0, (m @ m.T) / denom, np.nan)
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'pearson'/'cosine'/'jaccard', "
                         "a callable, or a precomputed matrix.")
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, np.nan)
    return S


def group_similarity(data: Dataset, members, *, other=None, metric="pearson",
                     reduce: str | None = "mean"):
    """User-user similarity for a group -- or *between* two (sub)groups.

    Works for **any** group (synthetic or a real group from data), and composes to
    arbitrary sub-groups because ``members`` and ``other`` are just user-id lists:

    * within a group / sub-group -- ``group_similarity(data, members)`` gives the mean
      pairwise similarity (the "cohesion"); pass a subset for a sub-group's cohesion.
    * between two (sub-)groups -- pass ``other=[...]`` to get the mean cross-similarity,
      e.g. an outlier's ``[u4]`` against its similar core ``[u1, u2, u3]``, or any two
      size-2 halves against each other.

    Parameters
    ----------
    metric : ``"pearson"`` / ``"cosine"`` / ``"jaccard"``, a callable ``f(data) -> (n,n)``
        matrix, or a precomputed ``(n_users, n_users)`` matrix -- the same options as
        :func:`synthetic`, so cohesion is measured with the metric groups were formed by.
    reduce : ``"mean"`` (default) / ``"min"`` / ``"max"`` / ``"median"`` over the relevant
        pairs, or ``None`` to return the raw similarity matrix (within) / block (cross);
        ``nan`` self- and undefined pairs are ignored.

    Returns a float (reduced) or an ``np.ndarray`` (``reduce=None``); an all-``nan`` set
    of pairs (e.g. a singleton group) reduces to ``nan``.
    """
    members = list(members)
    if other is None:
        S = _similarity_among(data, members, metric)
        if reduce is None:
            return S
        vals = S[np.triu_indices(len(members), k=1)]
    else:
        other = list(other)
        S = _similarity_among(data, members + other, metric)
        block = S[:len(members), len(members):]
        if reduce is None:
            return block
        vals = block.ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    fns = {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median}
    if reduce not in fns:
        raise ValueError(f"reduce must be one of {sorted(fns)} or None; got {reduce!r}")
    return float(fns[reduce](vals))


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


def derive_group_interactions(
    data: Dataset,
    groups: Groups,
    *,
    like_threshold: float = 4.0,
    min_members: int = 2,
    predicate=None,
) -> dict[int, list]:
    """Derive a per-group item signal from the members' *individual* feedback.

    When only individual user--item data and (synthetic) groups are available, deep
    group models and group-level evaluation still need a per-group signal (which items
    "belong" to a group). Rather than *simulating* group choices, this synthesises one
    deterministically from what the members themselves liked, under a transparent and
    fully overridable rule.

    Default rule -- *majority consensus*: an item is a group interaction if at least
    ``min_members`` members liked it, where a "like" is ``rating >= like_threshold``
    (or any recorded interaction when the dataset has no ratings). This is the notion
    the interactive case study surfaces as a group's "consensus items".

    Parameters
    ----------
    like_threshold : rating at/above which a member is considered to like an item.
    min_members : how many members must like an item for it to count (majority rule).
    predicate : optional ``f(n_likes, n_members) -> bool`` that overrides the majority
        rule entirely, e.g. ``lambda l, m: l == m`` (unanimity) or ``lambda l, m: l >= 1``
        ("any member"). It receives, per candidate item, the number of members who liked
        it and the group size.

    Returns
    -------
    dict[int, list]
        Group index -> list of item ids, the format consumed by the deep models and by
        :func:`grouprec.models.data.make_synthetic_group_data`.
    """
    from collections import Counter

    df = data.interactions
    liked_df = df[df["rating"] >= like_threshold] if data.has_ratings else df
    liked = {int(u): set(int(i) for i in sub["item"]) for u, sub in liked_df.groupby("user")}
    rule = predicate or (lambda n_likes, n_members: n_likes >= min_members)

    out: dict[int, list] = {}
    for gi, members in enumerate(groups):
        counts: Counter = Counter()
        for u in members:
            counts.update(liked.get(int(u), ()))
        m = len(list(members))
        out[gi] = [it for it, c in counts.items() if rule(c, m)]
    return out


def seen_items(data: Dataset, members, *, by: str | None = "any") -> set:
    """Item ids a group's members have already consumed -- the set usually excluded when
    recommending to a group (don't recommend what a member already has).

    The two families use different candidate conventions: aggregators rank *all* items
    minus the seen set, while the deep-model literature ranks a sampled 1-vs-N pool
    (see :func:`sample_candidates`). ``by`` parametrizes which "seen" set to exclude:

    * ``"any"`` (default) -- items consumed by **at least one** member. Matches the
      ``exclude_seen=True`` convention used by the evaluators.
    * ``"all"``           -- only items consumed by **every** member.
    * ``None``            -- the empty set (rank everything).

    Returns a ``set`` of item ids; pass it straight to ``recommend(..., exclude=...)``.
    """
    if by is None:
        return set()
    ui = data.user_index
    per_member = [set(int(x) for x in data.items[data.items_seen_by(u)])
                  for u in members if u in ui]
    if not per_member:
        return set()
    if by == "any":
        return set().union(*per_member)
    if by == "all":
        return set.intersection(*per_member)
    raise ValueError(f"by must be 'any', 'all', or None; got {by!r}")


def candidate_items(data: Dataset, members, *, exclude_seen: str | None = "any") -> np.ndarray:
    """Full-ranking candidate item ids for a group: all dataset items minus the seen set
    (``exclude_seen`` = ``"any"`` / ``"all"`` / ``None``; see :func:`seen_items`).

    This is the candidate convention typical for *aggregators*. Returns ids in dataset
    order; pass to ``recommend(..., candidates=...)`` (equivalently, use
    :func:`seen_items` with ``recommend(..., exclude=...)``)."""
    seen = seen_items(data, members, by=exclude_seen)
    return np.array([int(it) for it in data.items if int(it) not in seen], dtype=np.int64)


def sample_candidates(data: Dataset, members, positives, *, n_negatives: int,
                      exclude_seen: str | None = "any", seed: int | None = None) -> list:
    """Sampled 1-vs-N candidate list -- the protocol used by the deep GRS models
    (GroupIM / AGREE / ConsRec): the ``positives`` item id(s) plus ``n_negatives`` ids
    sampled uniformly without replacement from items no member has consumed
    (``exclude_seen`` policy) and not already a positive.

    Returns ``[*positives, *negatives]``; pass to ``recommend(..., candidates=...)``.
    Fewer than ``n_negatives`` are returned if the pool is smaller."""
    positives = ([int(positives)] if isinstance(positives, (int, np.integer))
                 else [int(p) for p in positives])
    seen = seen_items(data, members, by=exclude_seen)
    posset = set(positives)
    pool = [int(it) for it in data.items if int(it) not in seen and int(it) not in posset]
    k = min(int(n_negatives), len(pool))
    negs = np.random.default_rng(seed).choice(pool, size=k, replace=False).tolist() if k else []
    return positives + [int(x) for x in negs]


def _meta(kind, size, n, metric, seed, sim_high, sim_low) -> dict:
    return {
        "kind": kind, "size": size, "n_requested": n,
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        "seed": seed, "sim_high": sim_high, "sim_low": sim_low,
    }
