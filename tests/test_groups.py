"""Tests for synthetic group formation: the builder's similarity guarantees,
provenance, and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from grouprec import make_blobs_dataset
from grouprec.groups import (
    synthetic, similarity_matrix, build_predicate_group, build_outlier_group,
)


def test_kind_callable_custom_builder():
    """A custom 'kind' builder (sim, size, rng) -> indices plugs in like a custom metric."""
    d = clustered()

    def first_two(sim, size, rng):                 # trivial deterministic builder
        return [0, 1][:size]

    g = synthetic(d, kind=first_two, size=2, n=3, seed=0)
    assert len(g) == 3
    assert g.metadata["kind"] == "first_two"       # provenance records the builder name
    # public building blocks are exposed for composition
    sim = similarity_matrix(d, "pearson")
    rng = np.random.default_rng(0)
    idx = build_predicate_group(sim, 2, lambda r: r >= -1.0, rng)   # always-satisfiable
    assert idx is not None and len(idx) == 2
    assert callable(build_outlier_group)


def test_kind_unknown_string_raises():
    d = clustered()
    with pytest.raises(ValueError, match="unknown kind"):
        synthetic(d, kind="banana", size=2, n=1)


def clustered():
    # well-separated clusters so similar/divergent/outlier groups all exist
    return make_blobs_dataset(n_users=60, n_items=120, n_clusters=4, noise=0.3,
                              density=1.0, seed=0)


def _pairwise(sim, data, members):
    idx = [data.user_index[u] for u in members]
    return [sim[a, b] for i, a in enumerate(idx) for b in idx[i + 1:]]


# --------------------------------------------------------------------------- #
def test_random_groups_size_and_count():
    d = clustered()
    g = synthetic(d, kind="random", size=4, n=10, seed=0)
    assert len(g) == 10
    assert all(len(m) == 4 for m in g)
    assert all(len(set(m.tolist())) == 4 for m in g)  # distinct members
    assert g.metadata["kind"] == "random"


def test_similar_groups_satisfy_high_threshold():
    d = clustered()
    sim = similarity_matrix(d, "pearson")
    g = synthetic(d, kind="similar", size=4, n=10, metric="pearson",
                  sim_high=0.3, seed=1)
    assert len(g) >= 1
    for members in g:
        assert min(_pairwise(sim, d, members)) >= 0.3


def test_divergent_groups_satisfy_low_threshold():
    d = clustered()
    sim = similarity_matrix(d, "pearson")
    g = synthetic(d, kind="divergent", size=3, n=10, metric="pearson",
                  sim_low=0.1, seed=2)
    assert len(g) >= 1
    for members in g:
        assert max(_pairwise(sim, d, members)) <= 0.1


def test_outlier_group_structure():
    d = clustered()
    sim = similarity_matrix(d, "pearson")
    g = synthetic(d, kind="outlier", size=4, n=10, metric="pearson",
                  sim_high=0.3, sim_low=0.1, seed=3)
    assert len(g) >= 1
    members = g[0]
    idx = [d.user_index[u] for u in members]
    # exactly one member is divergent (<= sim_low) to all others; the rest form a
    # similar core (>= sim_high pairwise).
    n_outliers = 0
    for a in idx:
        others = [b for b in idx if b != a]
        if all(sim[a, b] <= 0.1 for b in others):
            n_outliers += 1
    assert n_outliers >= 1


def test_reproducible_with_seed():
    d = clustered()
    g1 = synthetic(d, kind="similar", size=4, n=5, seed=42)
    g2 = synthetic(d, kind="similar", size=4, n=5, seed=42)
    assert [m.tolist() for m in g1] == [m.tolist() for m in g2]


def test_metadata_records_provenance():
    d = clustered()
    g = synthetic(d, kind="similar", size=3, n=4, metric="cosine", seed=7,
                  sim_high=0.5, sim_low=0.2)
    md = g.metadata
    assert md["metric"] == "cosine" and md["size"] == 3 and md["seed"] == 7
    assert md["sim_high"] == 0.5


def test_size_must_be_at_least_two():
    d = clustered()
    with pytest.raises(ValueError):
        synthetic(d, kind="random", size=1, n=1)


def test_unbuildable_raises():
    d = clustered()
    # impossibly strict threshold -> no group can be formed
    with pytest.raises(RuntimeError):
        synthetic(d, kind="similar", size=5, n=5, sim_high=0.999999, max_tries=50)


@pytest.mark.parametrize("metric", ["pearson", "cosine", "jaccard"])
def test_similarity_matrix_shape_and_diagonal(metric):
    d = make_blobs_dataset(n_users=15, n_items=20, density=0.6, seed=0)
    s = similarity_matrix(d, metric)
    assert s.shape == (15, 15)
    assert np.all(np.isnan(np.diag(s)))


def test_custom_similarity_callable_and_matrix():
    import numpy as np
    from grouprec.groups import synthetic, similarity_matrix
    d = clustered()
    # callable custom metric (here: reuse pearson) -> builds groups
    g = synthetic(d, kind="similar", size=3, n=5,
                  metric=lambda data: similarity_matrix(data, "pearson"), seed=0)
    assert len(g) >= 1 and g.metadata["metric"] in ("<lambda>", "custom")
    # precomputed similarity matrix
    S = similarity_matrix(d, "cosine")
    g2 = synthetic(d, kind="similar", size=3, n=5, metric=S, seed=0)
    assert len(g2) >= 1
    # wrong shape rejected
    import pytest as _pt
    with _pt.raises(ValueError):
        similarity_matrix(d, np.zeros((3, 3)))
