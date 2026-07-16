"""Tests for Dataset, splitters, backends, and the GroupRecommender glue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grouprec as gr
from grouprec import Dataset, GroupRecommender, make_blobs_dataset, split
from grouprec.aggregators import AverageAggregator, LTPAggregator
from grouprec.backends import Popularity, Random, BaseRecommender


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def small_dataset():
    # item popularity (count): 10 > 20 > 30 ; ratings differ too
    rows = [
        (1, 10, 5.0), (2, 10, 4.0), (3, 10, 3.0),
        (1, 20, 2.0), (2, 20, 5.0),
        (1, 30, 1.0),
    ]
    return Dataset(pd.DataFrame(rows, columns=["user", "item", "rating"]))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_dataset_basics_and_indexing():
    d = small_dataset()
    assert d.n_users == 3 and d.n_items == 3
    assert list(d.items) == [10, 20, 30]
    assert d.item_index[20] == 1
    assert d.has_ratings and not d.has_timestamps


def test_user_item_matrix_rating_and_binary():
    d = small_dataset()
    m = d.user_item_matrix(value="rating")
    assert m.shape == (3, 3)
    # user 1 (row 0): item10=5, item20=2, item30=1
    assert m[0].tolist() == [5.0, 2.0, 1.0]
    b = d.user_item_matrix(value="binary")
    assert b[0].tolist() == [1.0, 1.0, 1.0]
    assert b[2].tolist() == [1.0, 0.0, 0.0]  # user 3 only rated item 10


def test_items_seen_by_and_binarize():
    d = small_dataset()
    assert sorted(d.items_seen_by(1).tolist()) == [0, 1, 2]
    assert d.items_seen_by(3).tolist() == [0]
    b = d.binarize(threshold=4.0)
    assert not b.has_ratings
    assert len(b) == 3  # ratings >=4: (1,10,5),(2,10,4),(2,20,5)


def test_from_pandas_renames():
    df = pd.DataFrame({"u": [1, 2], "i": [10, 20], "r": [3.0, 4.0]})
    d = Dataset.from_pandas(df, user_col="u", item_col="i", rating_col="r")
    assert d.has_ratings and d.n_users == 2


# --------------------------------------------------------------------------- #
# splitters
# --------------------------------------------------------------------------- #
def test_random_split_partitions():
    d = make_blobs_dataset(n_users=30, n_items=20, density=0.5, seed=1)
    s = split.random_split(d, test_frac=0.25, seed=0)
    assert len(s.train) + len(s.test) == len(d)
    assert round(len(s.test) / len(d), 1) == 0.2 or abs(len(s.test) - 0.25 * len(d)) <= 1


def test_crossval_covers_all_once():
    d = make_blobs_dataset(n_users=20, n_items=15, density=0.6, seed=2)
    folds = split.crossval(d, k=5, seed=0)
    assert len(folds) == 5
    test_sizes = sum(len(f.test) for f in folds)
    assert test_sizes == len(d)


def test_leave_one_out_one_per_user():
    d = make_blobs_dataset(n_users=25, n_items=20, density=0.7, seed=3)
    s = split.leave_one_out(d, seed=0)
    assert len(s.test) == d.n_users
    assert s.test["user"].nunique() == d.n_users
    assert len(s.train) + len(s.test) == len(d)


def test_leave_one_out_uses_latest_timestamp():
    rows = [(1, 10, 1.0, 100), (1, 20, 1.0, 200), (1, 30, 1.0, 50)]
    d = Dataset(pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"]))
    s = split.leave_one_out(d)
    assert s.test["item"].tolist() == [20]  # the latest


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def test_popularity_protocol_and_ranking():
    d = small_dataset()
    pop = Popularity(measure="count").fit(d)
    assert isinstance(pop, BaseRecommender)
    scores = pop.score([1, 2])  # any users -> same row
    assert scores.shape == (2, 3)
    # item 10 most popular (3), then 20 (2), then 30 (1)
    assert np.argmax(scores[0]) == d.item_index[10]
    assert scores[0].tolist() == scores[1].tolist()


def test_popularity_mean_measure():
    d = small_dataset()
    pop = Popularity(measure="mean").fit(d)
    s = pop.score([1])[0]
    # mean ratings: item10=(5+4+3)/3=4, item20=(2+5)/2=3.5, item30=1
    np.testing.assert_allclose(s, [4.0, 3.5, 1.0])


def test_popularity_requires_fit():
    with pytest.raises(RuntimeError):
        Popularity().score([1])


def test_popularity_items_subset():
    d = small_dataset()
    pop = Popularity().fit(d)
    s = pop.score([1], items=[30, 10])
    assert s.shape == (1, 2)
    assert s[0].tolist() == [1.0, 3.0]  # counts for 30 then 10


def test_random_backend_shape():
    d = small_dataset()
    s = Random(seed=0).fit(d).score([1, 2, 3])
    assert s.shape == (3, 3)


# --------------------------------------------------------------------------- #
# GroupRecommender
# --------------------------------------------------------------------------- #
def test_group_recommender_popularity_avg():
    d = small_dataset()
    rec = GroupRecommender(Popularity(measure="count"), AverageAggregator()).fit(d)
    out = rec.recommend([1, 2], k=3)
    assert list(out) == [10, 20, 30]  # by popularity


def test_group_recommender_exclude_item_ids():
    d = small_dataset()
    rec = GroupRecommender(Popularity(), AverageAggregator()).fit(d)
    out = rec.recommend([1, 2], k=2, exclude=[10])
    assert 10 not in out.tolist()
    assert list(out) == [20, 30]


def test_group_recommender_requires_fit():
    rec = GroupRecommender(Popularity(), AverageAggregator())
    with pytest.raises(RuntimeError):
        rec.recommend([1], k=1)


def test_group_recommender_sequential_state_persists():
    d = make_blobs_dataset(n_users=12, n_items=30, density=1.0, seed=4)
    rec = GroupRecommender(Random(seed=1), LTPAggregator(avoid_repeats=True)).fit(d)
    members = d.users[:4]
    first = set(rec.recommend(members, k=5).tolist())
    second = set(rec.recommend(members, k=5).tolist())
    assert first.isdisjoint(second)  # avoid_repeats carried across sessions


def test_hero_example_runs_end_to_end():
    data = make_blobs_dataset(seed=0)
    groups = gr.groups.synthetic(data, kind="similar", size=4, n=5, metric="pearson", seed=0)
    base = Popularity(measure="mean")
    rec = GroupRecommender(base, AverageAggregator()).fit(data)
    out = rec.recommend(groups[0], k=10)
    assert len(out) == 10 and len(set(out.tolist())) == 10


# --------------------------------------------------------------------------- #
# from_fitted: reuse one fitted base across aggregators
# --------------------------------------------------------------------------- #
def test_from_fitted_matches_fit_without_refitting_the_base():
    data = make_blobs_dataset(n_users=12, n_items=30, density=1.0, seed=5)
    members = data.users[:3]

    class CountingPopularity(Popularity):
        fits = 0

        def fit(self, dataset):
            CountingPopularity.fits += 1
            return super().fit(dataset)

    base = CountingPopularity(measure="mean").fit(data)
    assert CountingPopularity.fits == 1

    reference = GroupRecommender(Popularity(measure="mean"), AverageAggregator()).fit(data)
    reused = GroupRecommender.from_fitted(base, AverageAggregator(), data)

    assert CountingPopularity.fits == 1                        # from_fitted did not refit
    np.testing.assert_array_equal(reused.recommend(members, k=5),
                                  reference.recommend(members, k=5))


def test_from_fitted_carries_normalize_and_requires_dataset():
    data = make_blobs_dataset(n_users=12, n_items=30, density=1.0, seed=6)
    rec = GroupRecommender.from_fitted(Popularity().fit(data), AverageAggregator(),
                                       data, normalize="minmax")
    assert rec.normalize == "minmax" and rec.dataset_ is data
    assert len(rec.recommend(data.users[:3], k=4)) == 4        # usable without .fit()
