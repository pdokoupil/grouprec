"""Tests for the dataset registry, loaders, preprocessing, and HF wrapper.
No network: loaders are exercised on tiny fixtures; auto/auto_nc/manual policy logic
is tested via the registry without downloading."""

from __future__ import annotations

import pandas as pd
import pytest

import grouprec as gr
from grouprec import Dataset
from grouprec.datasets import loaders, preprocess
from grouprec.datasets import cache


# --------------------------------------------------------------------------- #
# registry metadata
# --------------------------------------------------------------------------- #
def test_registry_lists_expected_datasets():
    names = set(gr.datasets.list())
    assert {"ml-100k", "ml-1m", "ml-25m", "ml-32m", "ml-latest-small", "kgrec", "lastfm",
            "camra2011", "mafengwo", "weeplaces", "yelp", "douban"} <= names


def test_movielens_license_varies_by_release():
    # older 100k/1m forbid redistribution; newer 25m/32m/latest-small permit it under same terms
    assert "NO redistribution" in gr.datasets.info("ml-100k").license
    assert "NO redistribution" in gr.datasets.info("ml-1m").license
    assert "redistribution allowed" in gr.datasets.info("ml-25m").license
    spec = gr.datasets.info("ml-latest-small")
    assert "redistribution allowed" in spec.license
    assert spec.checksum and len(spec.checksum) == 64       # pinned snapshot for reproducibility


def test_info_carries_license_and_citation():
    spec = gr.datasets.info("kgrec")
    assert "CC BY-NC" in spec.license
    assert spec.policy == "auto_nc"
    assert spec.homepage.startswith("http")
    assert gr.datasets.info("ml-1m").policy == "auto"
    assert gr.datasets.info("yelp").policy == "auto_nc"      # Yin benchmark, gdrive + citations
    assert gr.datasets.info("mafengwo").policy == "manual"


def test_info_unknown_raises():
    with pytest.raises(KeyError):
        gr.datasets.info("does-not-exist")


# --------------------------------------------------------------------------- #
# policy logic (no downloads)
# --------------------------------------------------------------------------- #
def test_auto_nc_requires_license_acceptance(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPREC_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="non-commercial|accept_license"):
        gr.datasets.load("kgrec")  # no accept_license -> gated before any download


def test_confirm_license_non_interactive_defaults_to_decline():
    from grouprec.datasets import registry
    spec = gr.datasets.info("kgrec")
    # in pytest (no tty / no IPython) we must never block on input(): decline by default
    assert registry._interactive() is False
    assert registry._confirm_license("kgrec", spec, accept_license=False) is False
    assert registry._confirm_license("kgrec", spec, accept_license=True) is True


def test_confirm_license_interactive_prompt(monkeypatch):
    from grouprec.datasets import registry
    spec = gr.datasets.info("kgrec")
    monkeypatch.setattr(registry, "_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert registry._confirm_license("kgrec", spec, accept_license=False) is True
    monkeypatch.setattr("builtins.input", lambda *_: "")        # default = No
    assert registry._confirm_license("kgrec", spec, accept_license=False) is False


def test_accept_all_overrides_gate(monkeypatch):
    from grouprec.datasets import registry
    monkeypatch.setattr(registry, "_ACCEPT_ALL", False)        # isolate from other tests
    spec = gr.datasets.info("kgrec")
    assert registry._confirm_license("kgrec", spec, accept_license=False) is False
    try:
        gr.accept_all(output=None)                            # silent, session-wide
        assert registry._confirm_license("kgrec", spec, accept_license=False) is True
    finally:
        registry._ACCEPT_ALL = False                          # reset global


def test_manual_dataset_gives_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPREC_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="manually|Download|Homepage"):
        gr.datasets.load("mafengwo")


def test_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPREC_CACHE", str(tmp_path))
    assert cache.cache_dir() == tmp_path


# --------------------------------------------------------------------------- #
# loaders (fixtures)
# --------------------------------------------------------------------------- #
def test_movielens_100k_loader(tmp_path):
    (tmp_path / "u.data").write_text("1\t10\t5\t881250949\n1\t20\t3\t881250949\n2\t10\t4\t881250949\n")
    d = loaders.movielens_100k(tmp_path)
    assert d.n_users == 2 and d.n_items == 2 and d.has_ratings and d.has_timestamps
    assert d.name == "ml-100k"


def test_movielens_1m_loader(tmp_path):
    (tmp_path / "ratings.dat").write_text("1::10::5::978300760\n1::20::4::978300760\n2::20::2::978300760\n")
    d = loaders.movielens_1m(tmp_path)
    assert d.n_users == 2 and d.n_items == 2 and d.has_ratings


def test_movielens_csv_loader(tmp_path):
    sub = tmp_path / "ml-25m"
    sub.mkdir()
    (sub / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,10,5.0,1\n2,10,4.0,2\n2,20,1.0,3\n")
    d = loaders.movielens_25m(tmp_path)
    assert d.n_users == 2 and d.n_items == 2 and list(d.items) == [10, 20]


def test_kgrec_and_taste_profile_loaders(tmp_path):
    (tmp_path / "implicit_lf_dataset.csv").write_text("u1,s1,3\nu1,s2,1\nu2,s1,5\n")
    d = loaders.kgrec_music(tmp_path)
    assert d.n_users == 2 and d.n_items == 2 and d.has_ratings

    (tmp_path / "train_triplets.txt").write_text("ua\tsa\t2\nub\tsa\t1\nub\tsb\t4\n")
    t = loaders.taste_profile(tmp_path)
    assert t.n_users == 2 and t.n_items == 2


def test_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        loaders.movielens_100k(tmp_path)


def test_from_path_generic(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("user,item,rating\n1,10,5\n2,20,3\n")
    d = gr.datasets.from_path(p)
    assert d.n_users == 2 and d.has_ratings


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def _df(pairs):
    return Dataset(pd.DataFrame(pairs, columns=["user", "item"]))


def test_k_core_reaches_fixpoint():
    # user 3 has 1 interaction, item 99 has 1 -> removed; iterate to a 2-core
    pairs = [(1, 10), (1, 20), (2, 10), (2, 20), (3, 10), (1, 99)]
    d = preprocess.k_core(_df(pairs), k=2)
    uc = d.interactions["user"].value_counts()
    ic = d.interactions["item"].value_counts()
    assert (uc >= 2).all() and (ic >= 2).all()
    assert 3 not in d.users and 99 not in d.items


def test_filter_min_interactions():
    pairs = [(1, 10), (1, 20), (2, 10), (3, 30)]
    d = preprocess.filter_min_interactions(_df(pairs), min_per_user=2, min_per_item=1)
    assert set(d.users) == {1}


def test_binarize_preprocess():
    d = Dataset(pd.DataFrame([(1, 10, 5.0), (1, 20, 2.0)], columns=["user", "item", "rating"]))
    b = preprocess.binarize(d, threshold=4.0)
    assert not b.has_ratings and len(b) == 1


# --------------------------------------------------------------------------- #
# huggingface wrapper (friendly error when datasets missing)
# --------------------------------------------------------------------------- #
def test_from_huggingface_requires_datasets():
    pytest.importorskip  # keep import structure
    try:
        import datasets  # noqa: F401
        pytest.skip("datasets installed; skipping missing-dependency test")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="datasets|huggingface"):
        gr.datasets.from_huggingface("foo/bar")


def test_from_amazon_reviews_builds_expected_call(monkeypatch):
    """from_amazon_reviews maps category -> raw_review_<cat> config + Amazon columns,
    without hitting the network."""
    from grouprec.datasets import huggingface
    captured = {}

    def fake_from_hf(repo_id, **kw):
        captured["repo_id"] = repo_id
        captured.update(kw)
        return "DATASET"

    monkeypatch.setattr(huggingface, "from_huggingface", fake_from_hf)
    out = huggingface.from_amazon_reviews("Video_Games")
    assert out == "DATASET"
    assert captured["repo_id"] == "McAuley-Lab/Amazon-Reviews-2023"
    assert captured["config"] == "raw_review_Video_Games"
    assert captured["user_col"] == "user_id" and captured["item_col"] == "parent_asin"
    assert captured["trust_remote_code"] is True
