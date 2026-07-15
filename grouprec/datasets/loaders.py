"""Parsers that turn a downloaded/extracted dataset into a :class:`~grouprec.data.Dataset`.

Each loader takes the dataset's cache directory (where the archive was extracted, or
where the user dropped a manual download) and returns a standardized Dataset with
``user, item, rating?, timestamp?`` columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import Dataset
from .cache import find_file


def _need(path: Path | None, what: str, root: Path) -> Path:
    if path is None:
        raise FileNotFoundError(f"could not find {what} under {root}")
    return path


# --------------------------------------------------------------------------- #
# MovieLens
# --------------------------------------------------------------------------- #
def movielens_100k(root: Path) -> Dataset:
    f = _need(find_file(root, "u.data"), "u.data", root)
    df = pd.read_csv(f, sep="\t", names=["user", "item", "rating", "timestamp"])
    return Dataset(df, name="ml-100k")


def movielens_1m(root: Path) -> Dataset:
    f = _need(find_file(root, "ratings.dat"), "ratings.dat", root)
    df = pd.read_csv(f, sep="::", engine="python",
                     names=["user", "item", "rating", "timestamp"])
    return Dataset(df, name="ml-1m")


def _movielens_csv(root: Path, name: str) -> Dataset:
    f = _need(find_file(root, "ratings.csv"), "ratings.csv", root)
    df = pd.read_csv(f)
    df = df.rename(columns={"userId": "user", "movieId": "item"})
    return Dataset(df[["user", "item", "rating", "timestamp"]], name=name)


def movielens_latest_small(root: Path) -> Dataset:
    return _movielens_csv(root, "ml-latest-small")


def movielens_latest(root: Path) -> Dataset:
    return _movielens_csv(root, "ml-latest")


def movielens_25m(root: Path) -> Dataset:
    return _movielens_csv(root, "ml-25m")


def movielens_32m(root: Path) -> Dataset:
    return _movielens_csv(root, "ml-32m")


# --------------------------------------------------------------------------- #
# KGRec (music) -- implicit listening feedback (CC BY-NC 3.0)
# --------------------------------------------------------------------------- #
def kgrec_music(root: Path) -> Dataset:
    f = find_file(root, "implicit_lf_dataset.csv", "implicit.txt", "user_artist.csv")
    f = _need(f, "KGRec implicit interactions file", root)
    # tab- or comma-separated user, item[, count]
    df = pd.read_csv(f, sep=None, engine="python", header=None)
    df = df.iloc[:, :3]
    df.columns = ["user", "item", "rating"][: df.shape[1]]
    return Dataset(df, name="kgrec")


# --------------------------------------------------------------------------- #
# Last.fm -- Echo Nest Taste Profile (train_triplets.txt: user, song, plays)
# --------------------------------------------------------------------------- #
def taste_profile(root: Path) -> Dataset:
    f = _need(find_file(root, "train_triplets.txt"), "train_triplets.txt", root)
    df = pd.read_csv(f, sep="\t", names=["user", "item", "rating"])
    return Dataset(df, name="lastfm-tasteprofile")


# --------------------------------------------------------------------------- #
# Generic interactions reader (for manual / converted datasets and HF exports)
# --------------------------------------------------------------------------- #
def generic_interactions(
    path: Path,
    *,
    sep: str = ",",
    user_col: str = "user",
    item_col: str = "item",
    rating_col: str | None = "rating",
    timestamp_col: str | None = "timestamp",
    header: int | None = 0,
    name: str | None = None,
) -> Dataset:
    """Read a delimited interactions file into a Dataset by naming its columns."""
    df = pd.read_csv(path, sep=sep, header=header, engine="python")
    return Dataset.from_pandas(
        df, user_col=user_col, item_col=item_col,
        rating_col=rating_col if rating_col in df.columns else None,
        timestamp_col=timestamp_col if timestamp_col in df.columns else None,
        name=name,
    )
