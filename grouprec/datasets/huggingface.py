"""Generic Hugging Face Hub loader.

Lets anyone point grouprec at any interactions dataset on the Hub (or load one they
publish) without us hard-coding it. Requires the optional ``datasets`` package
(``pip install grouprec[huggingface]``). Because the Hub hosts essentially every
public recommendation benchmark -- MovieLens mirrors, the **Amazon Reviews** family,
Yelp, Steam, Last.fm, Book-Crossing, ... -- this one function unlocks them all; the
caller just maps the column names.
"""

from __future__ import annotations

from ..data import Dataset


def from_huggingface(
    repo_id: str,
    *,
    config: str | None = None,
    split: str = "train",
    user_col: str = "user",
    item_col: str = "item",
    rating_col: str | None = "rating",
    timestamp_col: str | None = "timestamp",
    name: str | None = None,
    **load_kwargs,
) -> Dataset:
    """Load a Hugging Face dataset and map its columns to the grouprec schema.

    Parameters
    ----------
    repo_id : Hub dataset id, e.g. ``"McAuley-Lab/Amazon-Reviews-2023"``.
    config : Hub *configuration* name (the second positional arg of
        ``datasets.load_dataset``), e.g. ``"raw_review_All_Beauty"``. Many large
        collections (Amazon Reviews, GLUE-style hubs) require this.
    split : which split to read (``"train"``, ``"full"``, ...).
    user_col, item_col, rating_col, timestamp_col : source column names to map onto
        grouprec's ``user`` / ``item`` / ``rating`` / ``timestamp`` schema. Missing
        rating/timestamp columns are dropped silently.
    name : grouprec dataset name (defaults to ``repo_id``).
    **load_kwargs : forwarded to ``datasets.load_dataset`` (e.g.
        ``trust_remote_code=True``, ``streaming=...``).

    Example
    -------
    >>> from grouprec.datasets import from_huggingface
    >>> data = from_huggingface(
    ...     "McAuley-Lab/Amazon-Reviews-2023",
    ...     config="raw_review_All_Beauty", split="full",
    ...     user_col="user_id", item_col="parent_asin",
    ...     rating_col="rating", timestamp_col="timestamp",
    ...     trust_remote_code=True,
    ... )
    """
    try:
        import datasets as hfds
    except ImportError as exc:
        raise ImportError(
            "from_huggingface requires the optional 'datasets' package. "
            "Install it with: pip install grouprec[huggingface]"
        ) from exc

    args = (repo_id,) if config is None else (repo_id, config)
    ds = hfds.load_dataset(*args, split=split, **load_kwargs)
    df = ds.to_pandas()
    return Dataset.from_pandas(
        df, user_col=user_col, item_col=item_col,
        rating_col=rating_col if rating_col and rating_col in df.columns else None,
        timestamp_col=timestamp_col if timestamp_col and timestamp_col in df.columns else None,
        name=name or (f"{repo_id}/{config}" if config else repo_id),
    )


def from_amazon_reviews(
    category: str = "All_Beauty",
    *,
    split: str = "full",
    repo_id: str = "McAuley-Lab/Amazon-Reviews-2023",
    **kwargs,
) -> Dataset:
    """Convenience wrapper for the **Amazon Reviews 2023** collection (McAuley Lab).

    ``category`` is an Amazon product category, e.g. ``"All_Beauty"``,
    ``"Books"``, ``"Electronics"``, ``"Video_Games"`` -- loaded as the
    ``raw_review_<category>`` config. See the dataset card for the full list and its
    (research-use) license; grouprec never re-hosts the bytes.

    >>> data = from_amazon_reviews("Video_Games")          # doctest: +SKIP
    """
    return from_huggingface(
        repo_id,
        config=f"raw_review_{category}",
        split=split,
        user_col="user_id",
        item_col="parent_asin",
        rating_col="rating",
        timestamp_col="timestamp",
        name=f"amazon-reviews-2023/{category}",
        trust_remote_code=kwargs.pop("trust_remote_code", True),
        **kwargs,
    )


__all__ = ["from_huggingface", "from_amazon_reviews"]
