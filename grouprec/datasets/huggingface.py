"""Generic Hugging Face Hub loader.

Lets anyone point grouprec at any interactions dataset on the Hub (or load one they
publish) without us hard-coding it. Requires the optional ``datasets`` package.
"""

from __future__ import annotations

from ..data import Dataset


def from_huggingface(
    repo_id: str,
    *,
    split: str = "train",
    user_col: str = "user",
    item_col: str = "item",
    rating_col: str | None = "rating",
    timestamp_col: str | None = "timestamp",
    name: str | None = None,
    **load_kwargs,
) -> Dataset:
    """Load a Hugging Face dataset and map its columns to the grouprec schema.

    Example
    -------
    >>> from grouprec.datasets import from_huggingface
    >>> data = from_huggingface("some/movie-ratings", user_col="userId", item_col="movieId")
    """
    try:
        import datasets as hfds
    except ImportError as exc:
        raise ImportError(
            "from_huggingface requires the optional 'datasets' package. "
            "Install it with: pip install grouprec[huggingface]"
        ) from exc

    ds = hfds.load_dataset(repo_id, split=split, **load_kwargs)
    df = ds.to_pandas()
    return Dataset.from_pandas(
        df, user_col=user_col, item_col=item_col,
        rating_col=rating_col if rating_col and rating_col in df.columns else None,
        timestamp_col=timestamp_col if timestamp_col and timestamp_col in df.columns else None,
        name=name or repo_id,
    )


__all__ = ["from_huggingface"]
