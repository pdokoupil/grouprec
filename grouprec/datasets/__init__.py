"""License-aware dataset registry + loaders + preprocessing.

    import grouprec as gr
    data = gr.datasets.load("ml-1m")                 # auto-fetched, cached
    data = gr.datasets.k_core(data, k=5)             # preprocess
    print(gr.datasets.info("kgrec").license)         # inspect licensing

See :mod:`grouprec.datasets.registry` for the download-policy design.
"""

from __future__ import annotations

from pathlib import Path

from ..data import Dataset
from .consrec import GroupBenchmarkData, load_consrec
from .groupim_format import load_groupim
from .yin import fetch_yin, load_yin
from .huggingface import from_huggingface
from .loaders import generic_interactions
from .preprocess import binarize, filter_min_interactions, k_core
from .registry import DatasetSpec, available, info, load


def from_path(path, **kwargs) -> Dataset:
    """Load a Dataset from a local delimited interactions file (see
    :func:`grouprec.datasets.loaders.generic_interactions`)."""
    return generic_interactions(Path(path), **kwargs)


def list() -> "list[str]":  # noqa: A001 - intentional public name
    """Names of all registered datasets."""
    return available()


__all__ = [
    "load",
    "list",
    "info",
    "available",
    "from_path",
    "from_huggingface",
    "load_consrec",
    "load_groupim",
    "load_yin",
    "fetch_yin",
    "GroupBenchmarkData",
    "k_core",
    "filter_min_interactions",
    "binarize",
    "DatasetSpec",
]
