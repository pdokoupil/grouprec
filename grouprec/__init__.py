"""grouprec -- group recommender systems for Python.

Results aggregation and profile aggregation as first-class citizens behind one
API. See the design doc for the full scope; this is an in-progress build.
"""

from __future__ import annotations

from . import aggregators, backends, datasets, groups, references, split
from .datasets import accept_all
from .bench import BenchmarkResult, BenchmarkTask, benchmark
from .bench.leaderboard import LeaderboardStore
from .data import Dataset, Groups, make_blobs_dataset
from .experiment import Experiment, environment, set_seed
from .profile import ProfileGroupRecommender
from .profiling import track_emissions
from .references import cite, collect_citations
from .eval import (
    Report,
    evaluate,
    evaluate_grouplevel,
    evaluate_sampled,
    evaluate_sequential,
)
from .pipeline import GroupRecommender

__version__ = "0.0.1"


def __getattr__(name: str):
    # Lazy access to the torch-only models subpackage, so `import grouprec` stays
    # dependency-light (torch is only imported when models are actually used).
    if name == "models":
        import importlib
        return importlib.import_module("grouprec.models")
    raise AttributeError(f"module 'grouprec' has no attribute {name!r}")

__all__ = [
    "aggregators",
    "backends",
    "datasets",
    "accept_all",
    "groups",
    "split",
    "Dataset",
    "Groups",
    "make_blobs_dataset",
    "GroupRecommender",
    "ProfileGroupRecommender",
    "evaluate",
    "evaluate_sequential",
    "evaluate_grouplevel",
    "evaluate_sampled",
    "benchmark",
    "BenchmarkTask",
    "BenchmarkResult",
    "LeaderboardStore",
    "Experiment",
    "set_seed",
    "environment",
    "track_emissions",
    "cite",
    "collect_citations",
    "references",
    "Report",
    "__version__",
]
