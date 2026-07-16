"""Aggregator registry and exports."""

from __future__ import annotations

from .base import Aggregator, SequentialAggregator, Paradigm
from .social_choice import (
    AdditiveAggregator,
    AverageAggregator,
    WeightedAverageAggregator,
    LeastMiseryAggregator,
    MultiplicativeAggregator,
    MostPleasureAggregator,
    AVGNoMiseryAggregator,
    BordaCountAggregator,
    FAIAggregator,
)
from .fairness import (
    EPFuzzDAAggregator,
    GFARAggregator,
    GreedyScalarizationAggregator,
    GreedyLMAggregator,
    PARAggregator,
    SPGreedyAggregator,
)
from .sequential import (
    RLPropAggregator,
    LTPAggregator,
    PeriodicFAIAggregator,
    EPFuzzDAWeightedAggregator,
    SDAAAggregator,
    SIAAAggregator,
)
from ._normalize import NORMALIZE_METHODS, normalize_mgains

# Name -> class registry, keyed by the short paper name. Used by gr.benchmark and
# the gr.aggregators.get(...) factory.
_REGISTRY: dict[str, type[Aggregator]] = {
    "ADD": AdditiveAggregator,
    "AVG": AverageAggregator,
    "wAVG": WeightedAverageAggregator,
    "LMS": LeastMiseryAggregator,
    "MUL": MultiplicativeAggregator,
    "MPL": MostPleasureAggregator,
    "AVGNM": AVGNoMiseryAggregator,
    "BDC": BordaCountAggregator,
    "FAI": FAIAggregator,
    "EPFuzzDA": EPFuzzDAAggregator,
    "GFAR": GFARAggregator,
    "GreedyLM": GreedyLMAggregator,
    "PAR": PARAggregator,
    "SPGreedy": SPGreedyAggregator,
    "RLProp": RLPropAggregator,
    "LTP": LTPAggregator,
    "PeriodicFAI": PeriodicFAIAggregator,
    "EPFuzzDAWeighted": EPFuzzDAWeightedAggregator,
    "SDAA": SDAAAggregator,
    "SIAA": SIAAAggregator,
}


def get(name: str, **kwargs) -> Aggregator:
    """Instantiate an aggregator by its short name (e.g. ``"GFAR"``)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown aggregator {name!r}; available: {sorted(_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def available() -> list[str]:
    """Sorted list of registered aggregator names."""
    return sorted(_REGISTRY)


__all__ = [
    "Aggregator",
    "SequentialAggregator",
    "Paradigm",
    "AdditiveAggregator",
    "AverageAggregator",
    "WeightedAverageAggregator",
    "LeastMiseryAggregator",
    "MultiplicativeAggregator",
    "MostPleasureAggregator",
    "AVGNoMiseryAggregator",
    "BordaCountAggregator",
    "FAIAggregator",
    "EPFuzzDAAggregator",
    "GFARAggregator",
    "GreedyScalarizationAggregator",
    "GreedyLMAggregator",
    "PARAggregator",
    "SPGreedyAggregator",
    "RLPropAggregator",
    "LTPAggregator",
    "PeriodicFAIAggregator",
    "EPFuzzDAWeightedAggregator",
    "SDAAAggregator",
    "SIAAAggregator",
    "normalize_mgains",
    "NORMALIZE_METHODS",
    "get",
    "available",
]
