"""Deep group-recommendation models (requires the ``torch`` extra).

    pip install grouprec[torch]
    from grouprec.models import AGREE, NCFGroup

These are profile-aggregation models (``paradigm="profile"``): they consume
group-item interactions and are evaluated **coupled / group-level** (see
``grouprec.eval.evaluate_grouplevel`` and ``gr.benchmark`` with ``level="group"``),
where they share a leaderboard with the results-aggregators.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised when torch missing
    raise ImportError(
        "grouprec.models requires the optional 'torch' dependency. "
        "Install it with: pip install grouprec[torch]"
    ) from exc

from .agree import AGREE
from .aligngroup import AlignGroup
from .base import GroupNNModel
from .consrec import ConsRec
from .data import GroupTrainData, make_synthetic_group_data
from .groupim import GroupIM
from .hypergraph import HyperGroup
from .ncf import NCFGroup

__all__ = ["AGREE", "NCFGroup", "GroupIM", "ConsRec", "HyperGroup", "AlignGroup",
           "GroupNNModel", "GroupTrainData", "make_synthetic_group_data"]
