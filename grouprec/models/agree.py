"""AGREE -- Attentive Group Recommendation (Cao et al., SIGIR'18).

Reimplemented to the common API: per-user/item embeddings, an item-conditioned
attention over member embeddings forming the group representation, and an NCF predict
layer on ``[g⊙i, g, i]``, trained jointly on user-item and group-item interactions.

Original code: https://github.com/LianHaiMiao/Attentive-Group-Recommendation
(license/attribution retained per the original repo).
"""

from __future__ import annotations

from .base import GroupNNModel


class AGREE(GroupNNModel):
    """Attentive group recommender (item-conditioned attention over members)."""

    attention = True


__all__ = ["AGREE"]
