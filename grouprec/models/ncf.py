"""NCF for groups -- mean-pooled member embeddings + NCF predict layer.

The non-attentive baseline in the AGREE family: the group representation is the mean
of member embeddings, scored against items by the same NCF head. Reference: He et al.,
Neural Collaborative Filtering (WWW'17), as used as a group baseline in AGREE.
"""

from __future__ import annotations

from .base import GroupNNModel


class NCFGroup(GroupNNModel):
    """Mean-pooling neural group recommender."""

    attention = False


__all__ = ["NCFGroup"]
