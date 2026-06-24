"""Per-member preference normalization (numpy/scipy only).

LTP/RLProp assume member preferences are *commensurable* (KAIS paper, Sect. 3.2);
when they are not, the rating matrix is normalized per member before aggregation.
Each function takes a ``(n_members, n_items)`` matrix and rescales **each row
independently**.

These are dependency-light reimplementations; ``"quantile"``/``"cdf"`` use an
empirical CDF rather than sklearn's ``QuantileTransformer`` (so they will not
bit-match the reference's sklearn path, but share its monotone, uniform-output
behaviour).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _minmax(row: np.ndarray) -> np.ndarray:
    lo, hi = row.min(), row.max()
    if hi <= lo:
        return np.zeros_like(row)
    return (row - lo) / (hi - lo)


def _standard(row: np.ndarray) -> np.ndarray:
    mu, sd = row.mean(), row.std()
    if sd == 0:
        return np.zeros_like(row)
    return (row - mu) / sd


def _robust(row: np.ndarray) -> np.ndarray:
    med = np.median(row)
    q1, q3 = np.percentile(row, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(row)
    return (row - med) / iqr


def _cdf(row: np.ndarray) -> np.ndarray:
    # empirical CDF in (0, 1]; ties share the max rank (matches rankdata default
    # behaviour for monotone rescaling).
    return rankdata(row, method="average") / row.size


_FUNCS = {
    "minmax": _minmax,
    "standard": _standard,
    "robust": _robust,
    "quantile": _cdf,
    "cdf": _cdf,
}


def normalize_mgains(rm: np.ndarray, method: str | None) -> np.ndarray:
    """Return a row-normalized copy of ``rm`` using ``method``.

    ``None`` / ``"none"`` returns ``rm`` unchanged.
    """
    if method is None or method == "none":
        return rm
    try:
        fn = _FUNCS[method]
    except KeyError:
        raise ValueError(
            f"unknown normalize method {method!r}; "
            f"choose from {sorted(set(_FUNCS) | {'none'})}"
        ) from None
    return np.vstack([fn(row.astype(float)) for row in rm])
