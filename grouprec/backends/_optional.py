"""Helper for friendly optional-dependency errors (never a raw ModuleNotFoundError)."""

from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(module: str, extra: str, purpose: str) -> ModuleType:
    """Import ``module`` or raise a clear install hint pointing at the extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via adapters' tests
        raise ImportError(
            f"{purpose} requires the optional '{extra}' dependency. "
            f"Install it with: pip install grouprec[{extra}]"
        ) from exc
