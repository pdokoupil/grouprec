"""Carbon / energy profiling -- PoC of the custom-metric story.

``track_emissions()`` is a context manager that estimates kg CO2e for the wrapped work.
If ``codecarbon`` is installed it uses it; otherwise it falls back to a transparent
estimate ``duration_h * power_kW * carbon_intensity`` (default grid intensity 0.4
kgCO2e/kWh). Use it to report the carbon cost of training/benchmarking alongside
accuracy/fairness metrics -- a custom, run-level metric.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Emissions:
    seconds: float = 0.0
    kwh: float = 0.0
    kg_co2e: float = 0.0
    backend: str = "estimate"


@contextmanager
def track_emissions(*, power_kw: float = 0.05, carbon_intensity: float = 0.4):
    """Context manager yielding an :class:`Emissions` populated on exit.

    power_kw : assumed average draw (default 50 W, a laptop CPU) for the fallback.
    carbon_intensity : kgCO2e per kWh of the local grid (default 0.4).
    """
    result = Emissions()
    try:
        from codecarbon import EmissionsTracker  # type: ignore
        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
        result.backend = "codecarbon"
        start = time.perf_counter()
        try:
            yield result
        finally:
            result.kg_co2e = float(tracker.stop() or 0.0)
            result.seconds = time.perf_counter() - start
    except ImportError:
        start = time.perf_counter()
        try:
            yield result
        finally:
            result.seconds = time.perf_counter() - start
            result.kwh = (result.seconds / 3600.0) * power_kw
            result.kg_co2e = result.kwh * carbon_intensity


__all__ = ["track_emissions", "Emissions"]
