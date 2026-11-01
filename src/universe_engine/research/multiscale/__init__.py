"""Adaptive multiscale fidelity research (Research Program A9).

Provides ScaleCoupler, ResolutionManager, AdaptiveScheduler, and
FixedReferenceSolver for error-controlled adaptive simulation research.
"""

from .adaptive import (
    RegionState, ExchangeRecord, ScaleCoupler, ResolutionManager,
    AdaptiveScheduler, FixedReferenceSolver, benchmark_adaptive_vs_fixed,
)

__all__ = [
    "RegionState", "ExchangeRecord", "ScaleCoupler", "ResolutionManager",
    "AdaptiveScheduler", "FixedReferenceSolver", "benchmark_adaptive_vs_fixed",
]
