"""Kardashev scale calculations."""
from __future__ import annotations
import numpy as np


def kardashev_index(energy_watts: float) -> float:
    """Continuous Kardashev index K = (log10(P) - 6) / 10."""
    if energy_watts <= 0:
        return -float("inf")
    p = max(energy_watts, 1e-18)
    return (np.log10(p) - 6.0) / 10.0


def kardashev_type(k: float) -> str:
    if k < 0.0:
        return "sub-Type-I"
    elif k < 1.0:
        return "Type-I"
    elif k < 2.0:
        return "Type-II"
    elif k < 3.0:
        return "Type-III"
    else:
        return f"Type-{int(k)}+"
