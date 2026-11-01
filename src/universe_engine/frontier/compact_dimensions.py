"""Compact extra-dimension research models (Requirement 8 / Test H).

Supports mathematically explicit examples:
  - Circle compactification: S^1 of radius R; KK modes m_n = n/R
  - Flat toroidal compactification: T^d with radii R_i; KK spectrum m_vec = (n_i/R_i)
  - Selected curved compact geometries (e.g. 2-sphere)
  - Higher-dimensional mode decomposition (Fourier modes on compact dimensions)
  - Kaluza-Klein mass spectra in tractable cases

Includes an explorer showing how a dimension may be compact at small scales
rather than visibly extended across ordinary space.

Scientific status: mathematical models, numerically verified against analytic
results. Does NOT claim a generic Calabi-Yau mesh is a solved physical
compactification, and does NOT promise full 10D/11D string/M-theory dynamics.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..geometry.geometry_core import (
    CoordinateChart, MetricSignature, MetricTensor, GeometrySpec, Field,
    circle_compactification,
)


@dataclass
class CompactDimension:
    """Description of one compact dimension."""
    name: str
    type: str              # 'circle' | 'torus' | 'sphere' | 'interval'
    radius: float          # characteristic radius (in model units)
    periodic: bool         # whether the coordinate is periodic
    range: Tuple[float, float]  # coordinate range

    def kk_mass(self, mode: int) -> float:
        """Kaluza-Klein mass for mode number n: m_n = n / R (circle)."""
        if self.type == "circle":
            return abs(mode) / self.radius
        return 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompactificationSpec:
    """Specification of a compact extra-dimensional geometry.

    compact_dims: list of CompactDimension objects
    base_D: number of non-compact (large) dimensions
    total_D: base_D + len(compact_dims)
    """
    base_D: int
    compact_dims: List[CompactDimension]
    total_D: int = field(init=False)

    def __post_init__(self):
        self.total_D = self.base_D + len(self.compact_dims)

    def kk_spectrum(self, max_mode: int = 10) -> Dict[str, List[float]]:
        """Kaluza-Klein mass spectrum for all compact dimensions.

        Returns {dim_name: [m_0, m_1, ..., m_max_mode]}.
        For a torus, returns combined spectrum.
        """
        spectrum: Dict[str, List[float]] = {}
        for cd in self.compact_dims:
            spectrum[cd.name] = [cd.kk_mass(n) for n in range(max_mode + 1)]
        return spectrum

    def total_kk_mass(self, mode_numbers: Dict[str, int]) -> float:
        """Total KK mass for a combination of mode numbers (toroidal)."""
        m2 = 0.0
        for cd in self.compact_dims:
            n = mode_numbers.get(cd.name, 0)
            m2 += cd.kk_mass(n) ** 2
        return math.sqrt(m2)

    def to_dict(self) -> Dict:
        return {
            "base_D": self.base_D,
            "total_D": self.total_D,
            "compact_dims": [cd.to_dict() for cd in self.compact_dims],
            "kk_spectrum": self.kk_spectrum(),
        }


class CircleCompactification:
    """Circle compactification S^1 of radius R.

    Metric: ds^2 = g_mu_nu(x) dx^mu dx^nu + R^2 dtheta^2, theta in [0, 2pi)
    
    A scalar field Phi(x, theta) decomposes into KK modes:
      Phi(x, theta) = sum_n phi_n(x) e^{i n theta / R}
    with 4D masses m_n^2 = m_0^2 + (n/R)^2.
    
    The dimension is compact at scale R, not visibly extended.
    """
    def __init__(self, radius: float):
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError("Circle radius must be positive")

    def kk_mass(self, n: int, m0: float = 0.0) -> float:
        """KK mass for mode n: m_n^2 = m_0^2 + (n/R)^2."""
        return math.sqrt(m0**2 + (n / self.radius) ** 2)

    def kk_spectrum(self, max_mode: int, m0: float = 0.0) -> List[float]:
        return [self.kk_mass(n, m0) for n in range(max_mode + 1)]

    def mode_function(self, n: int, theta: float) -> complex:
        """KK mode function: e^{i n theta / R} normalized."""
        return np.exp(1j * n * theta / self.radius)

    def geometry_spec(self) -> GeometrySpec:
        """Return the GeometrySpec for this compactification."""
        return circle_compactification(self.radius)

    def to_dict(self) -> Dict:
        return {
            "type": "circle",
            "radius": self.radius,
            "kk_spectrum_first_10": self.kk_spectrum(10),
        }


class ToroidalCompactification:
    """Flat toroidal compactification T^d with radii R_i.

    Metric: ds^2 = g_mu_nu(x) dx^mu dx^nu + sum_i R_i^2 dtheta_i^2
    
    KK spectrum: m_vec^2 = m_0^2 + sum_i (n_i / R_i)^2
    """
    def __init__(self, radii: List[float]):
        self.radii = [float(r) for r in radii]
        if any(r <= 0 for r in self.radii):
            raise ValueError("All torus radii must be positive")
        self.d = len(self.radii)

    def kk_mass(self, mode_numbers: List[int], m0: float = 0.0) -> float:
        """KK mass for mode vector n: m^2 = m_0^2 + sum_i (n_i/R_i)^2."""
        m2 = m0**2
        for n, R in zip(mode_numbers, self.radii):
            m2 += (n / R) ** 2
        return math.sqrt(m2)

    def kk_spectrum(self, max_mode_per_dim: int, m0: float = 0.0) -> List[float]:
        """Enumerate KK masses for all mode combinations up to max_mode_per_dim."""
        import itertools
        ranges = [range(-max_mode_per_dim, max_mode_per_dim + 1) for _ in range(self.d)]
        masses = []
        for modes in itertools.product(*ranges):
            masses.append(self.kk_mass(list(modes), m0))
        return sorted(masses)

    def compactification_spec(self) -> CompactificationSpec:
        dims = [
            CompactDimension(
                name=f"theta_{i}", type="circle",
                radius=R, periodic=True, range=(0.0, 2.0 * math.pi),
            )
            for i, R in enumerate(self.radii)
        ]
        return CompactificationSpec(base_D=4, compact_dims=dims)

    def to_dict(self) -> Dict:
        return {
            "type": "torus",
            "dimensions": self.d,
            "radii": self.radii,
            "kk_mass_example": self.kk_mass([1] * self.d),
        }


class SphereCompactification:
    """Curved compact geometry: 2-sphere of radius R.

    Metric on S^2: ds^2 = R^2 (dtheta^2 + sin^2(theta) dphi^2)
    
    Scalar Laplacian eigenvalues: l(l+1)/R^2, with degeneracy 2l+1.
    KK masses: m_l^2 = m_0^2 + l(l+1)/R^2
    
    This is a curved compact space (not flat), showing that compact dimensions
    need not be tori.
    """
    def __init__(self, radius: float):
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError("Sphere radius must be positive")

    def laplacian_eigenvalue(self, l: int) -> float:
        """Laplacian eigenvalue on S^2: lambda_l = l(l+1)/R^2."""
        return l * (l + 1) / self.radius**2

    def kk_mass(self, l: int, m0: float = 0.0) -> float:
        """KK mass for angular momentum l: m_l^2 = m_0^2 + l(l+1)/R^2."""
        return math.sqrt(m0**2 + self.laplacian_eigenvalue(l))

    def degeneracy(self, l: int) -> int:
        """Degeneracy of level l: 2l + 1."""
        return 2 * l + 1

    def kk_spectrum(self, max_l: int, m0: float = 0.0) -> List[Tuple[float, int]]:
        """Return [(mass, degeneracy), ...] for l = 0..max_l."""
        return [(self.kk_mass(l, m0), self.degeneracy(l)) for l in range(max_l + 1)]

    def to_dict(self) -> Dict:
        return {
            "type": "sphere_S2",
            "radius": self.radius,
            "spectrum_first_5": self.kk_spectrum(5),
        }


class CompactDimensionExplorer:
    """Explorer showing how a dimension may be compact at small scales.

    Demonstrates that an extra dimension of radius R is physically distinct
    from a large extended dimension: at distances much larger than R, the
    dimension is invisible (KK modes are too heavy to excite).
    """
    def __init__(self, radius: float):
        self.circle = CircleCompactification(radius)

    def effective_4d_mass(self, mode_n: int, m0: float, energy_scale_GeV: float) -> Dict:
        """Show whether KK mode n is accessible at a given energy scale."""
        m_kk = self.circle.kk_mass(mode_n, m0)
        accessible = energy_scale_GeV >= m_kk
        return {
            "mode": mode_n,
            "kk_mass": m_kk,
            "energy_scale_GeV": energy_scale_GeV,
            "accessible": accessible,
            "radius": self.circle.radius,
            "message": (
                f"KK mode {mode_n} has mass {m_kk:.3e}; at energy {energy_scale_GeV:.3e} "
                f"{'it is' if accessible else 'it is NOT'} kinematically accessible."
            ),
        }

    def visibility_vs_scale(self, scales_GeV: List[float], m0: float = 0.0) -> List[Dict]:
        """For each energy scale, list accessible KK modes."""
        results = []
        for E in scales_GeV:
            accessible_modes = []
            for n in range(0, 20):
                m = self.circle.kk_mass(n, m0)
                if E > m:
                    accessible_modes.append(n)
            results.append({
                "energy_GeV": E,
                "accessible_modes": accessible_modes,
                "n_accessible": len(accessible_modes),
                "compact_radius": self.circle.radius,
            })
        return results

    def to_dict(self) -> Dict:
        return {
            "type": "circle_explorer",
            "radius": self.circle.radius,
            "kk_mass_n1": self.circle.kk_mass(1),
        }


def verify_kk_circle_spectrum(radius: float, max_mode: int = 10) -> Dict:
    """Verify KK spectrum for circle compactification against analytic m_n = n/R."""
    circle = CircleCompactification(radius)
    spectrum = circle.kk_spectrum(max_mode)
    errors = []
    for n, m in enumerate(spectrum):
        expected = n / radius
        errors.append(abs(m - expected))
    return {
        "radius": radius,
        "spectrum": spectrum,
        "max_error": max(errors) if errors else 0.0,
        "all_exact": all(e < 1e-12 for e in errors),
    }


def verify_kk_torus_spectrum(radii: List[float], modes: List[List[int]]) -> Dict:
    """Verify KK spectrum for toroidal compactification."""
    torus = ToroidalCompactification(radii)
    results = []
    for m_vec in modes:
        m = torus.kk_mass(m_vec)
        expected = math.sqrt(sum((n / R) ** 2 for n, R in zip(m_vec, radii)))
        results.append({
            "modes": m_vec,
            "computed": m,
            "expected": expected,
            "error": abs(m - expected),
        })
    return {
        "radii": radii,
        "results": results,
        "all_exact": all(r["error"] < 1e-12 for r in results),
    }


def verify_sphere_spectrum(radius: float, max_l: int = 5) -> Dict:
    """Verify sphere Laplacian eigenvalues lambda_l = l(l+1)/R^2."""
    sphere = SphereCompactification(radius)
    spectrum = sphere.kk_spectrum(max_l)
    errors = []
    for l, (m, deg) in enumerate(spectrum):
        expected_lambda = l * (l + 1) / radius**2
        expected_m = math.sqrt(expected_lambda)
        errors.append(abs(m - expected_m))
        assert deg == 2 * l + 1, f"degeneracy mismatch at l={l}"
    return {
        "radius": radius,
        "spectrum": spectrum,
        "max_error": max(errors) if errors else 0.0,
        "all_exact": all(e < 1e-12 for e in errors),
    }
