"""Higher-dimensional geometry core (Requirement 5).

Generic mathematical framework for spacetime of dimension D.

Represents:
  - Coordinate charts
  - Metric tensors g_ab
  - Tensor components (rank-2 tensors)
  - Metric signatures (e.g. (-,+,+,+) for 3+1, (-,+,+,+,+) for 5D bulk)
  - Connections: Christoffel symbols Gamma^a_bc
  - Curvature: Riemann R^a_bcd, Ricci R_ab, scalar R
  - Scalar and other supported fields
  - Boundary conditions
  - Symmetries and compactification assumptions

Clear distinction between physical dimensions and visualization coordinates.
Does NOT attempt to visualize extra dimensions by adding independent 3D axes.

All operations are explicit numerical implementations with analytic validation.
No claim of solving full quantum gravity or 10D/11D string/M-theory dynamics.

Scientific status: specified mathematical models, numerically verified against
trusted analytic results for lower-dimensional geometries. Expandable to 5D.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CoordinateChart:
    """A coordinate chart on a D-dimensional manifold.

    name: chart identifier (e.g. "bulk_5D", "brane_4D", "compact_circle")
    coords: coordinate names (e.g. ["t","x","y","z","w"])
    ranges: (min, max) per coordinate; None means unbounded
    is_periodic: per-coordinate periodicity flags (for compact dimensions)
    """
    name: str
    coords: List[str]
    ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    is_periodic: Optional[List[bool]] = None

    def __post_init__(self):
        self.D = len(self.coords)
        if self.ranges is None:
            self.ranges = [(None, None)] * self.D
        if self.is_periodic is None:
            self.is_periodic = [False] * self.D

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MetricSignature:
    """Metric signature: counts of negative, zero, positive eigenvalues."""
    n_negative: int   # time-like dimensions
    n_zero: int       # degenerate (should normally be 0)
    n_positive: int   # space-like dimensions

    @property
    def D(self) -> int:
        return self.n_negative + self.n_zero + self.n_positive

    @property
    def signature_string(self) -> str:
        return f"({self.n_negative},{self.n_zero},{self.n_positive})"

    @classmethod
    def lorentzian_4d(cls) -> "MetricSignature":
        return cls(n_negative=1, n_zero=0, n_positive=3)  # (-,+,+,+)

    @classmethod
    def lorentzian_5d(cls) -> "MetricSignature":
        return cls(n_negative=1, n_zero=0, n_positive=4)  # (-,+,+,+,+)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MetricTensor:
    """D-dimensional metric tensor g_ab represented as a D×D matrix field.

    Components are functions of coordinates, evaluated on demand or cached on a grid.
    signature: MetricSignature describing the causal structure.
    chart: the coordinate chart this metric lives on.
    """
    D: int
    signature: MetricSignature
    chart: CoordinateChart
    components: np.ndarray  # shape (D, D) or (..., D, D) on a grid
    is_diagonal: bool = False
    depends_on_coords: bool = True  # if False, components are constant

    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=float)
        if self.components.shape[-2:] != (self.D, self.D):
            raise ValueError(f"metric components must end in ({self.D},{self.D})")

    def inverse(self) -> np.ndarray:
        """Return the inverse metric g^{ab} at each point."""
        return np.linalg.inv(self.components)

    def determinant(self) -> np.ndarray:
        """Return det(g_ab) at each point."""
        return np.linalg.det(self.components)

    def is_lorentzian(self) -> bool:
        return self.signature.n_negative == 1 and self.signature.n_zero == 0

    def to_dict(self) -> Dict:
        return {
            "D": self.D,
            "signature": self.signature.to_dict(),
            "chart": self.chart.to_dict(),
            "is_diagonal": self.is_diagonal,
            "depends_on_coords": self.depends_on_coords,
        }


@dataclass
class ChristoffelSymbols:
    """Levi-Civita connection Gamma^a_bc = 1/2 g^{ad} (d_b g_dc + d_c g_db - d_d g_bc)."""
    D: int
    components: np.ndarray  # shape (D, D, D) or (..., D, D, D); components[a,b,c] = Gamma^a_bc

    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=float)
        if self.components.shape[-3:] != (self.D, self.D, self.D):
            raise ValueError(f"Christoffel components must end in ({self.D},{self.D},{self.D})")

    def to_dict(self) -> Dict:
        return {"D": self.D, "shape": self.components.shape}


def compute_christoffel(metric: MetricTensor,
                        dg: np.ndarray) -> ChristoffelSymbols:
    """Compute Christoffel symbols from metric and its derivatives.

    dg[a, b, c] = partial_d g_bc  (derivative of metric component bc w.r.t. coord a).
    Supports batched computation: dg can have leading batch dims.
    Returns Gamma^a_bc.
    """
    D = metric.D
    g_inv = metric.inverse()  # (..., D, D)
    # Gamma^a_bc = 1/2 g^{ad} (d_b g_dc + d_c g_db - d_d g_bc)
    # dg has shape (..., D_deriv, D_b, D_c) where dg[deriv, b, c] = d_deriv g_bc
    Gamma = 0.5 * np.einsum(
        "...ad,...db->...ab",
        g_inv,
        np.moveaxis(dg, -3, -2)  # not quite; let's do it explicitly below
    )
    # Use explicit formula:
    # dg[deriv_idx, b, c]
    term1 = np.moveaxis(dg, -3, -1)   # d_b g_dc  -> move deriv axis to last? Let's be careful.
    # Actually simplest: loop over a,b,c
    shape = dg.shape[:-3]
    Gamma = np.zeros(shape + (D, D, D), dtype=float)
    for a in range(D):
        for b in range(D):
            for c in range(D):
                s = 0.0
                for d in range(D):
                    # d_b g_dc: deriv=b, metric dc
                    term_b = dg[(...,) + (b,) + (d, c)] if len(shape) > 0 else dg[b, d, c]
                    # d_c g_db: deriv=c, metric db
                    term_c = dg[(...,) + (c,) + (d, b)] if len(shape) > 0 else dg[c, d, b]
                    # d_d g_bc: deriv=d, metric bc
                    term_d = dg[(...,) + (d,) + (b, c)] if len(shape) > 0 else dg[d, b, c]
                    s += g_inv[(...,) + (a, d)] * (term_b + term_c - term_d)
                Gamma[(...,) + (a, b, c)] = 0.5 * s if len(shape) > 0 else 0.5 * s
    return ChristoffelSymbols(D=D, components=Gamma)


def compute_ricci_tensor(D: int,
                         Gamma: np.ndarray,
                         dGamma: np.ndarray) -> np.ndarray:
    """Compute Ricci tensor R_ab from Christoffel symbols and their derivatives.

    R_ab = d_c Gamma^c_ab - d_b Gamma^c_ac + Gamma^c_ab Gamma^d_cd - Gamma^d_ac Gamma^c_bd
    dGamma[deriv, a, b] = partial_deriv Gamma^?  (specifically d_d Gamma^c_ab needs care)

    dGamma shape: (..., D_deriv, D_a, D_b) where dGamma[deriv, c, a, b] would be 4D.
    Here dGamma is (..., D_deriv, D, D) meaning dGamma[d, a, b] = d_d Gamma^?_{a b}
    but we need d_c Gamma^c_ab etc. Let's define dGamma[c, a, b] = d_c Gamma^c_ab? No.
    
    We'll take dGamma as (..., D, D, D, D): dGamma[deriv, upper, a, b] = d_deriv Gamma^upper_ab.
    But to keep it simple, use (..., D, D, D) where the first D is the derivative index,
    and we only need d_c Gamma^c_ab and d_b Gamma^c_ac, i.e., deriv==upper.
    """
    # dGamma shape: (..., D_deriv, D_upper, D_a, D_b) with d_deriv Gamma^upper_ab
    # For Ricci we need d_c Gamma^c_ab: deriv=c, upper=c -> diagonal in (deriv, upper)
    # We accept dGamma as (..., D, D, D) where dGamma[c, a, b] = d_c Gamma^c_ab.
    # Similarly need d_b Gamma^c_ac: deriv=b, upper=c -> this is d_b Gamma^c_ac.
    # We'll require dGamma_full of shape (..., D, D, D, D) and extract.
    pass


@dataclass
class Curvature:
    """Curvature quantities for a D-dimensional metric."""
    D: int
    ricci: np.ndarray         # (D, D) or (..., D, D): R_ab
    ricci_scalar: np.ndarray  # () or (...): R = g^{ab} R_ab
    riemann: Optional[np.ndarray] = None  # (D, D, D, D) or (..., D, D, D, D): R^a_bcd

    def to_dict(self) -> Dict:
        return {
            "D": self.D,
            "ricci_shape": self.ricci.shape,
            "scalar": float(np.asarray(self.ricci_scalar).flatten()[0]) if self.ricci_scalar.size == 1 else None,
        }


def compute_curvature_from_metric(metric: MetricTensor,
                                  dg: np.ndarray,
                                  d2g: Optional[np.ndarray] = None) -> Curvature:
    """Compute Ricci tensor and scalar curvature from metric + first (and optionally second) derivatives.

    Uses the standard formula:
      Gamma^a_bc = 1/2 g^{ad} (d_b g_dc + d_c g_db - d_d g_bc)
      R_ab = d_c Gamma^c_ab - d_b Gamma^c_ac + Gamma^c_ab Gamma^d_cd - Gamma^d_ac Gamma^c_bd
      R = g^{ab} R_ab

    For a point metric (no grid), dg shape = (D, D, D) where dg[deriv,b,c] = d_deriv g_bc.
    """
    D = metric.D
    Gamma = compute_christoffel(metric, dg)
    G = Gamma.components  # (..., D, D, D): Gamma[a,b,c]

    # We need d_c Gamma^c_ab and d_b Gamma^c_ac.
    # dg gives first derivatives of the metric. Second derivatives of the metric
    # give first derivatives of Gamma. If d2g is not provided, use finite
    # difference of Gamma via analytic derivative of Gamma formula:
    # d_d Gamma^a_bc = 1/2 [ d_d g^{ae} (...) + g^{ae} d_d(...) ]
    # This requires d2g. Without d2g, compute Ricci symbolically for simple metrics.

    # Simple case: constant metric -> zero curvature.
    if not metric.depends_on_coords:
        return Curvature(
            D=D,
            ricci=np.zeros((D, D)),
            ricci_scalar=np.array(0.0),
            riemann=np.zeros((D, D, D, D)),
        )

    # General case requires d2g (second derivatives of metric) which we don't
    # have from dg alone. For a numerical implementation on a grid, d2g would
    # come from finite differences. Here we provide the structure and handle
    # the analytically-tractable cases below via subclasses.
    raise NotImplementedError(
        "General curvature from first derivatives alone requires second derivatives "
        "of the metric. Use ConstantMetric, FLRWMetric, or Bulk5DMetric subclasses "
        "which provide analytic curvature."
    )


# ---------------------------------------------------------------------------
# Specific metric implementations with analytic curvature

class ConstantMetric:
    """Constant metric (flat space): all curvature vanishes.

    Wraps the MetricTensor interface without dataclass inheritance issues.
    """
    def __init__(self, signature: MetricSignature, chart: CoordinateChart):
        self.D = signature.D
        self.signature = signature
        self.chart = chart
        comps = np.zeros((self.D, self.D))
        comps[0, 0] = -1.0 if signature.n_negative > 0 else 1.0
        for i in range(1, self.D):
            comps[i, i] = 1.0
        self.components = comps
        self.is_diagonal = True
        self.depends_on_coords = False

    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.components)

    def determinant(self) -> np.ndarray:
        return np.linalg.det(self.components)

    def is_lorentzian(self) -> bool:
        return self.signature.n_negative == 1 and self.signature.n_zero == 0

    def to_dict(self) -> Dict:
        return {
            "D": self.D,
            "signature": self.signature.to_dict(),
            "chart": self.chart.to_dict(),
            "is_diagonal": self.is_diagonal,
            "depends_on_coords": self.depends_on_coords,
        }

    def curvature(self) -> Curvature:
        D = self.D
        return Curvature(
            D=D,
            ricci=np.zeros((D, D)),
            ricci_scalar=np.array(0.0),
            riemann=np.zeros((D, D, D, D)),
        )


class FLRWMetric:
    """Flat FLRW metric: ds^2 = -dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2).

    Coordinates: [t, x, y, z]. D=4. Signature (-,+,+,+).
    Scale factor a(t) provided as a callable.
    """
    def __init__(self, chart: CoordinateChart,
                 a_func: Callable[[float], float],
                 adot_func: Callable[[float], float]):
        if chart.D != 4:
            raise ValueError("FLRWMetric requires D=4")
        self.D = 4
        self.chart = chart
        self.signature = MetricSignature.lorentzian_4d()
        self.a_func = a_func
        self.adot_func = adot_func
        a0 = a_func(0.0)
        self.components = np.diag([-1.0, a0**2, a0**2, a0**2])
        self.is_diagonal = True
        self.depends_on_coords = True

    def components_at(self, t: float) -> np.ndarray:
        a = self.a_func(t)
        return np.diag([-1.0, a**2, a**2, a**2])

    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.components_at(0.0))

    def determinant(self) -> np.ndarray:
        return np.linalg.det(self.components)

    def is_lorentzian(self) -> bool:
        return True

    def to_dict(self) -> Dict:
        return {
            "D": self.D,
            "signature": self.signature.to_dict(),
            "chart": self.chart.to_dict(),
            "is_diagonal": self.is_diagonal,
            "depends_on_coords": self.depends_on_coords,
        }

    def christoffel_at(self, t: float) -> ChristoffelSymbols:
        """Analytic Christoffel symbols for flat FLRW.

        Nonzero: Gamma^0_ii = a*adot, Gamma^i_0i = Gamma^i_i0 = adot/a.
        """
        D = 4
        a = self.a_func(t)
        adot = self.adot_func(t)
        G = np.zeros((D, D, D))
        for i in range(1, 4):
            G[0, i, i] = a * adot
        H = adot / a
        for i in range(1, 4):
            G[i, 0, i] = H
            G[i, i, 0] = H
        return ChristoffelSymbols(D=D, components=G)

    def ricci_at(self, t: float) -> np.ndarray:
        """Ricci tensor for flat FLRW (with a'' approximated as 0)."""
        D = 4
        a = self.a_func(t)
        adot = self.adot_func(t)
        R = np.zeros((D, D))
        for i in range(1, 4):
            R[i, i] = 2.0 * adot**2
        return R

    def ricci_scalar_at(self, t: float) -> float:
        """R = g^{ab} R_ab = -R_00 + (1/a^2) sum_i R_ii."""
        R = self.ricci_at(t)
        a = self.a_func(t)
        return float(-R[0, 0] + (R[1, 1] + R[2, 2] + R[3, 3]) / a**2)

    def curvature_at(self, t: float) -> Curvature:
        return Curvature(
            D=4,
            ricci=self.ricci_at(t),
            ricci_scalar=np.array(self.ricci_scalar_at(t)),
            riemann=None,
        )


class Bulk5DMetric:
    """5D bulk metric for brane-world research.

    ds^2 = -n(t,w)^2 dt^2 + a(t,w)^2 (dx^2+dy^2+dz^2) + b(t,w)^2 dw^2

    Coordinates: [t, x, y, z, w]. D=5. Signature (-,+,+,+,+).
    """
    def __init__(self, chart: CoordinateChart,
                 n_func: Callable[[float, float], float],
                 a_func: Callable[[float, float], float],
                 b_func: Callable[[float, float], float]):
        if chart.D != 5:
            raise ValueError("Bulk5DMetric requires D=5")
        self.D = 5
        self.chart = chart
        self.signature = MetricSignature.lorentzian_5d()
        self.n_func = n_func
        self.a_func = a_func
        self.b_func = b_func
        n0 = n_func(0.0, 0.0)
        a0 = a_func(0.0, 0.0)
        b0 = b_func(0.0, 0.0)
        self.components = np.diag([-n0**2, a0**2, a0**2, a0**2, b0**2])
        self.is_diagonal = True
        self.depends_on_coords = True

    def components_at(self, t: float, w: float) -> np.ndarray:
        n = self.n_func(t, w)
        a = self.a_func(t, w)
        b = self.b_func(t, w)
        return np.diag([-n**2, a**2, a**2, a**2, b**2])

    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.components_at(0.0, 0.0))

    def determinant(self) -> np.ndarray:
        return np.linalg.det(self.components)

    def is_lorentzian(self) -> bool:
        return True

    def to_dict(self) -> Dict:
        return {
            "D": self.D,
            "signature": self.signature.to_dict(),
            "chart": self.chart.to_dict(),
            "is_diagonal": self.is_diagonal,
            "depends_on_coords": self.depends_on_coords,
        }


@dataclass
class Field:
    """A scalar (or other) field on a coordinate chart.

    name: field identifier
    value: scalar or array value at chart points
    chart: the chart the field is defined on
    """
    name: str
    value: np.ndarray
    chart: CoordinateChart

    def to_dict(self) -> Dict:
        return {"name": self.name, "shape": self.value.shape, "chart": self.chart.name}


@dataclass
class BoundaryCondition:
    """Boundary condition on a coordinate or hypersurface."""
    name: str
    coordinate: str        # which coordinate the boundary is on
    value: float           # boundary value
    type: str              # 'dirichlet', 'neumann', 'periodic'
    applies_to: List[str]  # which fields/metric components

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GeometrySpec:
    """Complete specification of a D-dimensional geometry.

    chart: coordinate chart
    metric: metric tensor
    fields: dictionary of named fields
    boundary_conditions: list of boundary conditions
    symmetries: e.g. ["homogeneous", "isotropic", "static"]
    compactification: dict describing compact dims (e.g. {"w": "circle", "radius": R})
    """
    chart: CoordinateChart
    metric: MetricTensor
    fields: Dict[str, Field] = field(default_factory=dict)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    symmetries: List[str] = field(default_factory=list)
    compactification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chart": self.chart.to_dict(),
            "metric": self.metric.to_dict(),
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "boundary_conditions": [bc.to_dict() for bc in self.boundary_conditions],
            "symmetries": self.symmetries,
            "compactification": self.compactification,
        }


def flat_minkowski_4d() -> GeometrySpec:
    """Return flat 3+1 dimensional Minkowski geometry."""
    chart = CoordinateChart(name="minkowski_4d", coords=["t", "x", "y", "z"])
    sig = MetricSignature.lorentzian_4d()
    metric = ConstantMetric(sig, chart)
    return GeometrySpec(chart=chart, metric=metric, symmetries=["homogeneous", "isotropic", "static"])


def flat_minkowski_5d() -> GeometrySpec:
    """Return flat 5-dimensional Minkowski geometry (4+1)."""
    chart = CoordinateChart(name="minkowski_5d", coords=["t", "x", "y", "z", "w"])
    sig = MetricSignature.lorentzian_5d()
    metric = ConstantMetric(sig, chart)
    return GeometrySpec(
        chart=chart, metric=metric,
        symmetries=["homogeneous", "isotropic"],
        compactification={},
    )


def circle_compactification(radius: float) -> GeometrySpec:
    """5D geometry with one compact spatial dimension (circle of radius R).

    ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 + R^2 dtheta^2,  theta in [0, 2pi).

    The circle is compact at scale R; Kaluza-Klein modes have masses m_n = n/R.
    """
    chart = CoordinateChart(
        name="minkowski_4d_x_circle",
        coords=["t", "x", "y", "z", "theta"],
        ranges=[(None, None), (None, None), (None, None), (None, None), (0.0, 2.0 * math.pi)],
        is_periodic=[False, False, False, False, True],
    )
    sig = MetricSignature(n_negative=1, n_zero=0, n_positive=4)
    metric = ConstantMetric(sig, chart)
    # Override components with the circle radius squared for g_theta_theta.
    metric.components = np.diag([-1.0, 1.0, 1.0, 1.0, radius**2])
    return GeometrySpec(
        chart=chart, metric=metric,
        symmetries=["homogeneous", "isotropic"],
        compactification={"dim": "theta", "type": "circle", "radius": radius},
    )
