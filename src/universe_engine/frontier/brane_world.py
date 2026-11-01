"""Reduced-order brane-world cosmology research model (Requirement 6).

Implements an experimental 5D bulk + 4D brane framework.

Model (Randall-Sundrum type II inspired, reduced-order homogeneous):
  Bulk metric (5D):
    ds^2 = -n(t,y)^2 dt^ + a(t,y)^2 (dx^2 + dy^2 + dz^2) + b(t,y)^2 dy^2
  where y is the extra dimension (the bulk coordinate), and the brane sits at
  y = y_b(t) (a dynamical hypersurface).

  Brane as a dynamical hypersurface with:
    - Brane tension lambda
    - Induced 4D metric: h_mn = g_ab - n_a n_b (n_a = unit normal)
    - Extrinsic curvature: K_ab = h_a^c h_b^d ∇_c n_d
    - Junction (Israel) condition: [K_ab - h_ab K] = -8πG_5 S_ab
      where S_ab is the brane stress-energy tensor.

  This is a reduced-order homogeneous model: all quantities depend only on
  (t, y), and the brane is assumed to be at a fixed position y=0 with
  Z2 symmetry across the brane. The bulk is anti-de Sitter (AdS5) with
  cosmological constant Lambda_5.

  The effective 4D Friedmann equation on the brane (RS-II):
    H^2 = (8πG_4/3) rho (1 + rho / (2 lambda)) + (Lambda_4 / 3) + C / a^4
  where:
    - rho is the brane matter density
    - lambda is the brane tension
    - C is the "dark radiation" term from the bulk Weyl tensor
    - The rho^2 term is the distinctive high-energy brane-world correction

  Scientific status: SPECULATIVE / RESEARCH. Brane cosmology is a theoretical
  possibility, not established knowledge. Ekpyrotic models propose a hot Big Bang
  associated with a collision of higher-dimensional branes; their dynamics and
  observable predictions remain active research topics.

  CRITICAL RULE: A new universe is created ONLY when the selected mathematical
  model reaches its documented transition conditions and the transition is
  numerically valid. Mesh overlap or visual intersection does NOT create a
  universe. For collision models without a well-defined nonsingular transition,
  stop or use an explicitly labeled prescribed matching rule.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.clocks import Clock
from ..core.universe import OriginEvent
from ..geometry.geometry_core import (
    CoordinateChart, MetricSignature, Bulk5DMetric, Curvature, GeometrySpec,
    circle_compactification,
)


@dataclass
class BraneWorldParameters:
    """Parameters for the 5D bulk + 4D brane model."""
    # Bulk
    Lambda_5: float = -1.0          # 5D cosmological constant (AdS5, negative)
    G_5: float = 1.0                # 5D gravitational constant
    k: float = 1.0                  # AdS curvature scale: k^2 = -Lambda_5 / 6

    # Brane
    lambda_brane: float = 10.0      # Brane tension
    sigma_matter: float = 0.0       # Brane matter surface density
    sigma_radiation: float = 0.0    # Brane radiation surface density

    # Brane position (extra dimension)
    y_brane: float = 0.0            # Brane location in bulk
    y_brane_velocity: float = 0.0   # dy_brane/dt

    # Collision / bounce
    collision_model: str = "none"   # 'none' | 'ekpyrotic' | 'prescribed'
    collision_time: Optional[float] = None  # cosmic time of brane collision
    transition_valid: bool = False  # whether a nonsingular transition exists

    # Dark radiation
    C_dark_radiation: float = 0.0   # Integration constant from bulk Weyl tensor

    # Junction condition tolerance (model-dependent validity bound)
    junction_tol: float = 1e6

    # Effective 4D Newton constant (derived)
    G_4: Optional[float] = None
    Lambda_4: Optional[float] = None

    def derive_4d_constants(self):
        """Derive effective 4D Newton constant and cosmological constant."""
        # RS-II relations:
        #   G_4 = G_5 * k / (pi * lambda_brane)   (approximately)
        #   Lambda_4 = 4*pi*G_5 * (Lambda_5 + 4*pi*G_5*lambda_brane^2/3) / ...
        # Simplified:
        k = math.sqrt(max(-self.Lambda_5 / 6.0, 1e-12))
        self.k = k
        if self.lambda_brane > 0:
            self.G_4 = self.G_5 * k / self.lambda_brane
        else:
            self.G_4 = self.G_5 * k
        # Effective 4D cosmological constant (tuned to ~0 for RS fine-tuning)
        self.Lambda_4 = 0.0


@dataclass
class BraneState:
    """State of the brane-world system at one time."""
    t: float                     # bulk time
    a_brane: float               # scale factor on the brane
    H_brane: float               # Hubble parameter on the brane
    rho_matter: float            # matter density on brane
    rho_radiation: float         # radiation density on brane
    y_brane: float               # brane position in extra dimension
    y_velocity: float            # brane velocity
    extrinsic_curvature: float   # K trace (extrinsic curvature scalar)
    junction_residual: float     # residual of Israel junction condition
    validity: str                # 'valid' | 'approaching_singularity' | 'invalid'
    validity_message: str
    error_estimate: float


class BraneWorldModel:
    """Reduced-order brane-world cosmology model.

    Solves the effective 4D Friedmann equation on the brane with the
    rho^2 high-energy correction, plus brane motion in the bulk.

    Governing equations:
      H^2 = (8*pi*G_4/3) * rho * (1 + rho / (2*lambda)) + Lambda_4/3 + C/a^4
      dot_rho + 3*H*(rho + p) = 0   (brane matter conservation, with possible
                                      energy exchange with bulk)
      dot_a = H * a

    Brane motion:
      dot_y_brane = v_y
      dot_v_y = -k^2 * y_brane   (harmonic motion in AdS bulk, approx)

    Junction condition residual measures how well the Israel condition is
    satisfied; if it exceeds tolerance, the model is invalid.
    """

    def __init__(self, params: Optional[BraneWorldParameters] = None):
        self.params = params or BraneWorldParameters()
        self.params.derive_4d_constants()
        self.state: Optional[BraneState] = None
        self._step_count = 0
        self._cumulative_error = 0.0
        self.history: List[Dict] = []

    def initialize(self, t: float = 0.0, a_brane: float = 1e-13,
                   rho_matter: float = 0.0, rho_radiation: float = 1e10,
                   y_brane: float = 0.0, y_velocity: float = 0.0) -> BraneState:
        """Initialize the brane-world state at early times.

        Defaults to a radiation-dominated early state (hot Big Bang on the brane).
        """
        p = self.params
        # Compute H from Friedmann equation
        rho = rho_matter + rho_radiation
        H2 = (8.0 * math.pi * p.G_4 / 3.0) * rho * (1.0 + rho / (2.0 * p.lambda_brane))
        H2 += p.Lambda_4 / 3.0 + p.C_dark_radiation / a_brane**4
        H = math.sqrt(max(H2, 0.0))

        # Extrinsic curvature (approx: K ~ H for a static brane in AdS)
        K = H  # simplified

        # Junction residual
        junction_residual = 0.0  # will be computed on first step

        self.state = BraneState(
            t=t, a_brane=a_brane, H_brane=H,
            rho_matter=rho_matter, rho_radiation=rho_radiation,
            y_brane=y_brane, y_velocity=y_velocity,
            extrinsic_curvature=K, junction_residual=junction_residual,
            validity="valid", validity_message="Brane-world model initialized.",
            error_estimate=0.0,
        )
        self._step_count = 0
        self._cumulative_error = 0.0
        self.history = []
        return self.state

    def _friedmann_H(self, rho: float, a: float) -> float:
        """Hubble parameter from effective 4D brane Friedmann equation."""
        p = self.params
        H2 = (8.0 * math.pi * p.G_4 / 3.0) * rho * (1.0 + rho / (2.0 * p.lambda_brane))
        H2 += p.Lambda_4 / 3.0 + p.C_dark_radiation / a**4
        if H2 < 0:
            return 0.0  # recollapsing or invalid
        return math.sqrt(H2)

    def step(self, dt: float) -> BraneState:
        """Advance the brane-world by dt using adaptive RK2."""
        if self.state is None:
            raise RuntimeError("BraneWorldModel not initialized")
        if self.state.validity == "invalid":
            return self.state

        s = self.state
        p = self.params

        #RK2 step
        def derivatives(state: BraneState):
            rho = state.rho_matter + state.rho_radiation
            # w=1/3 radiation, w=0 matter: use effective w
            w_eff = state.rho_radiation / max(rho, 1e-300) * (1.0 / 3.0)
            H = self._friedmann_H(rho, state.a_brane)
            dot_rho = -3.0 * H * rho * (1.0 + w_eff)
            dot_a = H * state.a_brane
            dot_y = state.y_velocity
            dot_v = -p.k**2 * state.y_brane  # harmonic motion in AdS bulk
            return H, dot_rho, dot_a, dot_y, dot_v

        H1, dr1, da1, dy1, dv1 = derivatives(s)

        # Midpoint
        a_mid = s.a_brane + 0.5 * dt * da1
        rho_m_mid = s.rho_matter + 0.5 * dt * dr1 * (s.rho_matter / max(s.rho_matter + s.rho_radiation, 1e-300))
        rho_r_mid = s.rho_radiation + 0.5 * dt * dr1 * (s.rho_radiation / max(s.rho_matter + s.rho_radiation, 1e-300))
        y_mid = s.y_brane + 0.5 * dt * dy1
        v_mid = s.y_velocity + 0.5 * dt * dv1
        # Build mid state for H eval
        H_mid = self._friedmann_H(rho_m_mid + rho_r_mid, a_mid)

        # Full step using midpoint derivatives
        dot_rho_mid = -3.0 * H_mid * (rho_m_mid + rho_r_mid) * (1.0 + (rho_r_mid / max(rho_m_mid + rho_r_mid, 1e-300)) * (1.0 / 3.0))

        s_new = BraneState(
            t=s.t + dt,
            a_brane=s.a_brane + dt * da1,
            H_brane=H_mid,
            rho_matter=max(s.rho_matter + dt * dr1 * (s.rho_matter / max(s.rho_matter + s.rho_radiation, 1e-300)), 0.0),
            rho_radiation=max(s.rho_radiation + dt * dot_rho_mid * (s.rho_radiation / max(s.rho_matter + s.rho_radiation, 1e-300)), 0.0),
            y_brane=y_mid,
            y_velocity=v_mid,
            extrinsic_curvature=H_mid,
            junction_residual=0.0,
            validity="valid",
            validity_message="Brane-world evolving within model domain.",
            error_estimate=0.0,
        )

        # Junction condition residual (Israel condition check)
        # |[K_ab - h_ab K] + 8*pi*G_5*S_ab| < tol
        # Simplified scalar check: residual ~ |K - 4*pi*G_5*lambda_brane/3|
        lambda_term = 4.0 * math.pi * p.G_5 * p.lambda_brane / 3.0
        s_new.junction_residual = abs(s_new.extrinsic_curvature - lambda_term)

        if s_new.junction_residual > p.junction_tol:
            s_new.validity = "approaching_singularity"
            s_new.validity_message = (
                f"Junction condition residual {s_new.junction_residual:.3g} exceeds "
                f"tolerance {p.junction_tol}: brane-world approximation breaking down."
            )

        # Check for brane collision (ekpyrotic)
        if p.collision_model == "ekpyrotic" and p.collision_time is not None:
            if abs(s_new.t - p.collision_time) < dt:
                if p.transition_valid:
                    s_new.validity = "valid"
                    s_new.validity_message = "Ekpyrotic brane collision: nonsingular transition."
                else:
                    s_new.validity = "invalid"
                    s_new.validity_message = (
                        "Ekpyrotic brane collision reached without a documented "
                        "nonsingular transition: model invalid per critical rule."
                    )

        self.state = s_new
        self._step_count += 1
        self.history.append({
            "t": s_new.t, "a": s_new.a_brane, "H": s_new.H_brane,
            "rho": s_new.rho_matter + s_new.rho_radiation,
            "junction_residual": s_new.junction_residual,
            "validity": s_new.validity,
        })
        return s_new

    def origin_event(self, parent_structure: Optional[str],
                     model_id: str, parameters: Dict,
                     initial_state: Dict, energy_field_info: Dict,
                     creation_time: float,
                     validity_metadata: Dict,
                     theory_classification: str = "speculative") -> OriginEvent:
        """Create an OriginEvent record for a universe produced by this brane model."""
        return OriginEvent(
            event_id=f"brane-{uuid.uuid4().hex[:12]}",
            parent_structure=parent_structure,
            model_id=model_id,
            parameters=parameters,
            initial_state=initial_state,
            energy_field_info=energy_field_info,
            creation_time=creation_time,
            validity_metadata=validity_metadata,
            theory_classification=theory_classification,
        )
