"""Universe state: one evolving universe with its own clocks, cosmology, and provenance.

Each Universe instance owns:
  - A UniverseClockSet (independent time coordinates)
  - An FLWRCosmology (Friedmann background)
  - Current FLWRState (scale factor, Hubble, densities, etc.)
  - Model-dependent origin event metadata
  - Validity metadata and numerical error estimates

Universes are isolated: two universes never share clocks or state.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.clocks import (
    Clock, UniverseClockSet, SimScheduler, make_default_clock_set,
    FLWRCosmology, FLWRState, SECONDS_PER_GYR,
)
from ..core.clocks import cosmic_time_from_scale_factor, scale_factor_from_cosmic_time


@dataclass
class OriginEvent:
    """Metadata for a model-dependent universe-origin event.

    Every origin event records:
      - event_id: unique identity
      - parent_structure: parent cosmological structure when applicable (None for reference)
      - model_id: identifier of the physical model that produced this universe
      - parameters: the parameter set used
      - initial_state: initial physical state snapshot
      - energy_field_info: energy and field information defined by the model
      - creation_time: model coordinate / cosmic time of creation
      - validity_metadata: numerical validity information
      - theory_classification: 'reference' | 'speculative' | 'research'
    """
    event_id: str
    parent_structure: Optional[str]
    model_id: str
    parameters: Dict[str, Any]
    initial_state: Dict[str, Any]
    energy_field_info: Dict[str, Any]
    creation_time: float
    validity_metadata: Dict[str, Any]
    theory_classification: str  # 'reference' | 'speculative' | 'research'


@dataclass
class UniverseState:
    """Mutable physical state of one universe at its current cosmic time."""
    universe_id: str
    a: float                     # scale factor
    t_cosmic_s: float            # cosmic proper time (seconds)
    H: float                     # Hubble parameter (s^-1)
    q: float                     # deceleration parameter
    Omega_r: float               # radiation density parameter
    Omega_m: float               # matter density parameter
    Omega_Lambda: float          # dark-energy density parameter
    z: float                     # redshift
    age_Gyr: float               # cosmic age in Gyr
    expanding: bool
    validity: str                # 'valid' | 'approaching_limit' | 'invalid' | 'paused'
    validity_message: str
    numerical_error_estimate: float
    origin_event: Optional[OriginEvent] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["origin_event"] = asdict(self.origin_event) if self.origin_event else None
        return d


class Universe:
    """One evolving universe with independent clocks and its own FLRW cosmology.

    The universe advances via the FLWRCosmology Friedmann equations. Its scale
    factor can grow without bound (a > 1 means future evolution beyond today).
    If the model becomes invalid, the universe pauses with a scientific
    explanation rather than inventing continuation.
    """

    def __init__(self, universe_id: str, name: str, seed: int,
                 cosmology: Dict[str, float],
                 model_settings: Optional[Dict[str, Any]] = None,
                 origin_event: Optional[OriginEvent] = None,
                 clock_set: Optional[UniverseClockSet] = None):
        self.uuid = universe_id
        self.name = name
        self.seed = int(seed)
        self.cosmology = dict(cosmology)
        self.model_settings = dict(model_settings or {})
        self.status = "created"  # created | running | paused | finished | invalid
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.last_saved: Optional[str] = None
        self.provenance = ""
        self.schema = "universe/v1"
        self.engine_version = "0.1.0"

        # Clocks: each universe owns its own independent set.
        self.clocks = clock_set or make_default_clock_set(
            cosmic_t_max=cosmology.get("cosmic_t_max_s"),
            scale_factor_t_max=cosmology.get("scale_factor_t_max_a"),
        )

        # Friedmann cosmology.
        self.flrw = FLWRCosmology(
            H0_km_s_Mpc=float(cosmology["H0_km_s_Mpc"]),
            Omega_m=float(cosmology["Omega_m"]),
            Omega_r=float(cosmology["Omega_r"]),
            Omega_Lambda=float(cosmology["Omega_Lambda"]),
        )

        # Origin event.
        self.origin_event = origin_event

        # Initialize state at a_start.
        a_start = float(cosmology.get("a_start", 1e-13))
        self._state = self.flrw.state_at(a_start)
        self.clocks.cosmic.t = self._state.t
        self.clocks.scale_factor.t = self._state.a
        self.clocks.bulk.t = 0.0

        # Numerical error tracking.
        self._cumulative_error = 0.0
        self._last_dt = 0.0
        self._step_count = 0

        # Classification of this universe's cosmology.
        self.provenance = self._classify()

    def _classify(self) -> str:
        c = self.cosmology
        flat = abs(c["Omega_m"] + c["Omega_r"] + c["Omega_Lambda"] - 1.0)
        if (flat < 1e-6
                and 0.25 <= c["Omega_m"] <= 0.35
                and 0.60 <= c["Omega_Lambda"] <= 0.75
                and 0 <= c["Omega_r"] <= 2e-4
                and 60 <= c["H0_km_s_Mpc"] <= 75):
            return "observational"
        if flat < 1e-4 and 0.05 <= c["Omega_m"] <= 0.95 and 0.0 <= c["Omega_Lambda"] <= 0.95:
            return "hypothetical"
        return "outside_validated_domain"

    # ------------------------------------------------------------------ accessors
    @property
    def state(self) -> UniverseState:
        return self._build_state()

    def _build_state(self) -> UniverseState:
        s = self._state
        return UniverseState(
            universe_id=self.uuid,
            a=s.a, t_cosmic_s=s.t, H=s.H, q=s.q,
            Omega_r=s.Omega_r, Omega_m=s.Omega_m, Omega_Lambda=s.Omega_Lambda,
            z=s.z, age_Gyr=s.age_Gyr, expanding=s.expanding,
            validity=self.status if self.status in ("valid", "invalid", "paused") else "valid",
            validity_message=self._validity_message(),
            numerical_error_estimate=self._cumulative_error,
            origin_event=self.origin_event,
        )

    def _validity_message(self) -> str:
        if self.status == "invalid":
            return "Universe paused: selected model is outside its validity domain."
        if self.status == "paused":
            return "Universe paused by user or at a validity boundary."
        return "Universe evolving within model validity domain."

    # ------------------------------------------------------------------ stepping
    def advance(self, dt_master_s: float, scheduler: SimScheduler) -> Dict[str, Any]:
        """Advance the universe by dt_master_s (seconds of master time).

        Uses adaptive timestepping in the scale factor. Returns diagnostics.
        """
        if self.status in ("invalid", "finished"):
            return {"universe": self.uuid, "status": self.status, "advanced_s": 0.0}

        # Propose timestep from subsystem limits (here just the cosmological dt).
        # The cosmological dt is derived from the Hubble time: dt <= 0.1 / H.
        h = self._state.H
        cosmological_dt = 0.1 / h if h > 0 else 1e10

        subsystem_limits = {"cosmology": cosmological_dt}
        dt = scheduler.propose_dt(self.uuid, subsystem_limits)

        # Advance the scale factor: da/dt = a * H(a)  =>  d ln a / dt = H
        # We integrate in ln(a) with adaptive steps.
        a_curr = self._state.a
        t_curr = self._state.t
        dt_remaining = dt
        steps_taken = 0
        error_est = 0.0

        while dt_remaining > 1e-18 and self.status not in ("invalid", "finished"):
            # Step size in ln(a): d ln a = H * dt
            d_ln_a = h * dt_remaining
            # Limit d ln a for accuracy (adaptive).
            max_d_ln_a = 0.01
            if abs(d_ln_a) > max_d_ln_a:
                d_ln_a = max_d_ln_a * (1.0 if d_ln_a > 0 else -1.0)
                dt_used = abs(d_ln_a / h) if h > 0 else dt_remaining
            else:
                dt_used = dt_remaining

            a_next = a_curr * math.exp(d_ln_a)
            # Compute t at a_next via integral from a_curr.
            from ..core.clocks import _integrate_da_over_aH
            dt_cosmic = _integrate_da_over_aH(self.flrw, a_curr, a_next)
            t_next = t_curr + dt_cosmic

            # Local error estimate: compare with midpoint.
            a_mid = math.sqrt(a_curr * a_next)
            t_mid = _integrate_da_over_aH(self.flrw, a_curr, a_mid)
            dt_mid = t_curr + t_mid
            # second half
            dt_second = _integrate_da_over_aH(self.flrw, a_mid, a_next)
            error_est = abs((dt_mid + dt_second) - dt_cosmic) / max(abs(dt_cosmic), 1e-300)

            # Update state.
            self._state = self.flrw.state_at(a_next, t_next)
            self.clocks.cosmic.t = t_next
            self.clocks.scale_factor.t = a_next
            self.clocks.bulk.t += dt_used

            a_curr, t_curr = a_next, t_next
            dt_remaining -= dt_used
            steps_taken += 1
            self._last_dt = dt_used
            self._step_count += 1

            # Check validity bounds.
            max_a = self.clocks.scale_factor.t_max
            if max_a is not None and a_curr >= max_a:
                self.status = "paused"
                break

        self._cumulative_error += error_est
        self._state = self.flrw.state_at(a_curr, t_curr)
        self.clocks.cosmic.t = t_curr
        self.clocks.scale_factor.t = a_curr

        if self.status != "paused":
            self.status = "running"

        return {
            "universe": self.uuid,
            "status": self.status,
            "advanced_s": dt,
            "a": a_curr,
            "t_cosmic_s": t_curr,
            "age_Gyr": t_curr / SECONDS_PER_GYR,
            "H": self._state.H,
            "steps": steps_taken,
            "error_estimate": error_est,
        }

    def pause(self) -> None:
        self.status = "paused"
        for clk in [self.clocks.cosmic, self.clocks.scale_factor, self.clocks.bulk,
                    self.clocks.galactic, self.clocks.planetary, self.clocks.chemical,
                    self.clocks.biological, self.clocks.agent, self.clocks.render]:
            clk.paused = True

    def resume(self) -> None:
        if self.status == "invalid":
            return
        self.status = "running"
        for clk in [self.clocks.cosmic, self.clocks.scale_factor, self.clocks.bulk,
                    self.clocks.galactic, self.clocks.planetary, self.clocks.chemical,
                    self.clocks.biological, self.clocks.agent, self.clocks.render]:
            clk.paused = False

    def invalidate(self, reason: str) -> None:
        """Mark the universe as invalid because its model left its validity domain."""
        self.status = "invalid"
        self.pause()
        self._validity_reason = reason

    def checkpoint(self) -> Dict[str, Any]:
        """Produce a serializable checkpoint of this universe."""
        return {
            "schema": "universe-checkpoint/v1",
            "universe_id": self.uuid,
            "name": self.name,
            "seed": self.seed,
            "cosmology": self.cosmology,
            "model_settings": self.model_settings,
            "status": self.status,
            "created_at": self.created_at,
            "a": self._state.a,
            "t_cosmic_s": self._state.t,
            "H": self._state.H,
            "provenance": self.provenance,
            "origin_event": asdict(self.origin_event) if self.origin_event else None,
            "clocks": {
                name: {"t": clk.t, "rate": clk.rate, "paused": clk.paused,
                       "step_count": clk.step_count}
                for name, clk in [
                    ("bulk", self.clocks.bulk), ("cosmic", self.clocks.cosmic),
                    ("scale_factor", self.clocks.scale_factor),
                    ("galactic", self.clocks.galactic),
                    ("planetary", self.clocks.planetary),
                    ("chemical", self.clocks.chemical),
                    ("biological", self.clocks.biological),
                    ("agent", self.clocks.agent),
                    ("render", self.clocks.render),
                ]
            },
            "cumulative_error": self._cumulative_error,
            "step_count": self._step_count,
        }

    @classmethod
    def restore(cls, data: Dict[str, Any]) -> "Universe":
        """Restore a universe from a checkpoint."""
        uni = cls.__new__(cls)
        uni.uuid = data["universe_id"]
        uni.name = data["name"]
        uni.seed = int(data["seed"])
        uni.cosmology = data["cosmology"]
        uni.model_settings = data.get("model_settings", {})
        uni.status = data.get("status", "paused")
        uni.created_at = data.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        uni.last_saved = None
        uni.provenance = data.get("provenance", "")
        uni.schema = data.get("schema", "universe/v1")
        uni.engine_version = "0.1.0"

        uni.flrw = FLWRCosmology(
            H0_km_s_Mpc=float(uni.cosmology["H0_km_s_Mpc"]),
            Omega_m=float(uni.cosmology["Omega_m"]),
            Omega_r=float(uni.cosmology["Omega_r"]),
            Omega_Lambda=float(uni.cosmology["Omega_Lambda"]),
        )

        # Rebuild clocks.
        a_val = float(data["a"])
        t_val = float(data["t_cosmic_s"])
        uni.clocks = make_default_clock_set(
            cosmic_t_max=uni.cosmology.get("cosmic_t_max_s"),
            scale_factor_t_max=uni.cosmology.get("scale_factor_t_max_a"),
        )
        uni.clocks.cosmic.t = t_val
        uni.clocks.scale_factor.t = a_val
        uni.clocks.bulk.t = t_val  # approximate
        for name, clk_data in data.get("clocks", {}).items():
            clk = getattr(uni.clocks, name, None)
            if clk is not None:
                clk.t = float(clk_data.get("t", clk.t))
                clk.rate = float(clk_data.get("rate", clk.rate))
                clk.paused = bool(clk_data.get("paused", False))
                clk.step_count = int(clk_data.get("step_count", 0))

        uni._state = uni.flrw.state_at(a_val, t_val)
        uni._cumulative_error = float(data.get("cumulative_error", 0.0))
        uni._step_count = int(data.get("step_count", 0))
        uni._last_dt = 0.0

        oe = data.get("origin_event")
        uni.origin_event = OriginEvent(**oe) if oe else None
        return uni

    def to_public(self) -> Dict[str, Any]:
        s = self.state
        return {
            "uuid": self.uuid, "name": self.name, "seed": self.seed,
            "status": self.status, "created_at": self.created_at,
            "a": s.a, "t_cosmic_s": s.t_cosmic_s, "age_Gyr": s.age_Gyr,
            "H": s.H, "q": s.q, "z": s.z,
            "Omega_r": s.Omega_r, "Omega_m": s.Omega_m, "Omega_Lambda": s.Omega_Lambda,
            "expanding": s.expanding, "validity": s.validity,
            "validity_message": s.validity_message,
            "error_estimate": s.numerical_error_estimate,
            "provenance": self.provenance,
            "origin_event": s.origin_event,
            "last_saved": self.last_saved,
        }
