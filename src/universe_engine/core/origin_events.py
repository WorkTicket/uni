"""Model-dependent universe-origin events (Requirement 7 / Test F).

Origin events are produced by mathematical models reaching documented transition
conditions, NOT by arbitrary visual overlap or random per-frame spawning.

Each origin event records:
  - Event identity (UUID)
  - Parent cosmological structure when applicable
  - Model identifier
  - Parameter set
  - Initial physical state
  - Energy and field information defined by the model
  - Creation event time or model coordinate
  - Numerical validity metadata
  - Origin theory classification ('reference' | 'speculative' | 'research')

Supported origin mechanisms are separate plugins:
  - Reference FLRW: hot Big Bang from specified early conditions
  - Ekpyrotic/cyclic: brane collision -> hot phase transition
  - Inflationary bubble: scalar-field vacuum transition (if implemented)

A non-qualifying interaction produces no new universe.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from ..core.universe import OriginEvent, Universe, UniverseState
from ..core.clocks import FLWRCosmology, SECONDS_PER_GYR


class OriginModel(Protocol):
    """Interface for origin-model plugins.

    Each plugin must specify:
      - governing equations
      - initial conditions
      - transition rules
      - validity domain
      - verification benchmarks
    """
    model_id: str
    theory_classification: str  # 'reference' | 'speculative' | 'research'

    def check_transition(self, state: UniverseState, t: float) -> Optional[Dict]:
        """Return transition info if conditions are met, else None."""
        ...

    def create_initial_state(self, transition_info: Dict) -> Dict:
        """Produce the initial physical state for the new universe."""
        ...


@dataclass
class ReferenceFLRWOrigin:
    """Reference FLRW origin: hot Big Bang from specified early conditions.

    Implements an early hot, dense expanding state using radiation/matter
    evolution with documented initial conditions. Does NOT claim to describe
    the singularity itself.
    """
    model_id: str = "flrw_reference"
    theory_classification: str = "reference"

    def __init__(self, H0_km_s_Mpc: float = 67.4,
                 Omega_m: float = 0.315, Omega_r: float = 9.2e-5,
                 Omega_Lambda: float = 0.684908,
                 a_start: float = 1e-13,
                 T_start_K: float = 1e16):
        self.H0_km_s_Mpc = H0_km_s_Mpc
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_Lambda = Omega_Lambda
        self.a_start = a_start
        self.T_start_K = T_start_K

    def check_transition(self, state: UniverseState, t: float) -> Optional[Dict]:
        """Reference FLRW has no transition condition; it starts directly."""
        return None

    def create_initial_state(self, transition_info: Optional[Dict] = None) -> Dict:
        """Initial hot Big Bang state."""
        return {
            "a": self.a_start,
            "T_K": self.T_start_K,
            "rho_radiation": self.Omega_r * self.H0_km_s_Mpc**2,  # scaled
            "rho_matter": self.Omega_m * self.H0_km_s_Mpc**2,
            "equation_of_state": "radiation",
            "era": "radiation_dominated",
        }


@dataclass
class EkpyroticOrigin:
    """Ekpyrotic/cyclic research model: brane collision -> hot Big Bang.

    Models contraction, brane interaction, hot-phase transition, and subsequent
    cosmological evolution. The collision produces a hot Big Bang state on the
    brane only when a documented nonsingular transition exists.

    SPECULATIVE / RESEARCH: brane cosmology is a theoretical possibility,
    not established knowledge.
    """
    model_id: str = "ekpyrotic_brane_collision"
    theory_classification: str = "speculative"

    def __init__(self, Lambda_5: float = -6.0, G_5: float = 1.0,
                 lambda_brane: float = 10.0,
                 collision_time: float = 1.0,
                 transition_valid: bool = False):
        self.Lambda_5 = Lambda_5
        self.G_5 = G_5
        self.lambda_brane = lambda_brane
        self.collision_time = collision_time
        self.transition_valid = transition_valid

    def check_transition(self, state: UniverseState, t: float) -> Optional[Dict]:
        """Check if brane collision conditions are met.

        Returns transition info only if:
          - t reaches collision_time
          - transition_valid is True (nonsingular transition documented)
        """
        if abs(t - self.collision_time) < 1e-6:
            if self.transition_valid:
                return {
                    "transition_type": "brane_collision",
                    "collision_time": self.collision_time,
                    "energy_scale": self.lambda_brane,
                    "nonsingular": True,
                }
            else:
                return {
                    "transition_type": "brane_collision",
                    "collision_time": self.collision_time,
                    "nonsingular": False,
                    "error": "No documented nonsingular transition",
                }
        return None

    def create_initial_state(self, transition_info: Dict) -> Dict:
        """Produce the hot Big Bang state after brane collision."""
        if not transition_info.get("nonsingular", False):
            return {
                "error": "Cannot create universe: no nonsingular transition.",
                "transition_type": transition_info.get("transition_type"),
            }
        return {
            "a": 1e-13,
            "T_K": 1e15,
            "rho_radiation": 1e10,
            "rho_matter": 0.0,
            "equation_of_state": "radiation",
            "era": "post_collision_hot_phase",
            "origin": "ekpyrotic_brane_collision",
            "collision_time": transition_info["collision_time"],
        }


@dataclass
class InflationaryBubbleOrigin:
    """Inflationary bubble research model (optional).

    Uses an explicitly specified scalar-field potential, vacuum states, and a
    transition/nucleation model. Not implemented unless a concrete potential is
    provided.
    """
    model_id: str = "inflationary_bubble"
    theory_classification: str = "research"

    def __init__(self, potential: Optional[Callable[[float], float]] = None,
                 transition_valid: bool = False):
        self.potential = potential
        self.transition_valid = transition_valid

    def check_transition(self, state: UniverseState, t: float) -> Optional[Dict]:
        if self.potential is None:
            return None
        return None  # placeholder

    def create_initial_state(self, transition_info: Optional[Dict] = None) -> Dict:
        return {"error": "inflationary bubble model not yet implemented"}


@dataclass
class UniverseOriginRecord:
    """Persistent record of a universe-origin event."""
    event_id: str
    parent_universe_id: Optional[str]
    new_universe_id: str
    model_id: str
    theory_classification: str
    parameters: Dict
    initial_state: Dict
    energy_field_info: Dict
    creation_time: float
    validity_metadata: Dict
    transition_info: Dict
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict:
        return asdict(self)


class OriginEventManager:
    """Manages model-dependent universe-origin events.

    Tracks origin models, checks transition conditions during evolution,
    and creates new persistent universes when transitions are numerically valid.
    """

    def __init__(self, multiverse):
        self.multiverse = multiverse
        self.origin_models: Dict[str, OriginModel] = {}
        self.origin_history: List[UniverseOriginRecord] = []
        self._register_default_models()

    def _register_default_models(self):
        self.register_model(ReferenceFLRWOrigin())
        self.register_model(EkpyroticOrigin())
        self.register_model(InflationaryBubbleOrigin())

    def register_model(self, model: OriginModel):
        self.origin_models[model.model_id] = model

    def check_all_transitions(self, universe_id: str,
                              state: UniverseState, t: float) -> List[Dict]:
        """Check all registered origin models for transition conditions.

        Returns a list of transition events that occurred.
        """
        events: List[Dict] = []
        for model in self.origin_models.values():
            info = model.check_transition(state, t)
            if info is not None:
                events.append({
                    "model_id": model.model_id,
                    "theory_classification": model.theory_classification,
                    "transition_info": info,
                    "universe_id": universe_id,
                    "t": t,
                })
        return events

    def create_universe_from_transition(self, transition: Dict,
                                        parent_universe: Universe,
                                        multiverse) -> Optional[UniverseOriginRecord]:
        """Create a new universe from a validated transition.

        Returns None if the transition does not produce a valid new universe
        (e.g., no nonsingular transition for ekpyrotic).
        """
        model_id = transition["model_id"]
        model = self.origin_models.get(model_id)
        if model is None:
            return None

        info = transition["transition_info"]
        # For ekpyrotic without valid transition, do NOT create a universe.
        if not info.get("nonsingular", True):
            # Record the failed transition but create no universe.
            record = UniverseOriginRecord(
                event_id=f"failed-{uuid.uuid4().hex[:12]}",
                parent_universe_id=parent_universe.uuid,
                new_universe_id="",
                model_id=model_id,
                theory_classification=transition["theory_classification"],
                parameters={},
                initial_state={"error": "no_nonsingular_transition"},
                energy_field_info={},
                creation_time=transition["t"],
                validity_metadata={"valid": False, "reason": info.get("error", "")},
                transition_info=info,
            )
            self.origin_history.append(record)
            return record

        # Create initial state for the new universe.
        initial_state = model.create_initial_state(info)
        if "error" in initial_state:
            record = UniverseOriginRecord(
                event_id=f"failed-{uuid.uuid4().hex[:12]}",
                parent_universe_id=parent_universe.uuid,
                new_universe_id="",
                model_id=model_id,
                theory_classification=transition["theory_classification"],
                parameters={},
                initial_state=initial_state,
                energy_field_info={},
                creation_time=transition["t"],
                validity_metadata={"valid": False, "reason": initial_state["error"]},
                transition_info=info,
            )
            self.origin_history.append(record)
            return record

        # Build cosmology for the new universe from the origin model.
        cosmology = {
            "H0_km_s_Mpc": parent_universe.cosmology["H0_km_s_Mpc"],
            "Omega_m": parent_universe.cosmology["Omega_m"],
            "Omega_r": parent_universe.cosmology["Omega_r"],
            "Omega_Lambda": parent_universe.cosmology["Omega_Lambda"],
            "a_start": initial_state.get("a", 1e-13),
            "a_end": parent_universe.cosmology.get("a_end", 2.0),
            "steps": parent_universe.cosmology.get("steps", 500),
            "max_a_end": parent_universe.cosmology.get("max_a_end", 10.0),
        }

        # Create origin event.
        origin_event = OriginEvent(
            event_id=f"origin-{uuid.uuid4().hex[:12]}",
            parent_structure=parent_universe.uuid,
            model_id=model_id,
            parameters=transition.get("transition_info", {}),
            initial_state=initial_state,
            energy_field_info={
                "rho_radiation": initial_state.get("rho_radiation", 0.0),
                "rho_matter": initial_state.get("rho_matter", 0.0),
                "equation_of_state": initial_state.get("equation_of_state", "radiation"),
            },
            creation_time=transition["t"],
            validity_metadata={"valid": True, "transition": info},
            theory_classification=transition["theory_classification"],
        )

        # Create the new universe in the multiverse.
        new_universe = multiverse.create_universe(
            name=f"universe-from-{model_id}-{transition['t']:.3g}",
            seed=parent_universe.seed + 1,
            cosmology=cosmology,
            model_settings={"origin": model_id, "origin_time": transition["t"]},
            origin_event=origin_event,
        )

        # Get the live instance and set its state from the origin.
        inst = multiverse.get_instance(new_universe.uuid)
        # Override the initial scale factor and time from the origin event.
        a0 = initial_state.get("a", cosmology["a_start"])
        inst._state = inst.flrw.state_at(a0, t=transition["t"])
        inst.clocks.cosmic.t = transition["t"]
        inst.clocks.scale_factor.t = a0
        inst.status = "running"

        record = UniverseOriginRecord(
            event_id=origin_event.event_id,
            parent_universe_id=parent_universe.uuid,
            new_universe_id=new_universe.uuid,
            model_id=model_id,
            theory_classification=transition["theory_classification"],
            parameters=transition.get("transition_info", {}),
            initial_state=initial_state,
            energy_field_info=origin_event.energy_field_info,
            creation_time=transition["t"],
            validity_metadata={"valid": True},
            transition_info=info,
        )
        self.origin_history.append(record)
        return record

    def list_origins(self) -> List[Dict]:
        return [r.to_dict() for r in self.origin_history]
