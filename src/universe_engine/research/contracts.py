"""Shared agent/environment contracts for the Project Genesis research program.

These contracts define the boundary between agents (controllers) and the
environments they perceive and act upon. The design follows two principles:

1. An agent sees ONLY what its sensors provide. It never reads the
   underlying simulator database directly.
2. Every interaction is versioned and provenance-tracked: inputs, outputs,
   random seeds, and model assumptions are all recorded for reproducibility.

This module is the Sprint 1 foundation for Research Programs B, C, D, E,
and F.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import hashlib
import json
import time
import uuid

import numpy as np


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy arrays and other non-JSON types to JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Units and physical quantities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalQuantity:
    """A physical quantity with explicit units and provenance.

    All quantities flowing across the agent/environment boundary carry units
    so that camera/viewer transformations can never alter scientific results.
    """
    value: float
    units: str
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent identity and state
# ---------------------------------------------------------------------------

@dataclass
class AgentIdentity:
    """Persistent identity for one agent. Survives across checkpoints."""
    agent_id: str
    agent_type: str        # "reactive" | "recurrent_world_model" | "ecs_iw" | ...
    created_at: str
    parent_id: Optional[str] = None
    generation: int = 0
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """Mutable physiological/resource state of one agent.

    This is the agent's internal body/homeostasis state. The agent observes
    it only through its own sensors; the environment mutates it through
    modeled physical couplings.
    """
    energy: float = 0.0          # stored metabolic energy (units)
    nutrients: float = 0.0       # stored material units
    stress: float = 0.0          # accumulated damage (0..1)
    age: float = 0.0             # time since birth (sim time units)
    alive: bool = True
    timestamp: float = 0.0       # last update time

    def clone(self) -> "AgentState":
        return AgentState(
            energy=self.energy, nutrients=self.nutrients, stress=self.stress,
            age=self.age, alive=self.alive, timestamp=self.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sensory interface: what an agent can perceive
# ---------------------------------------------------------------------------

@dataclass
class SensoryObservation:
    """One timestep of sensory input for an agent.

    Fields are the agent's modeled sensory channels. Additional arbitrary
    channels can be carried in `extra`. The agent cannot read fields not
    present here.
    """
    time: float
    position: Tuple[float, ...]          # agent's sensed location
    energy_level: float                  # 0..1 sensed internal energy
    nutrient_level: float                # 0..1 sensed internal nutrients
    local_temperature: float             # K, sensed ambient temperature
    local_light: float                   # 0..1 sensed ambient light
    nearby_resources: float              # 0..1 sensed resource concentration
    nearby_hazards: float                # 0..1 sensed threat level
    nearby_agents: int                   # count of other agents sensed
    social_signals: List[str] = field(default_factory=list)   # signals heard
    extra: Dict[str, Any] = field(default_factory=dict)        # extensible channels
    noise_seed: Optional[int] = None     # RNG seed used for sensory noise
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Action interface: what an agent can do
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """One action selected by an agent's controller.

    The action schema is fixed per environment so that agents cannot invent
    physical capabilities they do not have.
    """
    action_type: str                     # "move" | "forage" | "rest" | "communicate" | "wait" | "experiment" | ...
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    confidence: Optional[float] = None   # agent's own confidence in this action (0..1)
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Environment contract
# ---------------------------------------------------------------------------

class Environment(Protocol):
    """A persistent environment that agents inhabit.

    An environment owns its authoritative state, exposes observations through
    a sensory interface, applies actions through a modeled physical coupling,
    and tracks provenance. Camera/viewer transformations never alter state.
    """

    env_id: str
    env_type: str
    seed: int
    created_at: str
    paused: bool

    def reset(self) -> SensoryObservation: ...
    def observe(self, agent_id: str) -> SensoryObservation: ...
    def act(self, agent_id: str, action: AgentAction) -> Dict[str, Any]: ...
    def step(self, dt: float) -> Dict[str, Any]: ...
    def checkpoint(self) -> Dict[str, Any]: ...
    def restore(self, data: Dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Agent controller contract
# ---------------------------------------------------------------------------

class AgentController(Protocol):
    """A controller that selects actions based on sensory observations.

    The controller never reads the simulator database. It receives only
    SensoryObservation objects and produces AgentAction objects.
    """

    agent_id: str
    controller_type: str

    def select_action(self, observation: SensoryObservation) -> AgentAction: ...
    def learn(self, observation: SensoryObservation, action: AgentAction,
              reward: float, next_observation: SensoryObservation) -> None: ...
    def checkpoint(self) -> Dict[str, Any]: ...
    def restore(self, data: Dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Experiment manifest
# ---------------------------------------------------------------------------

@dataclass
class ExperimentManifest:
    """A complete, reproducible description of one research experiment.

    Every experiment in the research program carries this manifest so that
    results can be independently reproduced. It records the question,
    assumptions, candidate mechanisms, baseline, seeds, budget, and
    validation plan.
    """
    experiment_id: str
    name: str
    created_at: str
    research_program: str            # "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H"
    question: str
    model_assumptions: Dict[str, Any]
    candidate_mechanisms: List[str]
    baseline_mechanism: str
    parameter_space: Dict[str, Any]
    random_seeds: List[int]
    computational_budget: Dict[str, Any]
    predicted_observables: List[str]
    numerical_methods: Dict[str, Any]
    statistical_methods: Dict[str, Any]
    held_out_validation: Dict[str, Any]
    uncertainty_analysis: Dict[str, Any]
    reproducibility_manifest: Dict[str, Any]
    status: str = "created"          # created | running | paused | completed | failed
    results: Optional[Dict[str, Any]] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentManifest":
        return cls(**data)

    def config_hash(self) -> str:
        """Content hash of the manifest (minus results) for reproducibility."""
        blob = json.dumps(_jsonify(self.to_dict()), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Experiment result record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Measured outcome of one experiment run.

    Separates raw measurements from interpretation. Results are stored
    alongside the manifest for reproducibility.
    """
    experiment_id: str
    run_id: str
    started_at: str
    finished_at: Optional[str]
    status: str                      # "running" | "completed" | "failed" | "aborted"
    seed: int
    code_revision: str               # git SHA at time of run
    config_hash: str                 # manifest content hash
    measurements: Dict[str, Any]     # raw measured values
    observables: Dict[str, float]    # scalar observables for comparison
    uncertainty: Dict[str, float]    # uncertainty estimates per observable
    passed_tolerances: bool          # whether results met declared tolerances
    notes: List[str] = field(default_factory=list)
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Checkpoint record for reproducible state
# ---------------------------------------------------------------------------

@dataclass
class ResearchCheckpoint:
    """A versioned checkpoint of research state.

    Contains enough information to deterministically resume an experiment:
    environment state, agent controller state, RNG state, manifest, and
    provenance.
    """
    checkpoint_id: str
    created_at: str
    schema: str = "research-checkpoint/v1"
    experiment_id: Optional[str] = None
    agent_id: Optional[str] = None
    env_id: Optional[str] = None
    environment_state: Dict[str, Any] = field(default_factory=dict)
    agent_state: Dict[str, Any] = field(default_factory=dict)
    controller_state: Dict[str, Any] = field(default_factory=dict)
    rng_state: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Runs one experiment manifest to completion with checkpoint/restore.

    Provides the core loop: observe -> act -> step -> learn -> record.
    Every step is logged for provenance.
    """

    def __init__(self, manifest: ExperimentManifest,
                 environment: Environment,
                 controller: AgentController,
                 seed: int = 0,
                 max_steps: int = 10_000,
                 dt: float = 0.1,
                 output_root: str = "runs/research"):
        self.manifest = manifest
        self.environment = environment
        self.controller = controller
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.output_root = output_root
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.t = 0.0
        self.status = "created"
        self.measurements: Dict[str, List[Any]] = {}
        self.log: List[Dict[str, Any]] = []
        Path(output_root).mkdir(parents=True, exist_ok=True)

    def run(self) -> ExperimentResult:
        """Execute the experiment loop until completion or budget exhaustion."""
        import pathlib
        pathlib.Path(self.output_root).mkdir(parents=True, exist_ok=True)

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        self.status = "running"
        started_at = now_iso()

        try:
            obs = self.environment.reset()
            while self.step_count < self.max_steps and self.status == "running":
                # 1. Observe
                obs = self.environment.observe(self.controller.agent_id)
                # 2. Act
                action = self.controller.select_action(obs)
                # 3. Apply action through environment
                result = self.environment.act(self.controller.agent_id, action)
                # 4. Observe next state
                next_obs = self.environment.observe(self.controller.agent_id)
                # 5. Learn (if controller supports it)
                reward = float(result.get("reward", 0.0))
                self.controller.learn(obs, action, reward, next_obs)
                # 6. Step environment
                step_result = self.environment.step(self.dt)
                self.t += self.dt
                self.step_count += 1
                # 7. Log
                self.log.append({
                    "step": self.step_count,
                    "t": self.t,
                    "observation": obs.to_dict(),
                    "action": action.to_dict(),
                    "result": result,
                    "next_observation": next_obs.to_dict(),
                    "reward": reward,
                })
                # 8. Check termination
                if step_result.get("terminated", False):
                    self.status = "completed"
                    break
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self.log.append({"error": str(exc), "step": self.step_count, "t": self.t})
            raise

        finished_at = now_iso() if self.status == "completed" else None
        return ExperimentResult(
            experiment_id=self.manifest.experiment_id,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status=self.status,
            seed=self.seed,
            code_revision="unknown",   # set by caller from git
            config_hash=self.manifest.config_hash(),
            measurements=self._collapse_measurements(),
            observables=self._compute_observables(),
            uncertainty={},
            passed_tolerances=False,
            notes=[f"ran {self.step_count} steps, t={self.t:.6g}"],
            provenance=self.manifest.provenance,
        )

    def checkpoint(self) -> ResearchCheckpoint:
        """Produce a reproducible checkpoint of current experiment state."""
        return ResearchCheckpoint(
            checkpoint_id=f"cp-{uuid.uuid4().hex[:12]}",
            created_at=now_iso(),
            experiment_id=self.manifest.experiment_id,
            agent_id=self.controller.agent_id,
            env_id=self.environment.env_id,
            environment_state=self.environment.checkpoint(),
            agent_state=asdict(self.environment.observe(self.controller.agent_id)),
            controller_state=self.controller.checkpoint(),
            rng_state=self._rng_state_dict(),
            manifest=self.manifest.to_dict(),
        )

    def restore(self, cp: ResearchCheckpoint) -> None:
        """Restore experiment state from a checkpoint."""
        self.environment.restore(cp.environment_state)
        self.controller.restore(cp.controller_state)
        self.step_count = cp.environment_state.get("step_count", 0)
        self.t = cp.environment_state.get("t", 0.0)
        self.status = "running"
        if cp.rng_state:
            self._set_rng_state(cp.rng_state)

    # -- internal helpers ---------------------------------------------------
    def _rng_state_dict(self) -> Dict[str, Any]:
        return {"bit_generator": self.rng.bit_generator.__class__.__name__,
                "state": self.rng.bit_generator.state}

    def _set_rng_state(self, state: Dict[str, Any]) -> None:
        bg = getattr(np.random, state["bit_generator"], None)
        if bg is not None:
            self.rng = bg()
            self.rng.bit_generator.state = state["state"]

    def _collapse_measurements(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, values in self.measurements.items():
            if values:
                arr = np.asarray(values, dtype=float)
                out[key] = {
                    "count": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        return out

    def _compute_observables(self) -> Dict[str, float]:
        """Scalar observables from the manifest's predicted list."""
        obs: Dict[str, float] = {}
        for name in self.manifest.predicted_observables:
            if name in self.measurements and self.measurements[name]:
                arr = np.asarray(self.measurements[name], dtype=float)
                obs[name] = float(arr[-1])  # final value
            else:
                obs[name] = 0.0
        return obs


from pathlib import Path  # noqa: E402  (imported at bottom to keep Protocol definitions first)
