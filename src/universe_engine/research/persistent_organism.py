"""Persistent organism integration (Sprint 12 Phase 5).

Connects a validated agent architecture to the existing planetary habitat
backend, producing one genuinely persistent computational organism:

- Persistent identity, body, energy/nutrient resources, developmental history
- Sensory observations come ONLY from the modeled local habitat patch
  (no privileged access to global simulator state)
- Actions change actual habitat chemistry and organism state
- Metabolic/resource costs paid for activity and computation
- Extinction (starvation) remains possible
- Checkpoint/replay across pause/resume/save/load
- Internal state exported for the Mind Inspector (Phase 6)

The best-supported candidate architecture (ECS-IW: workspace + self-model +
world model) is used by default; other controllers can be substituted for
controlled comparison. No claim of consciousness is made.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

from .contracts import (
    AgentAction, AgentState, SensoryObservation,
)
from .workspace.model import WorkspaceAgentController, GlobalWorkspace
from .self_model.model import make_default_self_model, SelfModel, BodySchema
from .environment.gridworld import ReactiveController, RecurrentWorldModelController
from .planetary_agent.experiment import (
    PlanetaryAgentConfig, make_planet_habitat,
)
from ..habitat import Habitat


# ---------------------------------------------------------------------------
# Identity and history
# ---------------------------------------------------------------------------

@dataclass
class OrganismIdentity:
    organism_id: str
    architecture: str           # "ecs_iw" | "reactive" | "recurrent" | "world_model"
    birth_time_gyr: float
    generation: int
    parent_id: Optional[str]
    provenance: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DevelopmentalEvent:
    t_gyr: float
    event: str                  # born | act | learn | reproduce | damage | checkpoint | restore
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    t_gyr: float
    observation: dict
    action: dict
    reward: float
    next_observation: dict


class EpisodicMemory:
    """Capacity-limited episodic memory.

    Stores recent experience tuples. Oldest entries are evicted when capacity
    is exceeded, modeling a bounded memory resource.
    """

    def __init__(self, capacity: int = 500):
        self.capacity = int(capacity)
        self.entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity:]

    def recent(self, n: int) -> List[MemoryEntry]:
        return list(self.entries[-n:])

    def clear(self) -> None:
        self.entries = []

    def to_dict(self) -> dict:
        """Lightweight inspector representation (bounded, lossy).

        Used by the Mind Inspector so UI payloads stay small. Not a
        lossless checkpoint: only the most recent `limit` entries survive.
        """
        return {
            "capacity": self.capacity,
            "n_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries[-50:]],
        }

    def inspector_summary(self, limit: int = 50) -> dict:
        """Lightweight inspector representation of the most recent entries.

        Mirrors `to_dict()` for backward compatibility but makes the
        inspector/lossy intent explicit. Callers that need full fidelity
        should use `checkpoint()` instead.
        """
        return {
            "capacity": self.capacity,
            "n_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries[-int(limit):]],
        }

    def checkpoint(self) -> dict:
        """Lossless checkpoint representation of all retained entries.

        Every entry currently held in memory (up to capacity) is serialized,
        so restore() can reconstruct memory exactly.
        """
        return {
            "schema": "episodic-memory/v1",
            "capacity": self.capacity,
            "n_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def restore(cls, data: dict) -> "EpisodicMemory":
        """Restore memory from a lossless checkpoint() dict.

        Maintains compatibility with legacy `to_dict()` checkpoints (which
        only carried the last 50 entries): restoring those yields only the
        entries that were serialized, which is the honest, documented
        behavior for old checkpoints.
        """
        m = cls(capacity=int(data.get("capacity", 500)))
        m.entries = [MemoryEntry(**e) for e in data.get("entries", [])]
        return m


# ---------------------------------------------------------------------------
# Persistent organism
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_controller(architecture: str, agent_id: str, seed: int) -> object:
    if architecture == "ecs_iw":
        return WorkspaceAgentController(agent_id=agent_id, seed=seed,
                                        workspace_capacity=1, lr=0.01)
    if architecture == "reactive":
        return ReactiveController(agent_id=agent_id, seed=seed)
    if architecture == "recurrent":
        return RecurrentWorldModelController(agent_id=agent_id, seed=seed, lr=0.05)
    if architecture == "world_model":
        # world-model controller with reward predictor (diagnostics)
        from .diagnostics.statistical_replication import WorldModelController
        return WorldModelController(agent_id=agent_id, seed=seed, hidden_dim=8, lr=0.05)
    raise ValueError(f"unknown architecture {architecture!r}")


class PersistentOrganism:
    """One persistent organism living in a habitat patch.

    The organism senses only its local patch (through a modeled sensory
    interface) and acts on the habitat (consuming nutrients, producing waste).
    It pays metabolic and computational costs and can starve to death.
    """

    def __init__(self, identity: OrganismIdentity, habitat: Habitat,
                 patch_index: int = 0, seed: int = 0,
                 initial_energy: float = 5.0, memory_capacity: int = 500):
        self.identity = identity
        self.habitat = habitat
        self.patch_index = int(patch_index)
        self.seed = int(seed)
        self.created_at = _now_iso()
        self.t_gyr = float(habitat.t_gyr)
        self.alive = True
        self.n_steps = 0
        self.lifetime_reward = 0.0
        self.total_cost_paid = 0.0
        self.developmental_history: List[DevelopmentalEvent] = [
            DevelopmentalEvent(t_gyr=self.t_gyr, event="born",
                               details={"patch": patch_index,
                                        "architecture": identity.architecture}),
        ]

        # Physiological state
        self.body = AgentState(
            energy=initial_energy, nutrients=1.0, stress=0.0,
            age=0.0, alive=True, timestamp=self.t_gyr,
        )

        # Memory
        self.memory = EpisodicMemory(capacity=memory_capacity)

        # Controller + self-model
        self.controller = _make_controller(identity.architecture, identity.organism_id, seed)
        self.controller_type = getattr(self.controller, "controller_type",
                                       identity.architecture)
        self.self_model = make_default_self_model()

        # Link to patch
        self._link_to_patch()

        # Record controller + self-model snapshots for inspector
        self._record("born", {})

    # ------------------------------------------------------------------
    # Patch link
    # ------------------------------------------------------------------

    def _link_to_patch(self) -> None:
        patches = self.habitat.patches
        if not patches:
            self.patch = None
            return
        if self.patch_index >= len(patches):
            self.patch_index = len(patches) - 1
        self.patch = patches[self.patch_index]

    # ------------------------------------------------------------------
    # Sensory interface (agent sees ONLY local patch + internal state)
    # ------------------------------------------------------------------

    def observe(self) -> SensoryObservation:
        """Produce the organism's sensory observation.

        The organism sees its patch's chemical concentrations, temperature,
        pH, liquid water, and its own internal state. It cannot read the full
        habitat, the planet, or other patches.
        """
        p = self.patch
        if p is None:
            return SensoryObservation(
                time=self.t_gyr, position=(float(self.patch_index), 0.0),
                energy_level=0.0, nutrient_level=0.0,
                local_temperature=288.0, local_light=0.0,
                nearby_resources=0.0, nearby_hazards=1.0,
                nearby_agents=0, social_signals=[],
                extra={"patch_index": self.patch_index, "no_patch": True},
                provenance="habitat sensory interface (no patch)",
            )

        species = self.habitat.species_names
        conc = p.c
        nearby_resources = 0.0
        nearby_hazards = 0.0
        if "N" in species:
            idx = species.index("N")
            nearby_resources = float(np.clip(conc[idx] * p.volume_L, 0.0, 1.0))
        if "Wst" in species:
            idx = species.index("Wst")
            nearby_hazards = float(np.clip(conc[idx] * p.volume_L * 0.1, 0.0, 1.0))

        hazard = 0.0
        if not p.liquid_water or p.temperature_K > 350.0:
            hazard = 1.0

        obs = SensoryObservation(
            time=self.t_gyr,
            position=(float(self.patch.index), 0.0),
            energy_level=float(np.clip(self.body.energy / 10.0, 0.0, 1.0)),
            nutrient_level=self.body.nutrients,
            local_temperature=float(p.temperature_K),
            local_light=1.0 if p.liquid_water else 0.0,
            nearby_resources=nearby_resources,
            nearby_hazards=float(max(nearby_hazards, hazard)),
            nearby_agents=0,
            social_signals=[],
            extra={
                "patch_index": self.patch.index,
                "pH": float(p.pH),
                "liquid_water": bool(p.liquid_water),
                "nutrient_mol": float(p.nutrient_mol),
                "waste_mol": float(p.waste_mol),
                "species": list(species),
                "concentrations": conc.tolist(),
            },
            provenance="habitat sensory interface",
        )
        return obs

    # ------------------------------------------------------------------
    # Action interface (organism acts on its patch)
    # ------------------------------------------------------------------

    def act(self, action: AgentAction) -> Dict[str, float]:
        """Apply an action to the organism + habitat patch.

        Returns {"reward", "success", "energy", "stress", "alive", ...}.
        """
        if not self.alive:
            return {"reward": 0.0, "success": False, "reason": "organism dead"}

        p = self.patch
        reward = 0.0
        success = True

        # Base metabolic cost (paid simply for being alive this step)
        metabolic_cost = 0.01
        self.body.energy -= metabolic_cost
        self.total_cost_paid += metabolic_cost

        if action.action_type == "forage":
            # Consume nutrient N from patch chemistry, gain energy
            species = self.habitat.species_names
            if "N" in species and p is not None:
                idx = species.index("N")
                available = float(p.c[idx] * p.volume_L)
                take = min(available, 0.5)
                if take > 0:
                    p.c[idx] -= take / p.volume_L
                    p.nutrient_mol = max(0.0, p.nutrient_mol - take * 0.1)
                    self.body.energy += 2.0 * take
                    self.body.nutrients += take
                    reward += take
            # Produce waste
            if "Wst" in species and p is not None:
                idx = species.index("Wst")
                p.c[idx] += 0.05 / p.volume_L

        elif action.action_type == "rest":
            gain = 0.1
            self.body.energy = min(self.body.energy + gain, 10.0)

        elif action.action_type == "move":
            cost = 0.2
            self.body.energy -= cost
            self.total_cost_paid += cost
            reward = -cost

        elif action.action_type == "wait":
            pass

        else:
            success = False
            reward = -0.1

        # Hazard
        if p is not None and (not p.liquid_water or p.temperature_K > 350.0):
            self.body.stress = min(1.0, self.body.stress + 0.01)
            self.body.energy -= 0.05
            self.total_cost_paid += 0.05
            reward -= 0.1

        # Death from energy depletion
        if self.body.energy <= 0:
            self.body.energy = 0.0
            self.body.alive = False
            self.alive = False
            reward = -1.0

        self.body.timestamp = self.t_gyr
        self.n_steps += 1
        self.lifetime_reward += reward

        return {"reward": float(reward), "success": bool(success),
                "energy": float(self.body.energy), "stress": float(self.body.stress),
                "alive": bool(self.alive)}

    # ------------------------------------------------------------------
    # Step: sense -> act -> learn -> habitat step -> record
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """One organism-environment interaction step."""
        if not self.alive:
            return {"t_gyr": self.t_gyr, "alive": False, "reason": "dead"}

        obs = self.observe()
        action = self.controller.select_action(obs)
        result = self.act(action)
        next_obs = self.observe()
        reward = float(result.get("reward", 0.0))

        # Learn
        self.controller.learn(obs, action, reward, next_obs)

        # Update self-model (predicted vs actual sensory consequence)
        actual_change = np.array([
            next_obs.nearby_resources - obs.nearby_resources,
            next_obs.nearby_hazards - obs.nearby_hazards,
        ])
        predicted = self.self_model.predict(action.action_type, action.parameters)
        self.self_model.update(
            action.action_type, action.parameters,
            predicted[:2], actual_change[:2]
        )

        # Record in episodic memory
        self.memory.add(MemoryEntry(
            t_gyr=self.t_gyr,
            observation=obs.to_dict(),
            action=action.to_dict(),
            reward=reward,
            next_observation=next_obs.to_dict(),
        ))

        # Advance habitat chemistry
        self.habitat.step(1e-6)  # one biological step in Gyr
        self.t_gyr = self.habitat.t_gyr
        self.body.age += 1e-6
        self._link_to_patch()

        self._record("act", {
            "action": action.action_type, "reward": reward,
            "energy": self.body.energy, "alive": self.alive,
        })

        return {
            "t_gyr": self.t_gyr,
            "reward": reward,
            "action": action.action_type,
            "alive": self.alive,
            "energy": self.body.energy,
            "stress": self.body.stress,
            "n_steps": self.n_steps,
            "lifetime_reward": self.lifetime_reward,
        }

    # ------------------------------------------------------------------
    # Recording / history
    # ------------------------------------------------------------------

    def _record(self, event: str, details: dict) -> None:
        self.developmental_history.append(
            DevelopmentalEvent(t_gyr=self.t_gyr, event=event, details=details))
        if len(self.developmental_history) > 2000:
            self.developmental_history = self.developmental_history[-2000:]

    # ------------------------------------------------------------------
    # Inspector state (Phase 6)
    # ------------------------------------------------------------------

    def inspector_state(self) -> dict:
        """Export actual model state for the Mind Inspector.

        Distinguishes true environment state (habitat patch), the organism's
        imperfect beliefs (self-model / memory), and derived metrics.
        """
        p = self.patch
        return {
            "identity": self.identity.to_dict(),
            "t_gyr": self.t_gyr,
            "alive": self.alive,
            "n_steps": self.n_steps,
            "lifetime_reward": self.lifetime_reward,
            "total_cost_paid": self.total_cost_paid,
            "body": self.body.to_dict(),
            "patch": (None if p is None else {
                "index": p.index,
                "nutrient_mol": float(p.nutrient_mol),
                "waste_mol": float(p.waste_mol),
                "temperature_K": float(p.temperature_K),
                "pH": float(p.pH),
                "liquid_water": bool(p.liquid_water),
                "concentrations": p.c.tolist(),
            }),
            "controller_type": self.controller_type,
            "self_model": {
                "calibration_error": self.self_model.calibration_error,
                "n_updates": self.self_model.n_updates,
                "sensorimotor_map": {k: v.tolist() for k, v in
                                     self.self_model.body.sensorimotor_map.items()},
            },
            "memory": self.memory.checkpoint(),
            "controller": self._controller_inspector(),
            "developmental_history": [
                {"t_gyr": e.t_gyr, "event": e.event, "details": e.details}
                for e in self.developmental_history[-50:]
            ],
        }

    def _controller_inspector(self) -> dict:
        out = {"controller_type": self.controller_type,
               "actions_taken": getattr(self.controller, "actions_taken", None)}
        if hasattr(self.controller, "workspace"):
            out["workspace"] = self.controller.workspace.checkpoint()
        if hasattr(self.controller, "self_model"):
            out["self_model_internal"] = self.controller.self_model.checkpoint()
        if hasattr(self.controller, "prediction_error_history"):
            out["recent_prediction_error"] = (
                self.controller.prediction_error_history[-10:]
                if self.controller.prediction_error_history else [])
        return out

    # ------------------------------------------------------------------
    # Checkpoint / replay (pause/resume/save/load)
    # ------------------------------------------------------------------

    def checkpoint(self) -> dict:
        return {
            "schema": "persistent-organism/v1",
            "created_at": _now_iso(),
            "identity": self.identity.to_dict(),
            "t_gyr": self.t_gyr,
            "seed": self.seed,
            "patch_index": self.patch_index,
            "alive": self.alive,
            "n_steps": self.n_steps,
            "lifetime_reward": self.lifetime_reward,
            "total_cost_paid": self.total_cost_paid,
            "body": self.body.to_dict(),
            "memory": self.memory.checkpoint(),
            "controller": self.controller.checkpoint(),
            "self_model": self.self_model.checkpoint(),
            "developmental_history": [
                {"t_gyr": e.t_gyr, "event": e.event, "details": e.details}
                for e in self.developmental_history
            ],
            "habitat": self.habitat.checkpoint(),
        }

    @classmethod
    def restore(cls, data: dict, habitat: Habitat) -> "PersistentOrganism":
        ident = OrganismIdentity(**data["identity"])
        org = cls.__new__(cls)
        org.identity = ident
        org.habitat = habitat
        org.patch_index = int(data.get("patch_index", 0))
        org.seed = int(data.get("seed", 0))
        org.created_at = data.get("created_at", "")
        org.t_gyr = float(data["t_gyr"])
        org.alive = bool(data["alive"])
        org.n_steps = int(data.get("n_steps", 0))
        org.lifetime_reward = float(data.get("lifetime_reward", 0.0))
        org.total_cost_paid = float(data.get("total_cost_paid", 0.0))
        org.body = AgentState(**data["body"])
        org.memory = EpisodicMemory.restore(data.get("memory", {}))
        org.controller = _make_controller(ident.architecture, ident.organism_id, org.seed)
        org.controller.restore(data.get("controller", {}))
        org.controller_type = getattr(org.controller, "controller_type",
                                      ident.architecture)
        org.self_model = SelfModel.restore(data.get("self_model", {}))
        org._link_to_patch()
        org.developmental_history = [
            DevelopmentalEvent(t_gyr=e["t_gyr"], event=e["event"],
                               details=e.get("details", {}))
            for e in data.get("developmental_history", [])
        ]
        # Rehydrate the shared habitat in place so all organisms restored from
        # the same world refer to ONE persistent Habitat object (no duplicated
        # independent habitats or resource inventories).
        habitat.rehydrate(data.get("habitat", {}), seed=habitat.seed)
        org.habitat = habitat
        org._link_to_patch()
        org.t_gyr = org.habitat.t_gyr
        return org


def make_default_habitat(seed: int = 0, n_patches: int = 4) -> Habitat:
    return make_planet_habitat(seed=seed, n_patches=n_patches, has_water=True)


def make_organism(architecture: str = "ecs_iw", seed: int = 0,
                  habitat: Optional[Habitat] = None,
                  patch_index: int = 0) -> PersistentOrganism:
    if habitat is None:
        habitat = make_default_habitat(seed=seed)
    identity = OrganismIdentity(
        organism_id=f"organism-{architecture}-{seed}",
        architecture=architecture,
        birth_time_gyr=float(habitat.t_gyr),
        generation=0,
        parent_id=None,
        provenance="persistent-organism/v1",
    )
    return PersistentOrganism(identity, habitat, patch_index=patch_index,
                              seed=seed, initial_energy=5.0)
