"""Planetary ecology integration (Sprint 8, Research Program C8/D8).

Connects the tested individual-agent architecture (ECS-IW controller) to
existing habitats, chemistry, and evolutionary populations.

An agent operates in a habitat patch:
- Senses chemical resources through its sensory interface (only local patch
  data, never the full habitat database)
- Acts on the habitat (consumes nutrients, produces waste)
- Pays metabolic costs from its energy reserve
- Can die from starvation (extinction remains possible)
- Is part of the evolutionary population with heritable genome

This does NOT claim the agent is conscious -- it measures behavioral
capabilities in a chemically explicit environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from ..contracts import (
    AgentState, SensoryObservation, AgentAction,
)
from ..workspace.model import WorkspaceAgentController
from ..self_model.model import make_default_self_model, SelfModel
from ..environment.gridworld import GridWorldConfig
from ...habitat import Habitat, Patch, make_default_network
from ...chemistry.network import Network
from ...planet.model import Planet, PlanetSpec


@dataclass
class PlanetaryAgentConfig:
    """Configuration for a planetary agent."""
    agent_id: str
    patch_index: int = 0
    initial_energy: float = 5.0
    seed: int = 0
    controller_type: str = "ecs_iw"  # "ecs_iw" | "recurrent" | "reactive"
    dt_gyr: float = 1e-6  # biological step in Gigayears


class PlanetaryAgent:
    """An individual agent embedded in a planetary habitat patch.

    The agent senses only its local patch's chemical resources (through its
    modeled sensory interface) and acts on the habitat by consuming nutrients
    and producing waste. It pays metabolic costs and can starve to death.

    The agent is coupled to the habitat: its actions change patch chemistry,
    and patch chemistry affects its sensory input and energy gain.
    """

    def __init__(self, config: PlanetaryAgentConfig,
                 habitat: Habitat,
                 controller: Optional[object] = None):
        self.cfg = config
        self.habitat = habitat
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.agent_id = config.agent_id
        self.patch_index = config.patch_index
        self.t_gyr = habitat.t_gyr
        self.alive = True
        self.n_steps = 0
        self.lifetime_reward = 0.0

        # Physiological state
        self.state = AgentState(
            energy=config.initial_energy,
            nutrients=1.0,
            stress=0.0,
            age=0.0,
            alive=True,
            timestamp=habitat.t_gyr,
        )

        # Controller
        if controller is not None:
            self.controller = controller
        else:
            self.controller = self._make_default_controller(config.seed)

        # Self-model (always present for ECS-IW)
        self.self_model = make_default_self_model()

        # Register with habitat patch
        self._link_to_patch()

    def _make_default_controller(self, seed: int) -> object:
        """Create the default ECS-IW controller."""
        from ..workspace.model import WorkspaceAgentController
        return WorkspaceAgentController(
            agent_id=self.agent_id, seed=seed,
            workspace_capacity=1, lr=0.01,
        )

    def _link_to_patch(self) -> None:
        """Link agent to its habitat patch."""
        patches = self.habitat.patches
        if self.patch_index >= len(patches):
            self.patch_index = len(patches) - 1
        self.patch = patches[self.patch_index]

    # ------------------------------------------------------------------
    # Sensory interface: agent sees only local patch chemistry
    # ------------------------------------------------------------------

    def observe(self) -> SensoryObservation:
        """Produce a sensory observation from the local patch.

        The agent sees only its patch's chemical concentrations, temperature,
        pH, and its own internal state. It cannot read the full habitat.
        """
        p = self.patch
        species = self.habitat.species_names
        conc = p.c

        # Map chemical concentrations to sensory channels
        nearby_resources = 0.0
        nearby_hazards = 0.0
        if "N" in species:
            idx = species.index("N")
            nearby_resources = float(np.clip(conc[idx] * p.volume_L, 0.0, 1.0))
        if "Wst" in species:
            idx = species.index("Wst")
            nearby_hazards = float(np.clip(conc[idx] * p.volume_L * 0.1, 0.0, 1.0))

        obs = SensoryObservation(
            time=self.t_gyr,
            position=(float(self.patch.index), 0.0),
            energy_level=float(np.clip(self.state.energy / 10.0, 0.0, 1.0)),
            nutrient_level=self.state.nutrients,
            local_temperature=float(p.temperature_K),
            local_light=1.0,
            nearby_resources=nearby_resources,
            nearby_hazards=nearby_hazards,
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
            noise_seed=None,
            provenance="planetary habitat sensory interface",
        )
        return obs

    # ------------------------------------------------------------------
    # Action interface: agent acts on the habitat
    # ------------------------------------------------------------------

    def act(self, action: AgentAction) -> Dict[str, float]:
        """Apply an action to the habitat patch.

        Actions:
        - "forage": consume nutrient M from the patch, gain energy
        - "rest": recover energy (pay metabolic cost only)
        - "move": move to a different patch (cost energy)
        - "wait": do nothing (pay metabolic cost only)
        """
        if not self.alive:
            return {"reward": 0.0, "success": False, "reason": "agent dead"}

        p = self.patch
        reward = 0.0
        success = True

        # Base metabolic cost
        metabolic_cost = 0.01 * self.cfg.dt_gyr
        self.state.energy -= metabolic_cost

        if action.action_type == "forage":
            # Consume monomer M from patch chemistry
            species = self.habitat.species_names
            if "M" in species:
                idx = species.index("M")
                available = float(p.c[idx] * p.volume_L)
                take = min(available, 0.1 * self.cfg.dt_gyr)
                if take > 0:
                    p.c[idx] -= take / p.volume_L
                    p.nutrient_mol = max(0.0, p.nutrient_mol - take * 0.1)
                    self.state.energy += 2.0 * take
                    self.state.nutrients += take
                    reward += take
            # Produce waste
            if "Wst" in species:
                idx = species.index("Wst")
                p.c[idx] += 0.05 * self.cfg.dt_gyr / p.volume_L

        elif action.action_type == "rest":
            gain = 0.1 * self.cfg.dt_gyr
            self.state.energy = min(self.state.energy + gain, 10.0)
            reward = 0.0

        elif action.action_type == "move":
            cost = 0.2 * self.cfg.dt_gyr
            self.state.energy -= cost
            reward = -cost

        elif action.action_type == "wait":
            reward = 0.0

        else:
            success = False
            reward = -0.1

        # Hazard from patch chemistry
        if p.temperature_K > 350.0 or not p.liquid_water:
            self.state.stress = min(1.0, self.state.stress + 0.01 * self.cfg.dt_gyr)
            self.state.energy -= 0.05 * self.cfg.dt_gyr
            reward -= 0.1

        # Death from energy depletion
        if self.state.energy <= 0:
            self.state.energy = 0.0
            self.state.alive = False
            self.alive = False
            reward = -1.0

        self.state.timestamp = self.t_gyr
        self.n_steps += 1
        self.lifetime_reward += reward

        return {"reward": float(reward), "success": bool(success),
                "energy": float(self.state.energy), "stress": float(self.state.stress),
                "alive": bool(self.alive)}

    # ------------------------------------------------------------------
    # Step: observe -> act -> learn -> habitat step
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """One agent-environment interaction step."""
        obs = self.observe()
        action = self.controller.select_action(obs)
        result = self.act(action)
        next_obs = self.observe()
        reward = float(result.get("reward", 0.0))
        self.controller.learn(obs, action, reward, next_obs)

        # Update self-model
        actual_change = np.array([
            next_obs.nearby_resources - obs.nearby_resources,
            next_obs.nearby_hazards - obs.nearby_hazards,
        ])
        predicted = self.self_model.predict(action.action_type, action.parameters)
        self.self_model.update(
            action.action_type, action.parameters,
            predicted[:2], actual_change[:2]
        )

        # Advance habitat chemistry
        self.habitat.step(self.cfg.dt_gyr)
        self.t_gyr = self.habitat.t_gyr
        self.state.age += self.cfg.dt_gyr

        # Update patch reference (habitat step may change patch properties)
        self._link_to_patch()

        return {
            "t_gyr": self.t_gyr,
            "reward": reward,
            "action": action.action_type,
            "alive": self.alive,
            "energy": self.state.energy,
            "stress": self.state.stress,
            "patch_food": float(self.patch.c[self.habitat.species_names.index("N")]
                                * self.patch.volume_L
                                if "N" in self.habitat.species_names else 0.0),
        }

    def is_alive(self) -> bool:
        return self.alive and self.state.alive

    def checkpoint(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "patch_index": self.patch_index,
            "t_gyr": self.t_gyr,
            "alive": self.alive,
            "n_steps": self.n_steps,
            "lifetime_reward": self.lifetime_reward,
            "state": self.state.to_dict(),
            "controller": self.controller.checkpoint(),
            "self_model": self.self_model.checkpoint(),
        }

    @classmethod
    def restore(cls, data: dict, habitat: Habitat,
                controller: Optional[object] = None) -> "PlanetaryAgent":
        cfg = PlanetaryAgentConfig(
            agent_id=data["agent_id"],
            patch_index=int(data.get("patch_index", 0)),
            initial_energy=float(data.get("initial_energy", 5.0)),
            seed=int(data.get("seed", 0)),
        )
        agent = cls.__new__(cls)
        agent.cfg = cfg
        agent.habitat = habitat
        agent.created_at = data.get("created_at", "")
        agent.agent_id = data["agent_id"]
        agent.patch_index = int(data.get("patch_index", 0))
        agent.t_gyr = float(data["t_gyr"])
        agent.alive = bool(data["alive"])
        agent.n_steps = int(data.get("n_steps", 0))
        agent.lifetime_reward = float(data.get("lifetime_reward", 0.0))
        agent.state = AgentState(**data["state"])
        agent.controller = controller
        agent.self_model = SelfModel.restore(data.get("self_model", {}))
        agent._link_to_patch()
        return agent


def make_planet_habitat(seed: int = 0, n_patches: int = 4,
                        has_water: bool = True) -> Habitat:
    """Create a default planet habitat for agent integration."""
    spec = PlanetSpec(mass_mearth=1.0, a_au=1.0, ocean_fraction=0.71,
                      has_surface_water=has_water, tectonics="mobile", seed=seed)
    planet = Planet(spec, t_gyr=4.5)
    network = make_default_network()
    h = Habitat(planet, network, n_patches=n_patches, seed=seed)
    for p in h.patches:
        p.liquid_water = has_water
    return h


def run_planetary_agent_experiment(seed: int = 0, n_steps: int = 50,
                                   controller_type: str = "ecs_iw") -> Dict:
    """Run a planetary agent experiment.

    One ECS-IW agent operates in a habitat patch, consuming resources and
    adapting to changing chemistry. Returns metrics on survival, reward,
    and self-model calibration.
    """
    habitat = make_planet_habitat(seed=seed, n_patches=4, has_water=True)

    # Seed replicators in the habitat so chemistry is active
    habitat.seed_replicator("ACGU", count=50, cell_index=0)
    habitat.replicators.pool = 10**6
    habitat.protocells.solute_rate = 0.0
    habitat.protocells.Pf = 0.0
    habitat.protocells.lyse = 1.0
    habitat.protocells.cells[0].N = 10**6

    config = PlanetaryAgentConfig(
        agent_id="agent-1", patch_index=0, seed=seed,
        controller_type=controller_type,
    )
    agent = PlanetaryAgent(config, habitat)

    history = []
    for _ in range(n_steps):
        result = agent.step()
        history.append(result)
        if not agent.is_alive():
            break

    alive = agent.is_alive()
    return {
        "seed": seed, "n_steps": len(history),
        "alive": alive,
        "final_energy": agent.state.energy,
        "final_stress": agent.state.stress,
        "lifetime_reward": agent.lifetime_reward,
        "self_model_calibration": agent.self_model.calibration_error,
        "n_self_model_updates": agent.self_model.n_updates,
        "history": history,
    }
