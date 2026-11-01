"""Minimal embodied agent environment (Sprint 3, Research Program D1-D2).

A persistent 2D grid-world environment with:
- Resources (food) distributed across patches that regenerate over time
- Hazards that damage agents
- Agents with limited sensory range (they only see their local patch)
- Physiological state: energy, nutrients, stress
- Explicit actions: move, forage, rest, wait
- No agent can read the simulator database directly

This is a minimal testbed for embodied cognition research. Agents operate
under changing environmental conditions, limited information, and constrained
resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..contracts import (
    AgentIdentity, AgentState, SensoryObservation, AgentAction,
    Environment, AgentController, PhysicalQuantity, now_iso,
)


@dataclass
class PatchState:
    """One patch in the grid world."""
    index: Tuple[int, int]
    food: float            # available food units (0..1)
    hazard: float          # hazard level (0..1)
    temperature: float     # K
    light: float           # 0..1 ambient light
    regeneration_rate: float  # food regen per step


@dataclass
class GridWorldConfig:
    """Configuration for GridWorld."""
    width: int = 8
    height: int = 8
    seed: int = 0
    initial_food: float = 0.5
    food_regeneration: float = 0.01
    hazard_probability: float = 0.1
    temperature_base: float = 288.0
    dt: float = 0.1
    max_energy: float = 10.0
    move_cost: float = 0.1
    forage_cost: float = 0.05
    rest_gain: float = 0.2
    sensory_range: int = 1   # patches in each direction the agent can sense


class GridWorld:
    """Persistent 2D grid world environment.

    Agents occupy patches, sense only their local neighborhood, and act through
    explicit actions. Food is consumed when foraged and regenerates slowly.
    Hazards inflict stress. Temperature affects metabolic cost.

    The environment owns the authoritative state. Agents never read it directly.
    """

    def __init__(self, config: Optional[GridWorldConfig] = None):
        self.cfg = config or GridWorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.env_id = f"gridworld-{self.cfg.seed}"
        self.env_type = "gridworld"
        self.created_at = now_iso()
        self.paused = False
        self.t = 0.0
        self.step_count = 0

        # Build grid
        self.patches: Dict[Tuple[int, int], PatchState] = {}
        for i in range(self.cfg.width):
            for j in range(self.cfg.height):
                hazard = 1.0 if self.rng.random() < self.cfg.hazard_probability else 0.0
                self.patches[(i, j)] = PatchState(
                    index=(i, j),
                    food=self.cfg.initial_food,
                    hazard=hazard,
                    temperature=self.cfg.temperature_base,
                    light=1.0,
                    regeneration_rate=self.cfg.food_regeneration,
                )

        # Agent states (agent_id -> AgentState) and positions
        self._agent_states: Dict[str, AgentState] = {}
        self._agent_positions: Dict[str, Tuple[int, int]] = {}

    def register_agent(self, agent_id: str, position: Tuple[int, int],
                       initial_energy: float = 5.0) -> None:
        """Register a new agent in the world."""
        x, y = int(position[0]) % self.cfg.width, int(position[1]) % self.cfg.height
        self._agent_positions[agent_id] = (x, y)
        self._agent_states[agent_id] = AgentState(
            energy=initial_energy, nutrients=1.0, stress=0.0,
            age=0.0, alive=True, timestamp=self.t,
        )

    def reset(self) -> SensoryObservation:
        """Reset environment to initial state. Returns observation for first agent."""
        self.t = 0.0
        self.step_count = 0
        for p in self.patches.values():
            p.food = self.cfg.initial_food
        # Return observation for any registered agent (or empty if none)
        if self._agent_states:
            aid = next(iter(self._agent_states))
            return self.observe(aid)
        return SensoryObservation(time=0.0, position=(0, 0), energy_level=0.0,
                                  nutrient_level=0.0, local_temperature=288.0,
                                  local_light=1.0, nearby_resources=0.0,
                                  nearby_hazards=0.0, nearby_agents=0)

    def observe(self, agent_id: str) -> SensoryObservation:
        """Produce a sensory observation for an agent.

        The agent only senses its local patch and immediate neighbors within
        sensory_range. It cannot read the full grid state.
        """
        if agent_id not in self._agent_states:
            raise ValueError(f"unknown agent {agent_id!r}")
        state = self._agent_states[agent_id]
        pos = self._agent_positions[agent_id]
        x, y = pos

        # Gather sensory data from local neighborhood
        r = self.cfg.sensory_range
        foods, hazards, temps, lights, n_agents = [], [], [], [], 0
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = (x + dx) % self.cfg.width, (y + dy) % self.cfg.height
                p = self.patches[(nx, ny)]
                foods.append(p.food)
                hazards.append(p.hazard)
                temps.append(p.temperature)
                lights.append(p.light)
        for aid, apos in self._agent_positions.items():
            if aid != agent_id and apos == pos:
                n_agents += 1

        # Sensory noise: add small Gaussian noise proportional to sensory range
        noise = 0.01 * self.rng.standard_normal()
        nearby_resources = float(np.clip(np.mean(foods) + noise, 0.0, 1.0))
        nearby_hazards = float(np.clip(max(hazards) + noise, 0.0, 1.0))

        obs = SensoryObservation(
            time=self.t,
            position=pos,
            energy_level=float(np.clip(state.energy / self.cfg.max_energy, 0.0, 1.0)),
            nutrient_level=state.nutrients,
            local_temperature=float(np.mean(temps)),
            local_light=float(np.mean(lights)),
            nearby_resources=nearby_resources,
            nearby_hazards=nearby_hazards,
            nearby_agents=n_agents,
            social_signals=[],
            extra={"patch_food": float(self.patches[pos].food),
                   "patch_hazard": float(self.patches[pos].hazard)},
            noise_seed=int(self.rng.integers(0, 2**31)),
            provenance="gridworld sensory interface",
        )
        return obs

    def act(self, agent_id: str, action: AgentAction) -> Dict[str, Any]:
        """Apply an agent's action to the environment.

        Returns a dict with 'reward', 'success', and any side effects.
        The action is interpreted through a physical model: movement costs
        energy, foraging consumes food and gains energy, resting recovers.
        """
        if agent_id not in self._agent_states:
            raise ValueError(f"unknown agent {agent_id!r}")
        state = self._agent_states[agent_id]
        if not state.alive:
            return {"reward": 0.0, "success": False, "reason": "agent not alive"}

        reward = 0.0
        success = True
        pos = self._agent_positions[agent_id]
        patch = self.patches[pos]

        # Energy cost for existing this step
        metabolic_cost = 0.01 * self.cfg.dt
        state.energy -= metabolic_cost

        if action.action_type == "move":
            # Parse direction from parameters
            dx = int(action.parameters.get("dx", 0))
            dy = int(action.parameters.get("dy", 0))
            cost = self.cfg.move_cost * self.cfg.dt
            state.energy -= cost
            if state.energy > 0:
                new_x = (pos[0] + dx) % self.cfg.width
                new_y = (pos[1] + dy) % self.cfg.height
                self._agent_positions[agent_id] = (new_x, new_y)
                reward = -cost  # small negative reward for movement
        elif action.action_type == "forage":
            cost = self.cfg.forage_cost * self.cfg.dt
            state.energy -= cost
            food_available = patch.food
            take = min(food_available, 0.5 * self.cfg.dt)
            patch.food -= take
            state.energy += 2.0 * take  # food -> energy conversion
            state.nutrients += take
            reward = take
        elif action.action_type == "rest":
            gain = self.cfg.rest_gain * self.cfg.dt
            state.energy = min(state.energy + gain, self.cfg.max_energy)
            reward = 0.0
        elif action.action_type == "wait":
            reward = 0.0
        else:
            success = False
            reward = -0.1

        # Hazard damage
        if patch.hazard > 0:
            state.stress = min(1.0, state.stress + 0.05 * self.cfg.dt)
            state.energy -= 0.05 * self.cfg.dt
            reward -= 0.1

        # Death from energy depletion
        if state.energy <= 0:
            state.alive = False
            state.energy = 0.0
            reward = -1.0

        state.timestamp = self.t
        return {"reward": float(reward), "success": bool(success),
                "energy": float(state.energy), "stress": float(state.stress),
                "alive": bool(state.alive)}

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance the environment by dt (regeneration, temperature drift)."""
        dt = dt if dt is not None else self.cfg.dt
        self.t += dt
        self.step_count += 1

        # Food regeneration
        for p in self.patches.values():
            p.food = min(1.0, p.food + p.regeneration_rate * dt)

        # Slow temperature drift (seasonal)
        for p in self.patches.values():
            p.temperature += 0.001 * dt * (1.0 if self.t % 100 < 50 else -1.0)

        return {"t": self.t, "step": self.step_count}

    def checkpoint(self) -> Dict[str, Any]:
        """Serialize environment state for checkpoint/restore."""
        return {
            "env_id": self.env_id,
            "t": self.t,
            "step_count": self.step_count,
            "patches": {f"{k[0]},{k[1]}": {
                "food": p.food, "hazard": p.hazard,
                "temperature": p.temperature, "light": p.light,
            } for k, p in self.patches.items()},
            "agent_states": {aid: s.to_dict() for aid, s in self._agent_states.items()},
            "agent_positions": self._agent_positions,
        }

    def restore(self, data: Dict[str, Any]) -> None:
        """Restore environment state from checkpoint."""
        self.t = float(data["t"])
        self.step_count = int(data["step_count"])
        for key, pd in data.get("patches", {}).items():
            i, j = key.split(",")
            p = self.patches[(int(i), int(j))]
            p.food = float(pd["food"])
            p.hazard = float(pd["hazard"])
            p.temperature = float(pd["temperature"])
            p.light = float(pd["light"])
        self._agent_states = {}
        for aid, sd in data.get("agent_states", {}).items():
            self._agent_states[aid] = AgentState(
                energy=float(sd["energy"]), nutrients=float(sd["nutrients"]),
                stress=float(sd["stress"]), age=float(sd["age"]),
                alive=bool(sd["alive"]), timestamp=float(sd.get("timestamp", self.t)),
            )
        self._agent_positions = {}
        for aid, pos in data.get("agent_positions", {}).items():
            self._agent_positions[aid] = (int(pos[0]), int(pos[1]))


# ---------------------------------------------------------------------------
# Simple agent controllers (baseline implementations)
# ---------------------------------------------------------------------------

class ReactiveController:
    """A reactive (feedforward) controller.

    Selects actions based only on the current observation, with no memory
    or internal model. This is the baseline for comparison.
    """

    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.controller_type = "reactive"
        self.rng = np.random.default_rng(seed)
        self.actions_taken = 0

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        """Choose action based only on current sensory input."""
        self.actions_taken += 1
        if observation.energy_level < 0.3:
            if observation.nearby_resources > 0.5:
                return AgentAction("forage", timestamp=observation.time)
            return AgentAction("move", {"dx": int(self.rng.integers(-1, 2)),
                                        "dy": int(self.rng.integers(-1, 2))},
                               timestamp=observation.time)
        if observation.nearby_hazards > 0.5:
            return AgentAction("move", {"dx": int(self.rng.integers(-1, 2)),
                                        "dy": int(self.rng.integers(-1, 2))},
                               timestamp=observation.time)
        if observation.nearby_resources > 0.5:
            return AgentAction("forage", timestamp=observation.time)
        return AgentAction("rest", timestamp=observation.time)

    def learn(self, observation: SensoryObservation, action: AgentAction,
              reward: float, next_observation: SensoryObservation) -> None:
        """No learning for reactive controller."""
        pass

    def checkpoint(self) -> Dict[str, Any]:
        return {"controller_type": "reactive", "actions_taken": self.actions_taken}

    def restore(self, data: Dict[str, Any]) -> None:
        self.actions_taken = int(data.get("actions_taken", 0))


class RecurrentWorldModelController:
    """A recurrent controller with a learned world model.

    Maintains an internal hidden state and a simple predictive model of how
    actions affect observations. Uses the prediction to select actions that
    maximize expected future reward.
    """

    def __init__(self, agent_id: str, seed: int = 0,
                 hidden_dim: int = 8, lr: float = 0.01):
        self.agent_id = agent_id
        self.controller_type = "recurrent_world_model"
        self.rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hidden = np.zeros(hidden_dim)
        self.actions_taken = 0
        # Simple linear world model: hidden' = tanh(W_h @ hidden + W_a @ action_emb + W_o @ obs_emb)
        # Predicts next observation features from current hidden + action
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_a = self.rng.standard_normal((hidden_dim, 4))  # 4 action types
        self.W_o = self.rng.standard_normal((hidden_dim, 6))  # 6 obs features
        self.W_pred = self.rng.standard_normal((6, hidden_dim)) * 0.1
        self.prediction_error_history: List[float] = []

    def _obs_vector(self, obs: SensoryObservation) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level / 10.0,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def _action_vector(self, action: AgentAction) -> np.ndarray:
        v = np.zeros(4)
        at = action.action_type
        if at == "move": v[0] = 1.0
        elif at == "forage": v[1] = 1.0
        elif at == "rest": v[2] = 1.0
        else: v[3] = 1.0
        return v

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        """Select action using the internal world model to predict outcomes."""
        self.actions_taken += 1
        obs_vec = self._obs_vector(observation)

        # Update hidden state (recurrent dynamics)
        # Use current action candidate - try all actions, predict outcomes
        best_action = "wait"
        best_value = -float("inf")
        best_action_vec = np.zeros(4)

        for at, idx in (("move", 0), ("forage", 1), ("rest", 2), ("wait", 3)):
            av = np.zeros(4); av[idx] = 1.0
            hidden_next = np.tanh(self.W_h @ self.hidden + self.W_a @ av + self.W_o @ obs_vec)
            pred_obs = self.W_pred @ hidden_next
            # Simple value: prefer predicted energy gain, avoid hazard
            predicted_energy = pred_obs[0]
            predicted_hazard = pred_obs[3]
            value = predicted_energy - 0.5 * predicted_hazard
            if value > best_value:
                best_value = value
                best_action = at
                best_action_vec = av
                self.hidden = hidden_next  # commit to chosen action's hidden state

        # Build action parameters
        params = {}
        if best_action == "move":
            params = {"dx": int(self.rng.integers(-1, 2)),
                      "dy": int(self.rng.integers(-1, 2))}

        action = AgentAction(best_action, params, timestamp=observation.time,
                             confidence=0.5)
        return action

    def learn(self, observation: SensoryObservation, action: AgentAction,
              reward: float, next_observation: SensoryObservation) -> None:
        """Learn world model parameters from experience.

        Simple gradient step on prediction error: predict next observation
        from current hidden + action.
        """
        obs_vec = self._obs_vector(observation)
        next_obs_vec = self._obs_vector(next_observation)
        action_vec = self._action_vector(action)

        # Predict next hidden state
        hidden_pred = np.tanh(self.W_h @ self.hidden + self.W_a @ action_vec + self.W_o @ obs_vec)
        pred_obs = self.W_pred @ hidden_pred

        # Prediction error
        error = pred_obs - next_obs_vec
        self.prediction_error_history.append(float(np.mean(np.abs(error))))
        if len(self.prediction_error_history) > 1000:
            self.prediction_error_history = self.prediction_error_history[-1000:]

        # Simple gradient update (stochastic gradient descent)
        self.W_pred -= self.lr * np.outer(error, hidden_pred)
        self.W_o -= self.lr * 0.01 * np.outer(hidden_pred, error)
        self.W_a -= self.lr * 0.01 * np.outer(hidden_pred, action_vec)

        # Update hidden to the actual next state (teacher forcing)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_a @ action_vec + self.W_o @ next_obs_vec)

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "controller_type": "recurrent_world_model",
            "hidden": self.hidden.tolist(),
            "W_h": self.W_h.tolist(), "W_a": self.W_a.tolist(),
            "W_o": self.W_o.tolist(), "W_pred": self.W_pred.tolist(),
            "actions_taken": self.actions_taken,
            "prediction_error_history": self.prediction_error_history[-100:],
        }

    def restore(self, data: Dict[str, Any]) -> None:
        self.hidden = np.array(data.get("hidden", np.zeros(self.hidden_dim)))
        self.W_h = np.array(data.get("W_h", self.W_h))
        self.W_a = np.array(data.get("W_a", self.W_a))
        self.W_o = np.array(data.get("W_o", self.W_o))
        self.W_pred = np.array(data.get("W_pred", self.W_pred))
        self.actions_taken = int(data.get("actions_taken", 0))
        self.prediction_error_history = list(data.get("prediction_error_history", []))
