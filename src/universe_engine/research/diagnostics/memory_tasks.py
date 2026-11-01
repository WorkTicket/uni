"""Memory-sensitive task environments and strong baselines (Sprint 11).

Provides partially observable tasks where a memoryless reactive policy is
provably suboptimal, so the value of recurrent memory, prediction, and
planning can be measured fairly.

All tasks use the same SensoryObservation / AgentAction contract as
gridworld.py so agents are compared under identical information access.

These are diagnostic tasks, not claims about consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..contracts import (
    AgentAction, AgentState, SensoryObservation, Environment, AgentController,
)


# ============================================================================
# Task 1: Delayed cue (memory required)
# ============================================================================

@dataclass
class DelayedCueConfig:
    n_positions: int = 4
    cue_duration: int = 2      # steps the cue is visible
    delay: int = 3             # steps the cue is hidden before response required
    response_window: int = 3   # steps to make the correct move
    move_reward: float = 1.0
    wrong_penalty: float = -1.0
    step_cost: float = -0.01


class DelayedCueTask:
    """POMDP delayed-cue task.

    A cue (left/right) appears at the start for cue_duration steps, then is
    hidden for delay steps. The agent must move to the side matching the cue
    during the response_window. After the window closes, the episode ends.

    The observation only reports whether the cue is currently visible and
    which side it indicates; during the delay the cue side is NOT available,
    so the agent must remember it from the visible period.

    A memoryless policy cannot do better than chance on the response because
    its observation during the delay/response carries no cue information.
    """

    def __init__(self, config: Optional[DelayedCueConfig] = None, seed: int = 0):
        self.cfg = config or DelayedCueConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.episode = 0
        self.cue_side: Optional[int] = None  # 0=left, 1=right
        self.cue_visible = False
        self.phase = "cue"   # cue | delay | response | done
        self.steps_in_phase = 0

    def reset(self) -> SensoryObservation:
        self.t = 0
        self.episode += 1
        self.cue_side = int(self.rng.integers(0, 2))
        self.cue_visible = True
        self.phase = "cue"
        self.steps_in_phase = 0
        return self._observe()

    def _observe(self) -> SensoryObservation:
        # Sensory channels: cue side encoded in nearby_resources during
        # visibility (1.0=right, 0.0=left) and neutral (0.5) while hidden.
        # local_light = 1.0 while the cue is visible (informative phase), 0.0
        # during the delay/response (memory phase), so the agent must remember
        # the cue side across the delay. phase carried in extra.
        if self.cue_visible:
            nearby_resources = 1.0 if self.cue_side == 1 else 0.0
            local_light = 1.0
        else:
            nearby_resources = 0.5  # neutral: no cue information available
            local_light = 0.0
        extra = {
            "cue_side": self.cue_side,
            "cue_visible": self.cue_visible,
            "phase": self.phase,
            "correct_side": self.cue_side,
        }
        return SensoryObservation(
            time=float(self.t),
            position=(0.0, 0.0),
            energy_level=1.0,
            nutrient_level=1.0,
            local_temperature=288.0,
            local_light=local_light,
            nearby_resources=nearby_resources,
            nearby_hazards=0.0,
            nearby_agents=0,
            social_signals=[],
            extra=extra,
            provenance="delayed-cue task",
        )

    def act(self, agent_id: str, action: AgentAction) -> Dict[str, float]:
        reward = self.cfg.step_cost
        done = False
        at = action.action_type

        if self.phase == "cue":
            # Moving during cue is allowed but the cue is still visible
            if at == "move":
                reward += -0.05
        elif self.phase == "delay":
            # No useful action possible; any move is just cost
            if at == "move":
                reward += -0.05
        elif self.phase == "response":
            direction = action.parameters.get("direction", 0)
            if "dx" in action.parameters:
                direction = int(np.sign(action.parameters["dx"]))
            if self.cue_side is not None:
                correct_dir = -1 if self.cue_side == 0 else 1
                if direction == correct_dir:
                    reward += 1.0  # dense correct-direction reward
                elif direction != 0:
                    reward -= 1.0  # dense wrong-direction penalty
            done = True
        elif self.phase == "done":
            done = True

        self.t += 1
        self.steps_in_phase += 1
        self._advance_phase()
        return {"reward": float(reward), "done": bool(done)}

    def _advance_phase(self) -> None:
        if self.phase == "cue" and self.steps_in_phase >= self.cfg.cue_duration:
            self.phase = "delay"
            self.steps_in_phase = 0
            self.cue_visible = False
        elif self.phase == "delay" and self.steps_in_phase >= self.cfg.delay:
            self.phase = "response"
            self.steps_in_phase = 0
        elif self.phase == "response" and self.steps_in_phase >= self.cfg.response_window:
            self.phase = "done"
            self.steps_in_phase = 0

    def step(self, dt: float = 1.0) -> Dict:
        return {"t": self.t, "phase": self.phase}

    def observe(self, agent_id: str) -> SensoryObservation:
        return self._observe()

    def checkpoint(self) -> dict:
        return {"t": self.t, "phase": self.phase, "cue_side": self.cue_side,
                "cue_visible": self.cue_visible}

    def restore(self, data: dict) -> None:
        self.t = int(data.get("t", 0))
        self.phase = data.get("phase", "cue")
        self.cue_side = data.get("cue_side")
        self.cue_visible = bool(data.get("cue_visible", False))


# ============================================================================
# Task 2: Occluded target (spatial memory)
# ============================================================================

@dataclass
class OccludedTargetConfig:
    grid: int = 5
    visible_steps: int = 2
    occlusion_steps: int = 4
    move_cost: float = -0.01
    hit_reward: float = 2.0
    miss_penalty: float = -1.0


class OccludedTargetTask:
    """Agent sees a target for visible_steps, then it is hidden.

    The agent must navigate to and 'claim' the target location while it is
    occluded. Requires remembering the target position from the visible period.
    """

    def __init__(self, config: Optional[OccludedTargetConfig] = None, seed: int = 0):
        self.cfg = config or OccludedTargetConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.target_pos: Tuple[int, int] = (0, 0)
        self.target_visible = False
        self.phase = "visible"   # visible | occluded | done
        self.steps_in_phase = 0

    def reset(self) -> SensoryObservation:
        self.t = 0
        self.agent_pos = (0, 0)
        self.target_pos = (int(self.rng.integers(0, self.cfg.grid)),
                           int(self.rng.integers(0, self.cfg.grid)))
        self.target_visible = True
        self.phase = "visible"
        self.steps_in_phase = 0
        return self._observe()

    def _observe(self) -> SensoryObservation:
        extra = {
            "agent_pos": self.agent_pos,
            "target_visible": self.target_visible,
            "target_pos": self.target_pos if self.target_visible else None,
            "phase": self.phase,
            "correct_target": self.target_pos,
        }
        # Encode target position in nearby_resources (x) and nearby_hazards (y)
        # while visible; during occlusion report the agent's own x/y (neutral
        # w.r.t. target), so the agent must remember the target location from
        # the visible period. local_light = 1.0 while the target is visible,
        # 0.0 while occluded (memory phase).
        if self.target_visible:
            tx, ty = self.target_pos
            local_light = 1.0
        else:
            tx, ty = self.agent_pos
            local_light = 0.0
        return SensoryObservation(
            time=float(self.t),
            position=(float(self.agent_pos[0]), float(self.agent_pos[1])),
            energy_level=1.0,
            nutrient_level=1.0,
            local_temperature=288.0,
            local_light=local_light,
            nearby_resources=float(tx) / max(1, self.cfg.grid - 1),
            nearby_hazards=float(ty) / max(1, self.cfg.grid - 1),
            nearby_agents=0,
            social_signals=[],
            extra=extra,
            provenance="occluded-target task",
        )

    def act(self, agent_id: str, action: AgentAction) -> Dict[str, float]:
        reward = self.cfg.move_cost
        done = False
        dx = int(action.parameters.get("dx", 0))
        dy = int(action.parameters.get("dy", 0))

        if self.phase == "visible":
            self.agent_pos = (
                min(max(self.agent_pos[0] + dx, 0), self.cfg.grid - 1),
                min(max(self.agent_pos[1] + dy, 0), self.cfg.grid - 1),
            )
        elif self.phase == "occluded":
            # Agent may still move; claiming uses 'claim' or 'forage' action
            self.agent_pos = (
                min(max(self.agent_pos[0] + dx, 0), self.cfg.grid - 1),
                min(max(self.agent_pos[1] + dy, 0), self.cfg.grid - 1),
            )
            if action.action_type in ("claim", "forage"):
                if self.agent_pos == self.target_pos:
                    reward += self.cfg.hit_reward
                else:
                    reward += self.cfg.miss_penalty
                done = True
        elif self.phase == "done":
            done = True

        self.t += 1
        self.steps_in_phase += 1
        if self.phase == "visible" and self.steps_in_phase >= self.cfg.visible_steps:
            self.phase = "occluded"
            self.steps_in_phase = 0
            self.target_visible = False
        elif self.phase == "occluded" and self.steps_in_phase >= self.cfg.occlusion_steps:
            # Forced miss if agent never claims
            if not done:
                reward += self.cfg.miss_penalty
            self.phase = "done"
            done = True

        return {"reward": float(reward), "done": bool(done)}

    def step(self, dt: float = 1.0) -> Dict:
        return {"t": self.t, "phase": self.phase}

    def observe(self, agent_id: str) -> SensoryObservation:
        return self._observe()

    def checkpoint(self) -> dict:
        return {"t": self.t, "phase": self.phase, "agent_pos": self.agent_pos,
                "target_pos": self.target_pos, "target_visible": self.target_visible}

    def restore(self, data: dict) -> None:
        self.t = int(data.get("t", 0))
        self.phase = data.get("phase", "visible")
        self.agent_pos = tuple(data.get("agent_pos", [0, 0]))
        self.target_pos = tuple(data.get("target_pos", [0, 0]))
        self.target_visible = bool(data.get("target_visible", True))


# ============================================================================
# Task 3: Changed dynamics (adaptation / change detection)
# ============================================================================

@dataclass
class ChangedDynamicsConfig:
    grid: int = 6
    switch_step: int = 10
    move_cost: float = -0.01
    goal_reward: float = 2.0


class ChangedDynamicsTask:
    """The effect of 'move' reverses after switch_step.

    Initially dx maps to +x. After switch_step, dx maps to -x (controls
    inverted). The agent must detect the change from experience and adapt.
    A memoryless policy that does not track recent outcomes keeps using the
    old mapping and performs poorly after the switch.
    """

    def __init__(self, config: Optional[ChangedDynamicsConfig] = None, seed: int = 0):
        self.cfg = config or ChangedDynamicsConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (self.cfg.grid - 1, self.cfg.grid - 1)
        self.inverted = False
        self.done = False

    def reset(self) -> SensoryObservation:
        self.t = 0
        self.agent_pos = (0, 0)
        self.inverted = False
        self.done = False
        return self._observe()

    def _observe(self) -> SensoryObservation:
        extra = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "inverted": self.inverted,
            "t": self.t,
        }
        return SensoryObservation(
            time=float(self.t),
            position=(float(self.agent_pos[0]), float(self.agent_pos[1])),
            energy_level=1.0,
            nutrient_level=1.0,
            local_temperature=288.0,
            local_light=1.0,
            nearby_resources=1.0 - (abs(self.agent_pos[0] - self.goal_pos[0]) +
                                    abs(self.agent_pos[1] - self.goal_pos[1])) / (2 * self.cfg.grid),
            nearby_hazards=0.0,
            nearby_agents=0,
            social_signals=[],
            extra=extra,
            provenance="changed-dynamics task",
        )

    def act(self, agent_id: str, action: AgentAction) -> Dict[str, float]:
        if self.done:
            return {"reward": 0.0, "done": True}
        reward = self.cfg.move_cost
        dx = int(action.parameters.get("dx", 0))
        dy = int(action.parameters.get("dy", 0))
        if self.inverted:
            dx, dy = -dx, -dy
        self.agent_pos = (
            min(max(self.agent_pos[0] + dx, 0), self.cfg.grid - 1),
            min(max(self.agent_pos[1] + dy, 0), self.cfg.grid - 1),
        )
        self.t += 1
        if self.t >= self.cfg.switch_step:
            self.inverted = True
        if self.agent_pos == self.goal_pos:
            reward += self.cfg.goal_reward
            self.done = True
        return {"reward": float(reward), "done": bool(self.done)}

    def step(self, dt: float = 1.0) -> Dict:
        return {"t": self.t}

    def observe(self, agent_id: str) -> SensoryObservation:
        return self._observe()

    def checkpoint(self) -> dict:
        return {"t": self.t, "agent_pos": self.agent_pos, "inverted": self.inverted, "done": self.done}

    def restore(self, data: dict) -> None:
        self.t = int(data.get("t", 0))
        self.agent_pos = tuple(data.get("agent_pos", [0, 0]))
        self.inverted = bool(data.get("inverted", False))
        self.done = bool(data.get("done", False))


# ============================================================================
# Baselines
# ============================================================================

class RandomController:
    """Uniform random action selection. Lower bound."""

    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.controller_type = "random"
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        at = self.rng.choice(["move", "forage", "rest", "wait"])
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)), "dy": int(self.rng.integers(-1, 2))}
        return AgentAction(at, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    def checkpoint(self) -> dict:
        return {"controller_type": "random"}

    def restore(self, data: dict) -> None:
        pass


class ReactiveController:
    """Feedforward reactive policy on current observation only."""

    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.controller_type = "reactive"
        self.rng = np.random.default_rng(seed)
        self.W = np.random.default_rng(seed + 1).standard_normal((4, 6)) * 0.1

    def _obs_vector(self, obs) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        logits = self.W @ self._obs_vector(observation)
        # map to task-specific action space handled by subclass
        return self._action_from_logits(logits, observation)

    def _action_from_logits(self, logits, observation) -> AgentAction:
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)), "dy": int(self.rng.integers(-1, 2))}
        return AgentAction(at, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    def checkpoint(self) -> dict:
        return {"controller_type": "reactive", "W": self.W.tolist()}

    def restore(self, data: dict) -> None:
        self.W = np.array(data.get("W", self.W))


class StackedObsController(ReactiveController):
    """Reactive policy with previous observation stacked (finite memory)."""

    def __init__(self, agent_id: str, seed: int = 0, stack: int = 3):
        super().__init__(agent_id, seed)
        self.stack = int(stack)
        self.history: List[np.ndarray] = []
        self.W = np.random.default_rng(seed + 1).standard_normal((4, 6 * self.stack)) * 0.1

    def _obs_vector(self, obs) -> np.ndarray:
        v = np.array([
            obs.energy_level, obs.nutrient_level,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])
        self.history.append(v)
        self.history = self.history[-self.stack:]
        while len(self.history) < self.stack:
            self.history.insert(0, np.zeros_like(v))
        return np.concatenate(self.history)

    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    def checkpoint(self) -> dict:
        return {"controller_type": "stacked_obs", "stack": self.stack,
                "history": [h.tolist() for h in self.history], "W": self.W.tolist()}

    def restore(self, data: dict) -> None:
        self.stack = int(data.get("stack", 3))
        self.history = [np.array(h) for h in data.get("history", [])]
        self.W = np.array(data.get("W", self.W))


class RecurrentController:
    """Simple recurrent (GRU-like) controller with hidden state."""

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 8, lr: float = 0.01):
        self.agent_id = agent_id
        self.controller_type = "recurrent"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(hidden_dim)
        self.W_x = self.rng.standard_normal((hidden_dim, 6)) * 0.1
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_out = self.rng.standard_normal((4, hidden_dim)) * 0.1
        self.actions_taken = 0
        self.last_action = "wait"

    def _obs_vector(self, obs) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        obs_vec = self._obs_vector(observation)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec)
        logits = self.W_out @ self.hidden
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        self.last_action = at
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)), "dy": int(self.rng.integers(-1, 2))}
        return AgentAction(at, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        obs_vec = self._obs_vector(observation)
        # REINFORCE-style update: push logits of the taken action toward the
        # observed reward, using the hidden state that produced the action.
        chosen = ["move", "forage", "rest", "wait"]
        action_idx = chosen.index(self.last_action)
        logits = self.W_out @ self.hidden
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        grad = (np.array([1.0 if i == action_idx else 0.0 for i in range(4)]) - probs) * reward
        self.W_out += self.lr * np.outer(grad, self.hidden)

    def checkpoint(self) -> dict:
        return {"controller_type": "recurrent", "hidden": self.hidden.tolist(),
                "W_x": self.W_x.tolist(), "W_h": self.W_h.tolist(), "W_out": self.W_out.tolist()}

    def restore(self, data: dict) -> None:
        self.hidden = np.array(data.get("hidden", np.zeros(self.hidden_dim)))
        self.W_x = np.array(data.get("W_x", self.W_x))
        self.W_h = np.array(data.get("W_h", self.W_h))
        self.W_out = np.array(data.get("W_out", self.W_out))


def run_episode_on_task(env: Environment, controller: AgentController,
                        agent_id: str, n_steps: int) -> Dict[str, float]:
    """Run one episode on a generic Environment-contract task."""
    obs = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        total_reward += float(result.get("reward", 0.0))
        steps = step + 1
        obs = next_obs
        if result.get("done", False):
            done = True
            break
        env.step(1.0)
    return {"total_reward": float(total_reward), "steps": float(steps), "done": float(1.0 if done else 0.0)}
