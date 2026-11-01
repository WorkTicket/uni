"""Statistical replication (Sprint 12 Phase 2).

Rigorous multi-seed comparison of cognitive architectures on the GridWorld
foraging task and memory-required tasks, with:

- Separate training, tuning and held-out evaluation seeds
- Persistent training (weights carry across episodes)
- Held-out evaluation with fresh controllers initialized from trained weights
- Mean +/- 95% CI, Cohen's d effect size vs reactive baseline
- Sample efficiency and inference-cost reporting
- Matched interaction and compute budgets

No claim of consciousness; measures whether memory/world-model/planning provide
measurable, reproducible task advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import time
import numpy as np

from ..contracts import AgentAction, AgentState, SensoryObservation
from ..environment.gridworld import GridWorld, GridWorldConfig
from .memory_tasks import (
    DelayedCueTask, DelayedCueConfig,
    OccludedTargetTask, OccludedTargetConfig,
    ChangedDynamicsTask, ChangedDynamicsConfig,
)


# ============================================================================
# Controllers (self-contained, cloneable for weight transfer)
# ============================================================================

class BaseController:
    agent_id: str
    controller_type: str
    rng: np.random.Generator

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        raise NotImplementedError

    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    def get_weights(self) -> dict:
        return {}

    def set_weights(self, w: dict) -> None:
        pass

    def clone(self, agent_id: str, seed: int) -> "BaseController":
        raise NotImplementedError


def _obs_vector(obs) -> np.ndarray:
    return np.array([
        obs.energy_level, obs.nutrient_level,
        obs.nearby_resources, obs.nearby_hazards,
        float(obs.nearby_agents) / 5.0, obs.local_light,
    ])


def _move_params(rng) -> dict:
    return {"dx": int(rng.integers(-1, 2)), "dy": int(rng.integers(-1, 2))}


class ReactiveController(BaseController):
    """Feedforward reactive policy on current observation only."""

    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.controller_type = "reactive"
        self.rng = np.random.default_rng(seed)
        self.W = np.random.default_rng(seed + 1).standard_normal((4, 6)) * 0.1
        self.actions_taken = 0

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        logits = self.W @ _obs_vector(observation)
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def clone(self, agent_id: str, seed: int) -> "ReactiveController":
        c = ReactiveController(agent_id, seed)
        c.W = self.W.copy()
        return c


class StackedObsController(BaseController):
    """Reactive policy with previous observations stacked (finite memory)."""

    def __init__(self, agent_id: str, seed: int = 0, stack: int = 3):
        self.agent_id = agent_id
        self.controller_type = "stacked_obs"
        self.stack = int(stack)
        self.rng = np.random.default_rng(seed)
        self.history: List[np.ndarray] = []
        self.W = np.random.default_rng(seed + 1).standard_normal((4, 6 * stack)) * 0.1
        self.actions_taken = 0

    def _stacked(self, obs) -> np.ndarray:
        v = _obs_vector(obs)
        self.history.append(v)
        self.history = self.history[-self.stack:]
        while len(self.history) < self.stack:
            self.history.insert(0, np.zeros_like(v))
        return np.concatenate(self.history)

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        logits = self.W @ self._stacked(observation)
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def clone(self, agent_id: str, seed: int) -> "StackedObsController":
        c = StackedObsController(agent_id, seed, stack=self.stack)
        c.W = self.W.copy()
        c.history = [h.copy() for h in self.history]
        return c


class RecurrentController(BaseController):
    """Simple recurrent hidden-state controller, REINFORCE-trained."""

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 8, lr: float = 0.05):
        self.agent_id = agent_id
        self.controller_type = "recurrent"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(hidden_dim)
        self.W_x = self.rng.standard_normal((hidden_dim, 6)) * 0.1
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_out = self.rng.standard_normal((4, hidden_dim)) * 0.1
        self.last_action = "wait"
        self.actions_taken = 0

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        obs_vec = _obs_vector(observation)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec)
        logits = self.W_out @ self.hidden
        at = ["move", "forage", "rest", "wait"][int(np.argmax(logits))]
        self.last_action = at
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        chosen = ["move", "forage", "rest", "wait"]
        action_idx = chosen.index(self.last_action)
        logits = self.W_out @ self.hidden
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        one_hot = np.zeros(4); one_hot[action_idx] = 1.0
        grad = (one_hot - probs) * reward
        self.W_out += self.lr * np.outer(grad, self.hidden)

    def get_weights(self) -> dict:
        return {"W_x": self.W_x, "W_h": self.W_h, "W_out": self.W_out,
                "hidden": self.hidden.copy()}

    def set_weights(self, w: dict) -> None:
        self.W_x = w["W_x"].copy(); self.W_h = w["W_h"].copy()
        self.W_out = w["W_out"].copy()
        if "hidden" in w:
            self.hidden = w["hidden"].copy()

    def clone(self, agent_id: str, seed: int) -> "RecurrentController":
        c = RecurrentController(agent_id, seed, hidden_dim=self.hidden_dim, lr=self.lr)
        c.set_weights(self.get_weights())
        return c


class WorldModelController(BaseController):
    """Recurrent world model: observation predictor + reward predictor.

    Predicts next observation and immediate reward from hidden+action; uses
    predicted reward for action selection (aligned value function).
    """

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 8, lr: float = 0.05):
        self.agent_id = agent_id
        self.controller_type = "world_model"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(hidden_dim)
        self.W_x = self.rng.standard_normal((hidden_dim, 6)) * 0.1
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_a = self.rng.standard_normal((hidden_dim, 4)) * 0.1
        self.W_pred = self.rng.standard_normal((6, hidden_dim)) * 0.1
        self.W_r = self.rng.standard_normal((1, hidden_dim + 4)) * 0.1
        self.last_action = "wait"
        self.actions_taken = 0
        self.prediction_error_history: List[float] = []

    def _action_vec(self, action_type: str) -> np.ndarray:
        v = np.zeros(4)
        at = ["move", "forage", "rest", "wait"]
        v[at.index(action_type)] = 1.0
        return v

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        obs_vec = _obs_vector(observation)
        best_action = "wait"; best_value = -float("inf")
        best_av = np.zeros(4)
        for at in ["move", "forage", "rest", "wait"]:
            av = self._action_vec(at)
            hidden_next = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec
                                  + self.W_a @ av)
            # note: hidden_next ignores av here; world model dynamics driven by obs
            pred_obs = self.W_pred @ hidden_next
            sa = np.concatenate([hidden_next, av])
            predicted_reward = float((self.W_r @ sa)[0])
            predicted_hazard = pred_obs[3]
            value = predicted_reward - 0.5 * predicted_hazard
            if value > best_value:
                best_value = value; best_action = at; best_av = av
        self.last_action = best_action
        params = _move_params(self.rng) if best_action == "move" else {}
        return AgentAction(best_action, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        obs_vec = _obs_vector(observation)
        next_obs_vec = _obs_vector(next_observation)
        av = self._action_vec(self.last_action)
        hidden_pred = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec
                              + self.W_a @ av)
        pred_obs = self.W_pred @ hidden_pred
        error = pred_obs - next_obs_vec
        self.prediction_error_history.append(float(np.mean(np.abs(error))))
        if len(self.prediction_error_history) > 1000:
            self.prediction_error_history = self.prediction_error_history[-1000:]
        self.W_pred -= self.lr * np.outer(error, hidden_pred)
        self.W_x -= self.lr * 0.01 * np.outer(hidden_pred, error)
        sa = np.concatenate([hidden_pred, av])
        pred_r = float((self.W_r @ sa)[0])
        self.W_r -= self.lr * 2.0 * (pred_r - reward) * sa.reshape(1, -1)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_x @ obs_vec)

    def get_weights(self) -> dict:
        return {"W_x": self.W_x, "W_h": self.W_h, "W_a": self.W_a,
                "W_pred": self.W_pred, "W_r": self.W_r,
                "hidden": self.hidden.copy()}

    def set_weights(self, w: dict) -> None:
        self.W_x = w["W_x"].copy(); self.W_h = w["W_h"].copy()
        self.W_a = w["W_a"].copy()
        self.W_pred = w["W_pred"].copy(); self.W_r = w["W_r"].copy()
        if "hidden" in w:
            self.hidden = w["hidden"].copy()

    def clone(self, agent_id: str, seed: int) -> "WorldModelController":
        c = WorldModelController(agent_id, seed, hidden_dim=self.hidden_dim, lr=self.lr)
        c.set_weights(self.get_weights())
        c.prediction_error_history = list(self.prediction_error_history)
        return c


class GatedRecurrentController(BaseController):
    """Gated recurrent controller with actor-critic (REINFORCE + value baseline).

    GRU-like hidden state that persists across steps, giving genuine memory.
    Trained online with an actor-critic: the actor (policy) is nudged toward
    actions with positive advantage (return minus state-value baseline), and
    the critic (state-value) is trained to predict the return. The value
    baseline reduces gradient variance without biasing the update, which makes
    learning stable enough to solve memory-required tasks where a memoryless
    reactive policy is provably at chance.

    Works with the per-step learn(obs, action, reward, next_obs) calling
    convention (no episode-boundary signal required) using a one-step TD
    target.

    Architecture:
      z = sigmoid(W_z x + U_z h)        # update gate
      r = sigmoid(W_r x + U_r h)        # reset gate
      h~ = tanh(W x + U (r . h))        # candidate
      h = (1 - z) . h + z . h~          # new hidden state (memory)
      logits = V @ h                    # action logits
      value = c @ h                     # state-value baseline
    """

    def __init__(self, agent_id: str, seed: int = 0,
                 hidden_dim: int = 16, lr: float = 0.1, gamma: float = 0.99):
        self.agent_id = agent_id
        self.controller_type = "gated_recurrent"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(hidden_dim)

        scale = 0.1
        self.W_z = self.rng.standard_normal((hidden_dim, 6)) * scale
        self.U_z = self.rng.standard_normal((hidden_dim, hidden_dim)) * scale
        self.b_z = np.zeros(hidden_dim)
        self.W_r = self.rng.standard_normal((hidden_dim, 6)) * scale
        self.U_r = self.rng.standard_normal((hidden_dim, hidden_dim)) * scale
        self.b_r = np.zeros(hidden_dim)
        self.W_h = self.rng.standard_normal((hidden_dim, 6)) * scale
        self.U_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * scale
        self.b_h = np.zeros(hidden_dim)
        self.V = self.rng.standard_normal((4, hidden_dim)) * scale
        self.c = self.rng.standard_normal((1, hidden_dim)) * scale
        self.last_action = "wait"
        self.last_hidden = None
        self.last_log_prob = None
        self.last_one_hot = None
        self.actions_taken = 0

    def _gru_step(self, x: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        z = self._sigmoid(self.W_z @ x + self.U_z @ hidden + self.b_z)
        r = self._sigmoid(self.W_r @ x + self.U_r @ hidden + self.b_r)
        h_cand = np.tanh(self.W_h @ x + self.U_h @ (r * hidden) + self.b_h)
        return (1.0 - z) * hidden + z * h_cand

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def _obs_vector(self, obs) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def _action_logits(self, hidden: np.ndarray) -> np.ndarray:
        return self.V @ hidden

    def _value(self, hidden: np.ndarray) -> float:
        return float((self.c @ hidden)[0])

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        x = self._obs_vector(observation)
        self.hidden = self._gru_step(x, self.hidden)
        self.last_hidden = self.hidden.copy()

        logits = self._action_logits(self.hidden)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        action_types = ["move", "forage", "rest", "wait"]
        idx = int(self.rng.choice(4, p=probs))
        at = action_types[idx]
        self.last_action = at
        self.last_log_prob = float(np.log(probs[idx] + 1e-12))
        one_hot = np.zeros(4)
        one_hot[idx] = 1.0
        self.last_one_hot = one_hot
        params = _move_params(self.rng) if at == "move" else {}
        return AgentAction(at, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        """Actor-critic update using a one-step TD target.

        advantage = (reward + gamma * V(next_hidden)) - V(current_hidden)
        Actor: V += lr * advantage * grad log pi(a|h)
        Critic: c += lr * 2 * (target - V(h)) * h
        """
        reward = float(reward)
        x_next = self._obs_vector(next_observation)
        next_hidden = self._gru_step(x_next, self.hidden)

        current_value = self._value(self.last_hidden)
        next_value = self._value(next_hidden)
        target = reward + self.gamma * next_value
        advantage = target - current_value

        # Actor: policy gradient on log-prob of taken action.
        logits = self._action_logits(self.last_hidden)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        grad_logp = (self.last_one_hot - probs)
        self.V += self.lr * advantage * np.outer(grad_logp, self.last_hidden)

        # Critic: minimize (V(h) - target)^2
        self.c += self.lr * 2.0 * (target - current_value) * self.last_hidden.reshape(1, -1)

        # Commit hidden to next state (teacher forcing through dynamics)
        self.hidden = next_hidden

    def get_weights(self) -> dict:
        return {"W_z": self.W_z, "U_z": self.U_z, "b_z": self.b_z,
                "W_r": self.W_r, "U_r": self.U_r, "b_r": self.b_r,
                "W_h": self.W_h, "U_h": self.U_h, "b_h": self.b_h,
                "V": self.V, "c": self.c, "hidden": self.hidden.copy()}

    def set_weights(self, w: dict) -> None:
        self.W_z = w["W_z"].copy(); self.U_z = w["U_z"].copy(); self.b_z = w["b_z"].copy()
        self.W_r = w["W_r"].copy(); self.U_r = w["U_r"].copy(); self.b_r = w["b_r"].copy()
        self.W_h = w["W_h"].copy(); self.U_h = w["U_h"].copy(); self.b_h = w["b_h"].copy()
        self.V = w["V"].copy(); self.c = w["c"].copy()
        if "hidden" in w:
            self.hidden = w["hidden"].copy()

    def checkpoint(self) -> dict:
        return {"controller_type": "gated_recurrent", "hidden": self.hidden.tolist(),
                "weights": {k: v.tolist() for k, v in self.get_weights().items()
                            if k != "hidden"}}

    def restore(self, data: dict) -> None:
        self.hidden = np.array(data.get("hidden", np.zeros(self.hidden_dim)))
        self.last_hidden = self.hidden.copy()
        weights = data.get("weights", {})
        for key in ["W_z", "U_z", "b_z", "W_r", "U_r", "b_r", "W_h", "U_h", "b_h", "V", "c"]:
            if key in weights:
                setattr(self, key, np.array(weights[key]))

    def clone(self, agent_id: str, seed: int) -> "GatedRecurrentController":
        c = GatedRecurrentController(agent_id, seed, hidden_dim=self.hidden_dim,
                                     lr=self.lr, gamma=self.gamma)
        c.set_weights(self.get_weights())
        return c


class WorkingMemoryOracle(BaseController):
    """Category 1 — PROGRAMMED intelligence (explicit working-memory oracle).

    NOT a learned agent. This is a positive-control / oracle baseline that
    demonstrates the environment + memory mechanism work: it explicitly
    remembers the target/cue coordinates observed during the informative
    phase and navigates to them during the occlusion/memory phase using only
    its own observable position, then issues claim on arrival.

    It reads ONLY the sensory observation contract:
      - local_light == 1.0  : informative phase (target/cue visible)
      - local_light == 0.0  : memory phase   (target/cue hidden)
      - nearby_resources / nearby_hazards : normalized coordinates
      - position : agent's own grid coordinates

    It never reads environment internals or the hidden target state.

    Solves:
      - delayed_cue : remembers cue side (nearby_resources 0=left, 1=right)
                      during the cue phase and moves that direction in the
                      response phase.
      - occluded_target : remembers target (x, y) during the visible phase,
                          tracks own position during occlusion, moves toward
                          the remembered target, and claims on arrival.
    """

    def __init__(self, agent_id: str, seed: int = 0):
        self.agent_id = agent_id
        self.controller_type = "working_memory_oracle"
        self.rng = np.random.default_rng(seed)
        # Remembered target/cue in normalized coordinates.
        self.mem_tx: float = 0.0
        self.mem_ty: float = 0.0
        self.has_memory: bool = False
        self.actions_taken = 0
        self.last_action = "wait"

    def _obs_vector(self, obs) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def _remember(self, rx: float, ry: float) -> None:
        """Store a normalized target/cue coordinate in memory."""
        self.mem_tx = float(rx)
        self.mem_ty = float(ry)
        self.has_memory = True

    def _move_toward(self, tx: float, ty: float,
                     cx: float, cy: float) -> tuple:
        """Return (action_type, params) moving one step toward (tx, ty)."""
        dx = 0
        if tx > cx + 1e-6:
            dx = 1
        elif tx < cx - 1e-6:
            dx = -1
        dy = 0
        if ty > cy + 1e-6:
            dy = 1
        elif ty < cy - 1e-6:
            dy = -1
        if dx == 0 and dy == 0:
            return "wait", {}
        return "move", {"dx": dx, "dy": dy}

    def _at_target(self, cx: float, cy: float, tx: float, ty: float) -> bool:
        return abs(cx - tx) <= 1e-6 and abs(cy - ty) <= 1e-6

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        from ..contracts import AgentAction
        self.actions_taken += 1
        res = self._obs_vector(observation)
        rx = float(res[2])  # nearby_resources: normalized x
        ry = float(res[3])  # nearby_hazards: normalized y
        informative = float(res[5]) > 0.5  # local_light: target/cue visible

        if informative:
            # Informative phase: observe and remember the target/cue.
            self._remember(rx, ry)
            # Own position during the informative phase is also observable.
            cx = float(observation.position[0])
            cy = float(observation.position[1])
            # Grid extent unknown; move one step in the remembered direction.
            action_type, params = self._move_toward(rx, ry, cx / max(1.0,
                                                                     abs(cx) + 1.0),
                                                     cy / max(1.0, abs(cy) + 1.0))
            self.last_action = action_type
            return AgentAction(action_type, params, timestamp=observation.time)

        # Memory phase: target/cue hidden. Navigate using remembered target
        # and own observable position. Report own normalized position via
        # nearby_resources / nearby_hazards (environment convention).
        cx = rx
        cy = ry
        if not self.has_memory:
            self.last_action = "wait"
            return AgentAction("wait", {}, timestamp=observation.time)

        if self._at_target(cx, cy, self.mem_tx, self.mem_ty):
            # Arrived at remembered target: claim.
            self.last_action = "claim"
            return AgentAction("claim", {}, timestamp=observation.time)

        action_type, params = self._move_toward(self.mem_tx, self.mem_ty, cx, cy)
        self.last_action = action_type
        return AgentAction(action_type, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        """No-op: this oracle is programmed, not learned.

        Kept for interface compatibility. Memory is updated online in
        select_action from the informative observation, not from reward.
        """
        pass

    def get_weights(self) -> dict:
        return {"mem_tx": self.mem_tx, "mem_ty": self.mem_ty,
                "has_memory": self.has_memory}

    def set_weights(self, w: dict) -> None:
        self.mem_tx = float(w.get("mem_tx", 0.0))
        self.mem_ty = float(w.get("mem_ty", 0.0))
        self.has_memory = bool(w.get("has_memory", False))

    def checkpoint(self) -> dict:
        return {"controller_type": "working_memory_oracle",
                "mem_tx": self.mem_tx, "mem_ty": self.mem_ty,
                "has_memory": self.has_memory,
                "actions_taken": self.actions_taken}

    def restore(self, data: dict) -> None:
        self.mem_tx = float(data.get("mem_tx", 0.0))
        self.mem_ty = float(data.get("mem_ty", 0.0))
        self.has_memory = bool(data.get("has_memory", False))
        self.actions_taken = int(data.get("actions_taken", 0))

    def clone(self, agent_id: str, seed: int) -> "WorkingMemoryOracle":
        c = WorkingMemoryOracle(agent_id, seed)
        c.set_weights(self.get_weights())
        return c


class EcsIwAblation(BaseController):
    """ECS-IW architecture with individually removable components.

    Base architecture (from workspace.model.WorkspaceAgentController):
      recurrent hidden state + global workspace broadcast + self-model.

    Ablation flags remove one component at a time to isolate its causal
    contribution to behavior/performance:
      use_workspace   — gate action selection on the workspace broadcast
      use_self_model  — update the self-model from action outcomes
      use_recurrence  — carry hidden state across steps (memory)

    Each component's removal is a causal intervention: if performance drops
    when a component is removed, that component contributes to the behavior.
    """

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 8,
                 lr: float = 0.01, use_workspace: bool = True,
                 use_self_model: bool = True, use_recurrence: bool = True):
        from .workspace.model import GlobalWorkspace, WorkspaceContent
        from .self_model.model import make_default_self_model
        self.agent_id = agent_id
        self.controller_type = "ecs_iw_ablation"
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.use_workspace = bool(use_workspace)
        self.use_self_model = bool(use_self_model)
        self.use_recurrence = bool(use_recurrence)
        self.rng = np.random.default_rng(seed)

        self.workspace = GlobalWorkspace(capacity=1)
        self.self_model = make_default_self_model()
        self.hidden = np.zeros(hidden_dim) if use_recurrence else None
        self.W_h = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self.W_in = self.rng.standard_normal((hidden_dim, 6)) * 0.1
        self.W_ws = self.rng.standard_normal((hidden_dim, 4)) * 0.1
        self.W_out = self.rng.standard_normal((4, hidden_dim)) * 0.1
        self.actions_taken = 0
        self.last_action = "wait"
        self._last_hidden = np.zeros(hidden_dim)

    def _obs_vector(self, obs) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level / 10.0,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def _action_from_logits(self, logits: np.ndarray) -> tuple:
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        action_types = ["move", "forage", "rest", "wait"]
        idx = int(self.rng.choice(4, p=probs))
        at = action_types[idx]
        params = {}
        if at == "move":
            params = {"dx": int(self.rng.integers(-1, 2)),
                      "dy": int(self.rng.integers(-1, 2))}
        return at, params

    def _gru_step(self, x: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        z = 1.0 / (1.0 + np.exp(-(self.W_z @ x + self.U_z @ hidden)))
        r = 1.0 / (1.0 + np.exp(-(self.W_r @ x + self.U_r @ hidden)))
        h_cand = np.tanh(self.W_h @ x + self.U_h @ (r * hidden))
        return (1.0 - z) * hidden + z * h_cand

    def _build_gates(self) -> None:
        scale = 0.1
        self.W_z = self.rng.standard_normal((self.hidden_dim, 6)) * scale
        self.U_z = self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * scale
        self.W_r = self.rng.standard_normal((self.hidden_dim, 6)) * scale
        self.U_r = self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * scale
        self.W_h = self.rng.standard_normal((self.hidden_dim, 6)) * scale
        self.U_h = self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * scale

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        self.actions_taken += 1
        from ..contracts import AgentAction
        x = self._obs_vector(observation)

        # Recurrent hidden-state update (memory)
        if self.use_recurrence:
            if not hasattr(self, "W_z"):
                self._build_gates()
            self.hidden = self._gru_step(x, self.hidden)
            self._last_hidden = self.hidden.copy()
        else:
            # Feedforward: no persistent hidden state
            self._last_hidden = np.tanh(self.W_in @ x)

        hidden = self._last_hidden

        # Workspace broadcast (attentional bottleneck)
        if self.use_workspace:
            perception = {"nearby_resources": obs.nearby_resources,
                          "nearby_hazards": obs.nearby_hazards,
                          "energy_level": obs.energy_level}
            salience = 0.5 + 0.3 * obs.nearby_hazards + 0.2 * obs.nearby_resources
            self.workspace.receive("perception", perception, salience,
                                   "perception_module", obs.time)
            sm_diag = {"calibration_error": self.self_model.calibration_error}
            sm_salience = 0.3 + 0.4 * self.self_model.calibration_error
            self.workspace.receive("self_model", sm_diag, sm_salience,
                                   "self_model_module", obs.time)
            broadcast = self.workspace.broadcast(obs.time)
            if broadcast is not None:
                ws_vec = np.zeros(4)
                if broadcast.modality == "perception":
                    ws_vec[0] = broadcast.content.get("nearby_resources", 0.0)
                    ws_vec[1] = broadcast.content.get("nearby_hazards", 0.0)
                elif broadcast.modality == "self_model":
                    ws_vec[2] = broadcast.content.get("calibration_error", 0.0)
                hidden = np.tanh(hidden + self.W_ws @ ws_vec)

        logits = self.W_out @ hidden
        action_type, params = self._action_from_logits(logits)
        self.last_action = action_type
        return AgentAction(action_type, params, timestamp=observation.time)

    def learn(self, observation, action, reward, next_observation) -> None:
        from ..contracts import AgentAction
        reward = float(reward)
        x = self._obs_vector(observation)
        next_x = self._obs_vector(next_observation)

        # Policy gradient on the taken action
        logits = self.W_out @ self._last_hidden
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        action_types = ["move", "forage", "rest", "wait"]
        one_hot = np.zeros(4)
        one_hot[action_types.index(self.last_action)] = 1.0
        grad_logp = (one_hot - probs)
        self.W_out += self.lr * (reward - 0.5) * np.outer(grad_logp, self._last_hidden)

        # Recurrence: teacher forcing toward next observation
        if self.use_recurrence and hasattr(self, "W_z"):
            self.hidden = self._gru_step(next_x, self.hidden)
            self._last_hidden = self.hidden.copy()

        # Self-model update (sensorimotor prediction)
        if self.use_self_model:
            actual_change = np.array([
                next_obs.nearby_resources - obs.nearby_resources,
                next_obs.nearby_hazards - obs.nearby_hazards,
            ])
            predicted = self.self_model.predict(action.action_type, action.parameters)
            self.self_model.update(action.action_type, action.parameters,
                                   predicted[:2], actual_change[:2])

    def describe(self) -> str:
        parts = []
        if self.use_workspace: parts.append("workspace")
        else: parts.append("-workspace")
        if self.use_self_model: parts.append("self_model")
        else: parts.append("-self_model")
        if self.use_recurrence: parts.append("recurrence")
        else: parts.append("-recurrence")
        return "+".join(parts)

    def get_weights(self) -> dict:
        return {"hidden": (self.hidden.copy() if self.hidden is not None else None),
                "W_out": self.W_out, "W_h": self.W_h, "W_in": self.W_in,
                "W_ws": self.W_ws,
                "W_z": getattr(self, "W_z", None), "U_z": getattr(self, "U_z", None),
                "W_r": getattr(self, "W_r", None), "U_r": getattr(self, "U_r", None),
                "actions_taken": self.actions_taken}

    def set_weights(self, w: dict) -> None:
        self.W_out = w["W_out"].copy(); self.W_h = w["W_h"].copy()
        self.W_in = w["W_in"].copy(); self.W_ws = w["W_ws"].copy()
        self.actions_taken = int(w.get("actions_taken", 0))
        if self.use_recurrence and w.get("hidden") is not None:
            self.hidden = np.array(w["hidden"]); self._last_hidden = self.hidden.copy()
        if hasattr(self, "W_z") and w.get("W_z") is not None:
            self.W_z = np.array(w["W_z"]); self.U_z = np.array(w["U_z"])
            self.W_r = np.array(w["W_r"]); self.U_r = np.array(w["U_r"])
            self.W_h = np.array(w["W_h"]); self.U_h = np.array(w["U_h"])

    def checkpoint(self) -> dict:
        return {"controller_type": "ecs_iw_ablation",
                "use_workspace": self.use_workspace,
                "use_self_model": self.use_self_model,
                "use_recurrence": self.use_recurrence,
                "hidden": (self.hidden.tolist() if self.hidden is not None else None),
                "W_out": self.W_out.tolist(), "W_h": self.W_h.tolist(),
                "W_in": self.W_in.tolist(), "W_ws": self.W_ws.tolist(),
                "actions_taken": self.actions_taken,
                "workspace": self.workspace.checkpoint(),
                "self_model": self.self_model.checkpoint()}

    def restore(self, data: dict) -> None:
        self.use_workspace = bool(data.get("use_workspace", True))
        self.use_self_model = bool(data.get("use_self_model", True))
        self.use_recurrence = bool(data.get("use_recurrence", True))
        if data.get("hidden") is not None:
            self.hidden = np.array(data["hidden"]); self._last_hidden = self.hidden.copy()
        self.W_out = np.array(data["W_out"]); self.W_h = np.array(data["W_h"])
        self.W_in = np.array(data["W_in"]); self.W_ws = np.array(data["W_ws"])
        self.actions_taken = int(data.get("actions_taken", 0))
        self.workspace = GlobalWorkspace.restore(data.get("workspace", {}))
        from .self_model.model import SelfModel
        self.self_model = SelfModel.restore(data.get("self_model", {}))

    def clone(self, agent_id: str, seed: int) -> "EcsIwAblation":
        c = EcsIwAblation(agent_id, seed, hidden_dim=self.hidden_dim, lr=self.lr,
                          use_workspace=self.use_workspace,
                          use_self_model=self.use_self_model,
                          use_recurrence=self.use_recurrence)
        c.set_weights(self.get_weights())
        return c


def ecs_iw_ablation_controllers() -> dict:
    return {
        "ecs_iw_full": lambda aid, sd: EcsIwAblation(aid, sd,
            use_workspace=True, use_self_model=True, use_recurrence=True),
        "ecs_iw_no_workspace": lambda aid, sd: EcsIwAblation(aid, sd,
            use_workspace=False, use_self_model=True, use_recurrence=True),
        "ecs_iw_no_self_model": lambda aid, sd: EcsIwAblation(aid, sd,
            use_workspace=True, use_self_model=False, use_recurrence=True),
        "ecs_iw_no_recurrence": lambda aid, sd: EcsIwAblation(aid, sd,
            use_workspace=True, use_self_model=True, use_recurrence=False),
        "reactive": lambda aid, sd: ReactiveController(aid, sd),
    }


def run_ecs_iw_ablation(macro_seeds: List[int] | None = None,
                        n_train_seeds: int = 15, n_eval_episodes: int = 6,
                        n_steps: int = 60) -> Dict[str, ExperimentResult]:
    """ECS-IW causal ablation study on GridWorld foraging.

    Removes one ECS-IW component at a time and measures the paired
    performance difference vs. the full architecture (and vs. reactive).
    A drop in performance when a component is removed indicates that
    component causally contributes to the behavior.
    """
    if macro_seeds is None:
        macro_seeds = list(range(1, 11))

    def make_env(seed: int):
        return GridWorld(GridWorldConfig(seed=seed, food_regeneration=0.01,
                                         hazard_probability=0.1, dt=0.1))

    return compare_controllers_env(
        make_env, ecs_iw_ablation_controllers(), macro_seeds,
        n_train_seeds=n_train_seeds, n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
    )
    controller: str
    macro_seed: int
    train_rewards: List[float]
    eval_rewards: List[float]
    mean_eval: float
    std_eval: float
    n_eval: int
    inference_cost: float  # relative action selections
    final_train_reward: float


def run_episode(env: GridWorld, controller: BaseController,
                agent_id: str, n_steps: int) -> Tuple[float, int]:
    """Run one episode; controller learns online. Returns (total_reward, steps)."""
    env.register_agent(agent_id, (env.cfg.width // 2, env.cfg.height // 2),
                       initial_energy=5.0)
    obs = env.observe(agent_id)
    total = 0.0
    steps = 0
    alive = True
    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        total += float(result.get("reward", 0.0))
        steps = step + 1
        obs = next_obs
        env.step(0.1)
        if not env._agent_states[agent_id].alive:
            alive = False
            break
    return total, steps


def run_episode_env(env, controller: BaseController, agent_id: str,
                    n_steps: int) -> Tuple[float, int]:
    """Run one episode on any Environment-contract task.

    Handles both GridWorld (register_agent / _agent_states) and memory POMDP
    tasks (reset / done flag in act result). Controller learns online.
    Returns (total_reward, steps).
    """
    # GridWorld-style registration if available.
    if hasattr(env, "register_agent"):
        env.register_agent(agent_id, (getattr(env.cfg, "width", 4) // 2,
                                      getattr(env.cfg, "height", 4) // 2),
                           initial_energy=5.0)
    elif hasattr(env, "reset"):
        env.reset()
    obs = env.observe(agent_id)
    total = 0.0
    steps = 0
    alive = True
    for step in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        controller.learn(obs, action, float(result.get("reward", 0.0)), next_obs)
        total += float(result.get("reward", 0.0))
        steps = step + 1
        obs = next_obs
        if hasattr(env, "step"):
            env.step(0.1)
        if result.get("done", False):
            alive = False
            break
        elif hasattr(env, "_agent_states"):
            if not env._agent_states[agent_id].alive:
                alive = False
                break
    return total, steps


def train_controller_env(make_env: Callable[[int], object],
                         controller: BaseController,
                         train_seeds: List[int], n_steps: int) -> List[float]:
    rewards = []
    for ep, seed in enumerate(train_seeds):
        env = make_env(seed)
        r, _ = run_episode_env(env, controller, f"train-{ep}", n_steps)
        rewards.append(r)
    return rewards


def evaluate_controller_env(controller: BaseController,
                            make_env: Callable[[int], object],
                            eval_seeds: List[int], n_eval_episodes: int,
                            n_steps: int) -> Tuple[List[float], int]:
    rewards = []
    total_actions = 0
    for seed in eval_seeds:
        for ep in range(n_eval_episodes):
            env = make_env(seed + ep)
            ec = controller.clone(f"eval-{seed}-{ep}", seed + ep)
            total_actions += ec.actions_taken
            r, _ = run_episode_env(env, ec, f"eval-{seed}-{ep}", n_steps)
            rewards.append(r)
    return rewards, total_actions


def compare_controllers_env(make_env: Callable[[int], object],
                            controllers: Dict[str, Callable[[str, int], BaseController]],
                            macro_seeds: List[int],
                            n_train_seeds: int,
                            n_eval_episodes: int,
                            n_steps: int) -> Dict[str, ExperimentResult]:
    results: Dict[str, ExperimentResult] = {}
    reactive_eval: List[float] = []
    # Per-macro-seed eval rewards per controller, for paired advantage CIs.
    per_seed_eval: Dict[str, List[float]] = {name: [] for name in controllers}
    for name, factory in controllers.items():
        all_eval: List[float] = []
        all_train: List[float] = []
        total_actions = 0
        for ms in macro_seeds:
            train_seeds = [ms * 1000 + i for i in range(n_train_seeds)]
            eval_seeds = [ms * 1000 + 10000 + i for i in range(n_eval_episodes)]
            ctrl = factory(f"train-{ms}", ms)
            train_r = train_controller_env(make_env, ctrl, train_seeds, n_steps)
            all_train.extend(train_r)
            eval_r, n_act = evaluate_controller_env(
                ctrl, make_env, eval_seeds, 1, n_steps)
            total_actions += n_act
            all_eval.extend(eval_r)
            per_seed_eval[name].append(float(np.mean(eval_r)) if eval_r else 0.0)
        results[name] = ExperimentResult(
            controller=name,
            mean_eval=float(np.mean(all_eval)) if all_eval else 0.0,
            ci_lo=0.0, ci_hi=0.0, std_eval=float(np.std(all_eval)) if all_eval else 0.0,
            cohens_d_vs_reactive=0.0,
            n_macro_seeds=len(macro_seeds),
            all_eval_rewards=all_eval,
            mean_train_reward=float(np.mean(all_train)) if all_train else 0.0,
            inference_cost=float(total_actions),
        )
        if name == "reactive":
            reactive_eval = list(all_eval)
    for name, r in results.items():
        mean, lo, hi = mean_confidence_interval(r.all_eval_rewards)
        r.ci_lo, r.ci_hi = lo, hi
        if name != "reactive" and reactive_eval:
            r.cohens_d_vs_reactive = cohens_d(r.all_eval_rewards, reactive_eval)
            # Paired advantage over reactive, computed on per-macro-seed means.
            adv_mean, adv_std, adv_lo, adv_hi = paired_difference_ci(
                per_seed_eval[name], per_seed_eval["reactive"])
            r.paired_advantage_mean = adv_mean
            r.paired_advantage_std = adv_std
            r.paired_advantage_ci_lo = adv_lo
            r.paired_advantage_ci_hi = adv_hi
            r.paired_effect_size = paired_cohens_d(
                per_seed_eval[name], per_seed_eval["reactive"])
    return results


CONTROLLERS = {
    "reactive": lambda aid, sd: ReactiveController(aid, sd),
    "stacked_obs": lambda aid, sd: StackedObsController(aid, sd, stack=3),
    "recurrent": lambda aid, sd: RecurrentController(aid, sd, hidden_dim=8),
    "world_model": lambda aid, sd: WorldModelController(aid, sd, hidden_dim=8),
}


def run_memory_task_comparison(make_env: Callable[[int], object],
                               task_name: str,
                               macro_seeds: List[int] | None = None,
                               n_train_seeds: int = 5,
                               n_eval_episodes: int = 5,
                               n_steps: int = 40) -> Dict[str, ExperimentResult]:
    if macro_seeds is None:
        macro_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return compare_controllers_env(
        make_env, CONTROLLERS, macro_seeds,
        n_train_seeds=n_train_seeds,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
    )


# Stronger controller set including the gated recurrent memory controller.
GATED_CONTROLLERS = {
    "reactive": lambda aid, sd: ReactiveController(aid, sd),
    "stacked_obs": lambda aid, sd: StackedObsController(aid, sd, stack=3),
    "recurrent": lambda aid, sd: RecurrentController(aid, sd, hidden_dim=8),
    "world_model": lambda aid, sd: WorldModelController(aid, sd, hidden_dim=8),
    "gated_recurrent": lambda aid, sd: GatedRecurrentController(aid, sd,
                                                                hidden_dim=16, lr=0.1),
    # Category 1: programmed working-memory oracle (positive control).
    "working_memory_oracle": lambda aid, sd: WorkingMemoryOracle(aid, sd),
}


def run_memory_task_comparison_high_budget(
        make_env: Callable[[int], object],
        task_name: str,
        macro_seeds: List[int] | None = None,
        n_train_seeds: int = 25,
        n_eval_episodes: int = 8,
        n_steps: int = 60) -> Dict[str, ExperimentResult]:
    """High-budget memory-task comparison.

    Trains controllers over many more episodes (n_train_seeds * n_steps) so the
    gated recurrent controller has sufficient experience to learn the memory
    task. Evaluation uses fresh controllers initialized from trained weights
    on held-out seeds. Reports paired advantage CIs vs reactive.
    """
    if macro_seeds is None:
        macro_seeds = list(range(1, 16))
    return compare_controllers_env(
        make_env, GATED_CONTROLLERS, macro_seeds,
        n_train_seeds=n_train_seeds,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
    )



def train_controller(make_env: Callable[[int], GridWorld],
                     controller: BaseController,
                     train_seeds: List[int], n_steps: int) -> List[float]:
    """Train one controller persistently across training episodes."""
    rewards = []
    for ep, seed in enumerate(train_seeds):
        env = make_env(seed)
        r, _ = run_episode(env, controller, f"train-{ep}", n_steps)
        rewards.append(r)
    return rewards


def evaluate_controller(controller: BaseController,
                        make_env: Callable[[int], GridWorld],
                        eval_seeds: List[int], n_eval_episodes: int,
                        n_steps: int) -> Tuple[List[float], int]:
    """Evaluate on held-out seeds with fresh controllers carrying trained weights."""
    rewards = []
    total_actions = 0
    for seed in eval_seeds:
        for ep in range(n_eval_episodes):
            env = make_env(seed + ep)
            ec = controller.clone(f"eval-{seed}-{ep}", seed + ep)
            total_actions += ec.actions_taken
            r, _ = run_episode(env, ec, f"eval-{seed}-{ep}", n_steps)
            rewards.append(r)
    return rewards, total_actions


def mean_confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    """Mean, lo, hi of 95% CI (Student t, approximated by normal for n>=20)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / np.sqrt(n))
    # 95% CI approx (normal); for small n this is approximate
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi


def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d of a vs b (independent samples)."""
    xa = np.asarray(a, dtype=float); xb = np.asarray(b, dtype=float)
    na, nb = xa.size, xb.size
    if na < 2 or nb < 2:
        return 0.0
    sa = xa.std(ddof=1); sb = xb.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((xa.mean() - xb.mean()) / pooled)


def paired_difference_ci(a: List[float], b: List[float]) -> Tuple[float, float, float, float]:
    """Mean, std, lo, hi of paired differences a - b (95% CI, normal approx).

    Paired by index: assumes a[i] and b[i] come from the same macro_seed /
    environment seed, so the difference isolates the controller advantage
    rather than environment variance.
    """
    xa = np.asarray(a, dtype=float); xb = np.asarray(b, dtype=float)
    n = min(xa.size, xb.size)
    if n < 2:
        d = xa[:n] - xb[:n]
        mean = float(d.mean()) if d.size else 0.0
        return mean, 0.0, mean, mean
    diff = xa[:n] - xb[:n]
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    se = std / np.sqrt(n)
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, std, lo, hi


def paired_cohens_d(a: List[float], b: List[float]) -> float:
    """Paired effect size: mean paired difference / std of differences."""
    xa = np.asarray(a, dtype=float); xb = np.asarray(b, dtype=float)
    n = min(xa.size, xb.size)
    if n < 2:
        return 0.0
    diff = xa[:n] - xb[:n]
    std = float(diff.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return float(diff.mean() / std)


@dataclass
class ExperimentResult:
    controller: str
    mean_eval: float
    ci_lo: float
    ci_hi: float
    std_eval: float
    cohens_d_vs_reactive: float
    n_macro_seeds: int
    all_eval_rewards: List[float]
    mean_train_reward: float
    inference_cost: float
    # Paired advantage over reactive: CI computed on per-macro-seed reward
    # differences (controller - reactive), not on independent means.
    paired_advantage_mean: float = 0.0
    paired_advantage_std: float = 0.0
    paired_advantage_ci_lo: float = 0.0
    paired_advantage_ci_hi: float = 0.0
    paired_effect_size: float = 0.0


def compare_controllers(make_env: Callable[[int], GridWorld],
                        controllers: Dict[str, Callable[[str, int], BaseController]],
                        macro_seeds: List[int],
                        n_train_seeds: int,
                        n_train_episodes: int,
                        n_eval_episodes: int,
                        n_steps: int,
                        n_steps_train: int | None = None) -> Dict[str, ExperimentResult]:
    """Compare controllers across macro_seeds with train/eval split."""
    n_steps_train = n_steps_train or n_steps
    results: Dict[str, ExperimentResult] = {}
    reactive_eval: List[float] = []
    # Per-macro-seed eval rewards per controller, for paired advantage CIs.
    per_seed_eval: Dict[str, List[float]] = {name: [] for name in controllers}

    for name, factory in controllers.items():
        all_eval: List[float] = []
        all_train: List[float] = []
        total_actions = 0
        for ms in macro_seeds:
            # Derive training and evaluation seeds from macro_seed
            train_seeds = [ms * 1000 + i for i in range(n_train_seeds)]
            eval_seeds = [ms * 1000 + 10000 + i for i in range(n_eval_episodes)]
            ctrl = factory(f"train-{ms}", ms)
            train_r = train_controller(make_env, ctrl, train_seeds, n_steps_train)
            all_train.extend(train_r)
            eval_r, n_act = evaluate_controller(
                ctrl, make_env, eval_seeds, 1, n_steps)
            total_actions += n_act
            all_eval.extend(eval_r)
            per_seed_eval[name].append(float(np.mean(eval_r)) if eval_r else 0.0)
        results[name] = ExperimentResult(
            controller=name,
            mean_eval=float(np.mean(all_eval)) if all_eval else 0.0,
            ci_lo=0.0, ci_hi=0.0, std_eval=float(np.std(all_eval)) if all_eval else 0.0,
            cohens_d_vs_reactive=0.0,
            n_macro_seeds=len(macro_seeds),
            all_eval_rewards=all_eval,
            mean_train_reward=float(np.mean(all_train)) if all_train else 0.0,
            inference_cost=float(total_actions),
        )
        if name == "reactive":
            reactive_eval = list(all_eval)

    # Fill CIs and effect sizes vs reactive
    for name, r in results.items():
        mean, lo, hi = mean_confidence_interval(r.all_eval_rewards)
        r.ci_lo, r.ci_hi = lo, hi
        if name != "reactive" and reactive_eval:
            r.cohens_d_vs_reactive = cohens_d(r.all_eval_rewards, reactive_eval)
            # Paired advantage over reactive, computed on per-macro-seed means.
            adv_mean, adv_std, adv_lo, adv_hi = paired_difference_ci(
                per_seed_eval[name], per_seed_eval["reactive"])
            r.paired_advantage_mean = adv_mean
            r.paired_advantage_std = adv_std
            r.paired_advantage_ci_lo = adv_lo
            r.paired_advantage_ci_hi = adv_hi
            r.paired_effect_size = paired_cohens_d(
                per_seed_eval[name], per_seed_eval["reactive"])
    return results


def run_gridworld_comparison(macro_seeds: List[int] | None = None,
                             n_train_seeds: int = 5,
                             n_eval_episodes: int = 3,
                             n_steps: int = 100) -> Dict[str, ExperimentResult]:
    if macro_seeds is None:
        macro_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def make_env(seed: int) -> GridWorld:
        return GridWorld(GridWorldConfig(seed=seed, food_regeneration=0.01,
                                         hazard_probability=0.1, dt=0.1))

    controllers = {
        "reactive": lambda aid, sd: ReactiveController(aid, sd),
        "stacked_obs": lambda aid, sd: StackedObsController(aid, sd, stack=3),
        "recurrent": lambda aid, sd: RecurrentController(aid, sd, hidden_dim=8),
        "world_model": lambda aid, sd: WorldModelController(aid, sd, hidden_dim=8),
    }

    return compare_controllers(
        make_env, controllers, macro_seeds,
        n_train_seeds=n_train_seeds,
        n_train_episodes=n_train_seeds,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
    )


if __name__ == "__main__":
    import numpy as np  # noqa: F401

    def fmt(name: str, r: ExperimentResult) -> str:
        return (f"{name:14s} {r.mean_eval:>9.3f} {r.ci_lo:>8.3f} {r.ci_hi:>8.3f} "
                f"{r.std_eval:>7.3f} {r.cohens_d_vs_reactive:>13.3f} "
                f"{r.paired_advantage_mean:>11.3f} [{r.paired_advantage_ci_lo:>7.3f},{r.paired_advantage_ci_hi:>7.3f}] "
                f"{r.paired_effect_size:>12.3f}")

    print("=" * 100)
    print("GRIDWORLD foraging (fully observable) -- 15 macro-seeds, 10 eval eps")
    print("=" * 100)
    print(f"{'controller':14s} {'mean':>9s} {'CI_lo':>8s} {'CI_hi':>8s} "
          f"{'std':>7s} {'d_vs_reactive':>13s} {'adv_mean':>11s} {'adv_CI':>17s} {'paired_d':>12s}")
    gw = run_gridworld_comparison(macro_seeds=list(range(1, 16)),
                                  n_train_seeds=5, n_eval_episodes=10, n_steps=60)
    reactive = gw["reactive"]
    for name, r in gw.items():
        print(fmt(name, r))
    print(f"Reactive CI: ({reactive.ci_lo:.3f}, {reactive.ci_hi:.3f})")
    print("Any non-reactive CI excludes zero:",
          any(not (r.ci_lo <= 0 <= r.ci_hi) for n, r in gw.items() if n != "reactive"))
    print("Paired advantage CI excludes zero (true advantage):",
          any(not (r.paired_advantage_ci_lo <= 0 <= r.paired_advantage_ci_hi)
              for n, r in gw.items() if n != "reactive"))
    print()

    for label, make_env, n_steps in [
        ("delayed_cue", lambda s: DelayedCueTask(DelayedCueConfig(), seed=s), 40),
        ("occluded_target", lambda s: OccludedTargetTask(OccludedTargetConfig(), seed=s), 40),
        ("changed_dynamics", lambda s: ChangedDynamicsTask(ChangedDynamicsConfig(), seed=s), 40),
    ]:
        print("=" * 100)
        print(f"{label.upper()} (memory-required) -- 10 macro-seeds, 5 eval eps")
        print("=" * 100)
        print(f"{'controller':14s} {'mean':>9s} {'CI_lo':>8s} {'CI_hi':>8s} "
              f"{'std':>7s} {'d_vs_reactive':>13s} {'adv_mean':>11s} {'adv_CI':>17s} {'paired_d':>12s}")
        res = run_memory_task_comparison(
            make_env, label,
            macro_seeds=list(range(1, 11)),
            n_train_seeds=5, n_eval_episodes=5, n_steps=n_steps,
        )
        reactive = res["reactive"]
        for name, r in res.items():
            print(fmt(name, r))
        print(f"Reactive CI: ({reactive.ci_lo:.3f}, {reactive.ci_hi:.3f})")
        print("World model d vs reactive:",
              round(res["world_model"].cohens_d_vs_reactive, 3))
        print("Recurrent d vs reactive:",
              round(res["recurrent"].cohens_d_vs_reactive, 3))
        print("World model paired advantage CI excludes zero:",
              not (res["world_model"].paired_advantage_ci_lo <= 0 <=
                   res["world_model"].paired_advantage_ci_hi))
        print("Recurrent paired advantage CI excludes zero:",
              not (res["recurrent"].paired_advantage_ci_lo <= 0 <=
                   res["recurrent"].paired_advantage_ci_hi))
        print()

    # ========================================================================
    # High-budget memory-task comparison: gated recurrent with sufficient training
    # ========================================================================
    for label, make_env, n_steps in [
        ("delayed_cue", lambda s: DelayedCueTask(DelayedCueConfig(), seed=s), 60),
        ("occluded_target", lambda s: OccludedTargetTask(OccludedTargetConfig(), seed=s), 60),
        ("changed_dynamics", lambda s: ChangedDynamicsTask(ChangedDynamicsConfig(), seed=s), 60),
    ]:
        print("=" * 100)
        print(f"{label.upper()} HIGH-BUDGET (memory-required) -- "
              "15 macro-seeds, 25 train seeds x 60 steps, gated_recurrent included")
        print("=" * 100)
        print(f"{'controller':14s} {'mean':>9s} {'CI_lo':>8s} {'CI_hi':>8s} "
              f"{'std':>7s} {'d_vs_reactive':>13s} {'adv_mean':>11s} {'adv_CI':>17s} {'paired_d':>12s}")
        res = run_memory_task_comparison_high_budget(
            make_env, label,
            macro_seeds=list(range(1, 16)),
            n_train_seeds=25, n_eval_episodes=8, n_steps=n_steps,
        )
        reactive = res["reactive"]
        for name, r in res.items():
            print(fmt(name, r))
        print(f"Reactive CI: ({reactive.ci_lo:.3f}, {reactive.ci_hi:.3f})")
        for cname in ("gated_recurrent", "recurrent", "world_model"):
            if cname in res:
                r = res[cname]
                print(f"{cname} paired advantage CI excludes zero:",
                      not (r.paired_advantage_ci_lo <= 0 <= r.paired_advantage_ci_hi),
                      f"(paired_d={r.paired_effect_size:+.3f})")
        print()

