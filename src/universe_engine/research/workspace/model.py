"""Global workspace module (Research Program D6).

Implements a limited global workspace that selects information from
competing modules (perception, memory, planning, action) and broadcasts it
to the rest of the system. The workspace implements a bottleneck: only a
small amount of information can be broadcast at any given time.

Provenance: global workspace is a computational architecture inspired by
Global Neuronal Workspace Theory, implemented as an attentional bottleneck.
Not a claim about consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class WorkspaceContent:
    """Information currently held in the global workspace."""
    modality: str               # "perception" | "memory" | "plan" | "action" | "self-model"
    content: Dict                 # the information being broadcast
    salience: float             # 0..1, how important this content is
    source_module: str          # which module sent it
    timestamp: float = 0.0


@dataclass
class GlobalWorkspace:
    """Limited-capacity global workspace.

    Competing modules send content to the workspace. The workspace selects
    the most salient content and broadcasts it to all modules. Only one
    content item can be broadcast at a time (the bottleneck).

    Parameters
    ----------
    capacity : int
        Maximum number of content items held simultaneously (default 1,
        representing the attentional bottleneck).
    decay : float
        How quickly content salience decays (0..1).
    """

    capacity: int = 1
    decay: float = 0.1
    contents: List[WorkspaceContent] = field(default_factory=list)
    broadcast_history: List[Dict] = field(default_factory=list)
    n_broadcasts: int = 0

    def receive(self, modality: str, content: Dict, salience: float,
                source_module: str, timestamp: float = 0.0) -> None:
        """A module sends content to the workspace."""
        wc = WorkspaceContent(
            modality=modality, content=content, salience=float(salience),
            source_module=source_module, timestamp=timestamp,
        )
        self.contents.append(wc)
        # Keep only top-capacity by salience
        self.contents.sort(key=lambda c: c.salience, reverse=True)
        self.contents = self.contents[:self.capacity]

    def broadcast(self, timestamp: float = 0.0) -> Optional[WorkspaceContent]:
        """Select and broadcast the most salient content.

        Returns the broadcast content or None if workspace is empty.
        After broadcasting, decays all content salience.
        """
        if not self.contents:
            return None
        selected = self.contents[0]
        self.n_broadcasts += 1
        self.broadcast_history.append({
            "modality": selected.modality,
            "source_module": selected.source_module,
            "salience": selected.salience,
            "timestamp": timestamp,
        })
        if len(self.broadcast_history) > 1000:
            self.broadcast_history = self.broadcast_history[-1000:]
        # Decay remaining contents
        for c in self.contents:
            c.salience = float(np.clip(c.salience * (1.0 - self.decay), 0.0, 1.0))
        self.contents = [c for c in self.contents if c.salience > 1e-4]
        return selected

    def clear(self) -> None:
        """Clear all workspace contents."""
        self.contents = []

    def workspace_occupancy(self) -> float:
        """Fraction of workspace capacity occupied."""
        return len(self.contents) / self.capacity if self.capacity > 0 else 0.0

    def top_salience(self) -> float:
        """Salience of the top content, or 0 if empty."""
        return self.contents[0].salience if self.contents else 0.0

    def checkpoint(self) -> dict:
        return {
            "capacity": self.capacity,
            "decay": self.decay,
            "n_broadcasts": self.n_broadcasts,
            "contents": [
                {"modality": c.modality, "content": c.content,
                 "salience": c.salience, "source_module": c.source_module,
                 "timestamp": c.timestamp}
                for c in self.contents
            ],
            "broadcast_history": self.broadcast_history[-50:],
        }

    @classmethod
    def restore(cls, data: dict) -> "GlobalWorkspace":
        gw = cls(capacity=int(data.get("capacity", 1)),
                 decay=float(data.get("decay", 0.1)))
        gw.contents = [
            WorkspaceContent(
                modality=c["modality"], content=c["content"],
                salience=float(c["salience"]), source_module=c["source_module"],
                timestamp=float(c.get("timestamp", 0.0)),
            ) for c in data.get("contents", [])
        ]
        gw.n_broadcasts = int(data.get("n_broadcasts", 0))
        gw.broadcast_history = list(data.get("broadcast_history", []))
        return gw


class WorkspaceAgentController:
    """A controller that uses a global workspace to coordinate action selection.

    Architecture D (from D6): Recurrent predictive controller with persistent
    self-model and workspace broadcasting.

    The controller:
    1. Receives sensory observation
    2. Sends perception content to workspace
    3. Sends self-model state to workspace
    4. Broadcasts the most salient content
    5. Selects action based on broadcast content + internal state
    6. Learns from the outcome
    """

    def __init__(self, agent_id: str, seed: int = 0,
                 workspace_capacity: int = 1, lr: float = 0.01):
        from ..self_model.model import make_default_self_model
        self.agent_id = agent_id
        self.controller_type = "ecs_iw"
        self.rng = np.random.default_rng(seed)
        self.lr = lr

        # Components
        self.workspace = GlobalWorkspace(capacity=workspace_capacity)
        self.self_model = make_default_self_model()
        self.hidden = np.zeros(8)  # recurrent hidden state
        self.W_h = self.rng.standard_normal((8, 8)) * 0.1
        self.W_in = self.rng.standard_normal((8, 6)) * 0.1   # obs input
        self.W_ws = self.rng.standard_normal((8, 4)) * 0.1   # workspace broadcast
        self.W_out = self.rng.standard_normal((4, 8)) * 0.1  # hidden -> action logits
        self.actions_taken = 0

    def _obs_vector(self, obs) -> np.ndarray:
        return np.array([
            obs.energy_level, obs.nutrient_level / 10.0,
            obs.nearby_resources, obs.nearby_hazards,
            float(obs.nearby_agents) / 5.0, obs.local_light,
        ])

    def _action_logits(self, hidden: np.ndarray) -> np.ndarray:
        return self.W_out @ hidden

    def _select_action_from_logits(self, logits: np.ndarray) -> Tuple[str, Dict]:
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

    def select_action(self, obs) -> object:
        """Select action using the ECS-IW architecture.

        1. Encode observation into hidden state
        2. Send perception + self-model content to workspace
        3. Broadcast most salient content
        4. Incorporate broadcast into hidden state
        5. Select action from action logits
        """
        from ..contracts import AgentAction

        self.actions_taken += 1
        obs_vec = self._obs_vector(obs)

        # Step 1: Update recurrent hidden state from observation
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_in @ obs_vec)

        # Step 2: Send perception content to workspace
        perception_content = {
            "energy_level": obs.energy_level,
            "nearby_resources": obs.nearby_resources,
            "nearby_hazards": obs.nearby_hazards,
            "position": obs.position,
        }
        perception_salience = 0.5 + 0.3 * obs.nearby_hazards + 0.2 * obs.nearby_resources
        self.workspace.receive("perception", perception_content, perception_salience,
                               "perception_module", obs.time)

        # Step 3: Send self-model content to workspace
        sm_diag = {
            "calibration_error": self.self_model.calibration_error,
            "self_caused_fraction": 0.5,  # placeholder, updated after action
        }
        sm_salience = 0.3 + 0.4 * self.self_model.calibration_error
        self.workspace.receive("self_model", sm_diag, sm_salience,
                               "self_model_module", obs.time)

        # Step 4: Broadcast
        broadcast = self.workspace.broadcast(obs.time)

        # Step 5: Incorporate broadcast into hidden state
        if broadcast is not None:
            ws_vec = np.zeros(4)
            if broadcast.modality == "perception":
                ws_vec[0] = broadcast.content.get("nearby_resources", 0.0)
                ws_vec[1] = broadcast.content.get("nearby_hazards", 0.0)
            elif broadcast.modality == "self_model":
                ws_vec[2] = broadcast.content.get("calibration_error", 0.0)
            self.hidden = np.tanh(self.hidden + self.W_ws @ ws_vec)

        # Step 6: Select action
        logits = self._action_logits(self.hidden)
        action_type, params = self._select_action_from_logits(logits)

        # Step 7: Use self-model to predict sensory consequence
        predicted = self.self_model.predict(action_type, params)

        return AgentAction(action_type, params, timestamp=obs.time,
                           confidence=0.5, provenance="ecs_iw")

    def learn(self, obs, action, reward, next_obs) -> None:
        """Learn from experience: update hidden state, self-model, and
        workspace content selection.

        Updates:
        - Recurrent weights (simple gradient on reward)
        - Self-model (compares predicted vs actual sensory change)
        - Workspace (no direct learning, but content selection is shaped by
          salience)
        """
        from ..contracts import AgentAction

        obs_vec = self._obs_vector(obs)
        next_obs_vec = self._obs_vector(next_obs)

        # Simple reward-based weight update (policy gradient style)
        logits = self._action_logits(self.hidden)
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        action_types = ["move", "forage", "rest", "wait"]
        action_idx = action_types.index(action.action_type)
        grad = np.zeros(4)
        grad[action_idx] = reward - 0.5  # baseline
        self.W_out += self.lr * np.outer(grad, self.hidden)

        # Update self-model: compare predicted vs actual sensory change
        # Actual change = difference in key observation features
        actual_change = np.array([
            next_obs.nearby_resources - obs.nearby_resources,
            next_obs.nearby_hazards - obs.nearby_hazards,
        ])
        predicted = self.self_model.predict(action.action_type, action.parameters)
        # Only compare first 2 dimensions (resources, hazards)
        sm_diag = self.self_model.update(
            action.action_type, action.parameters, predicted[:2], actual_change[:2]
        )

        # Update hidden state with next observation (teacher forcing)
        self.hidden = np.tanh(self.W_h @ self.hidden + self.W_in @ next_obs_vec)

    def checkpoint(self) -> dict:
        return {
            "controller_type": "ecs_iw",
            "hidden": self.hidden.tolist(),
            "W_h": self.W_h.tolist(), "W_in": self.W_in.tolist(),
            "W_ws": self.W_ws.tolist(), "W_out": self.W_out.tolist(),
            "actions_taken": self.actions_taken,
            "workspace": self.workspace.checkpoint(),
            "self_model": self.self_model.checkpoint(),
        }

    def restore(self, data: dict) -> None:
        self.hidden = np.array(data.get("hidden", np.zeros(8)))
        self.W_h = np.array(data.get("W_h", self.W_h))
        self.W_in = np.array(data.get("W_in", self.W_in))
        self.W_ws = np.array(data.get("W_ws", self.W_ws))
        self.W_out = np.array(data.get("W_out", self.W_out))
        self.actions_taken = int(data.get("actions_taken", 0))
        self.workspace = GlobalWorkspace.restore(data.get("workspace", {}))
        from ..self_model.model import SelfModel
        self.self_model = SelfModel.restore(data.get("self_model", {}))
