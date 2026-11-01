"""CognitiveController: optional evolvable cognitive architecture.

Intelligence is not a universal property. Controllers range from stimulus-response
to associative learning, spatial memory, reward-based learning, short-horizon
planning, social learning, individual recognition and communication. Each level
carries a metabolic/developmental cost tracked in the organism's state.

No large language model is connected. No behaviour is interpreted as evidence of
subjective consciousness.

Provenance: computational-cognition abstractions (Q-learning, Hebbian association,
spatial maps) are declared inputs with bounded scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SensoryInput:
    """Perception available to one organism through its modeled senses."""
    time: float
    location: Tuple[float, ...]
    food_nearby: float         # 0..1 sensed food concentration
    danger_nearby: float       # 0..1 sensed threat (predator/toxin)
    conspecifics_nearby: int   # count of same-species individuals sensed
    mates_nearby: int
    light: float               # 0..1 ambient light
    temperature: float         # K
    objects: List[dict] = field(default_factory=list)  # nearby objects {kind, location, value}
    social_signals: List[str] = field(default_factory=list)  # signals heard


@dataclass
class CognitiveState:
    """Internal memory/model of an individual organism."""
    # Associative value table: cue -> value (reward prediction).
    values: Dict[str, float] = field(default_factory=dict)
    # Spatial memory: location key -> value (e.g. food at location).
    spatial: Dict[Tuple[float, ...], float] = field(default_factory=dict)
    # Individual recognition: id -> last-seen value/valence.
    individuals: Dict[str, float] = field(default_factory=dict)
    # Learned communication conventions: signal -> meaning.
    communication: Dict[str, str] = field(default_factory=dict)
    # Short-horizon plan: list of intended actions.
    plan: List[str] = field(default_factory=list)


@dataclass
class CognitiveController:
    """An evolvable cognitive architecture for one organism.

    architecture:
        "stimulus_response" | "habituation" | "associative" |
        "spatial_memory" | "reward_learning" | "planning" |
        "social_learning" | "communication"
    learning_rate:
        how fast associations update (0..1).
    discount:
        future-reward discount (0..1).
    exploration:
        probability of trying a random action (0..1).
    memory_capacity:
        max number of spatial/associative entries.
    cost_per_tick:
        metabolic cost per simulation tick for this architecture.
    """

    architecture: str = "stimulus_response"
    learning_rate: float = 0.1
    discount: float = 0.9
    exploration: float = 0.1
    memory_capacity: int = 64
    cost_per_tick: float = 0.0

    def __post_init__(self) -> None:
        costs = {
            "stimulus_response": 0.0,
            "habituation": 0.001,
            "associative": 0.005,
            "spatial_memory": 0.01,
            "reward_learning": 0.02,
            "planning": 0.04,
            "social_learning": 0.03,
            "communication": 0.03,
        }
        if self.cost_per_tick == 0.0:
            self.cost_per_tick = costs.get(self.architecture, 0.0)

    # ---- action selection --------------------------------------------------
    def choose_action(self, sensory: SensoryInput, state: CognitiveState,
                      actions: List[str], now: float = 0.0) -> str:
        """Select an action using the architecture's decision rule.

        Uses only `sensory` (what the organism can perceive) and `state` (its
        memory). Never global simulation state.
        """
        if not actions:
            return "wait"
        arch = self.architecture

        if arch in ("stimulus_response", "habituation"):
            # Simple reflex: move toward food, away from danger.
            if sensory.food_nearby > 0.5:
                return "forage"
            if sensory.danger_nearby > 0.5:
                return "avoid"
            if sensory.mates_nearby > 0:
                return "seek_mate"
            return "explore"

        if arch == "associative":
            # Value-based: pick action with highest learned value for current cue.
            cue = f"{sensory.food_nearby:.1f}-{sensory.danger_nearby:.1f}"
            best = max(actions, key=lambda a: state.values.get(f"{cue}:{a}", 0.0))
            if self.rng_random() < self.exploration:
                return self.rng_choice(actions)
            return best

        if arch == "spatial_memory":
            # Return to remembered good locations.
            best_loc = None; best_v = -float("inf")
            for loc, v in state.spatial.items():
                if v > best_v:
                    best_v = v; best_loc = loc
            if best_loc is not None and sensory.food_nearby < 0.5:
                return "navigate"
            if sensory.food_nearby > 0.5:
                return "forage"
            return "explore"

        if arch == "reward_learning":
            cue = f"{sensory.food_nearby:.1f}-{sensory.danger_nearby:.1f}"
            best = max(actions, key=lambda a: state.values.get(f"{cue}:{a}", 0.0))
            if self.rng_random() < self.exploration:
                return self.rng_choice(actions)
            return best

        if arch == "planning":
            if state.plan:
                return state.plan.pop(0)
            if sensory.food_nearby > 0.5:
                return "forage"
            return "explore"

        if arch in ("social_learning", "communication"):
            if sensory.social_signals:
                sig = sensory.social_signals[0]
                meaning = state.communication.get(sig)
                if meaning == "food" and sensory.food_nearby < 0.5:
                    return "forage"
                if meaning == "danger":
                    return "avoid"
            if sensory.food_nearby > 0.5:
                return "forage"
            return "explore"

        return actions[0]

    # ---- learning updates --------------------------------------------------
    def update(self, sensory: SensoryInput, state: CognitiveState, action: str,
               reward: float, now: float = 0.0) -> None:
        """Update internal model after taking `action` and receiving `reward`."""
        arch = self.architecture
        if arch in ("associative", "reward_learning"):
            cue = f"{sensory.food_nearby:.1f}-{sensory.danger_nearby:.1f}"
            key = f"{cue}:{action}"
            old = state.values.get(key, 0.0)
            # Q-learning style update: V <- V + lr*(reward + discount*max_next - V)
            next_max = max([state.values.get(f"{cue}:{a}", 0.0) for a in [action]] or [0.0])
            state.values[key] = old + self.learning_rate * (reward + self.discount * next_max - old)
        elif arch == "spatial_memory" and reward > 0:
            loc = tuple(round(float(x), 1) for x in sensory.location)
            state.spatial[loc] = state.spatial.get(loc, 0.0) + reward
            self._prune(state.spatial)
        elif arch == "social_learning" and sensory.social_signals:
            # Associate a signal with the experienced outcome.
            sig = sensory.social_signals[0]
            state.communication[sig] = "food" if reward > 0 else ("danger" if reward < 0 else "neutral")

    def _prune(self, d: dict) -> None:
        if len(d) > self.memory_capacity:
            # Drop lowest-value entries.
            items = sorted(d.items(), key=lambda kv: kv[1])
            d.clear()
            for k, v in items[-self.memory_capacity:]:
                d[k] = v

    # ---- RNG (bounded, deterministic) --------------------------------------
    def rng_random(self) -> float:
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(0)
        return float(self._rng.random())

    def rng_choice(self, seq: List[str]) -> str:
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(0)
        return str(self._rng.choice(seq))
