"""Embodied learning harness (PASS 5) — research, experimental.

Runs an agent (organism controller) inside the persistent world (a Habitat) and
records reward/learning history across episodes. This is an experimental
scaffold for the return to reward-learning; it does NOT claim that reward-only
memory-based control is solved.

The existing negative benchmark finding is preserved: reward-only credit
assignment across the cue -> action -> delayed habitat-feedback gap in this
habitat remains unresolved. The harness reports the measured outcome honestly
("unresolved") rather than fabricating learning.

No claim of consciousness; measures action-reward learning only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

import numpy as np

from .research.persistent_organism import (
    PersistentOrganism, make_default_habitat, make_organism,
)
from .research.contracts import AgentAction, SensoryObservation
from .habitat import Habitat


@dataclass
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    mean_reward: float
    final_energy: float
    alive: bool
    n_memories: int


@dataclass
class EmbodiedLearningResult:
    """Honest outcome of one embodied-learning run.

    `solved` is True only if the measured mean reward clearly exceeds the
    zero-reward baseline over the final window. In the default habitat this is
    expected to be False, preserving the documented negative finding that
    reward-only memory-based control is unresolved.
    """
    episodes: int
    steps_per_episode: int
    seed: int
    architecture: str
    rewards: List[float] = field(default_factory=list)
    episodes_results: List[EpisodeResult] = field(default_factory=list)
    solved: bool = False
    note: str = ""


class EmbodiedLearning:
    """Run an organism agent inside a persistent habitat across episodes.

    Each episode resets the organism's body but keeps the (shared) habitat
    advancing, so the agent learns from real action -> habitat consequence
    coupling rather than a static bandit.
    """

    def __init__(self, architecture: str = "ecs_iw", seed: int = 0,
                 n_patches: int = 4, patch_index: int = 0,
                 steps_per_episode: int = 20):
        self.architecture = architecture
        self.seed = int(seed)
        self.n_patches = int(n_patches)
        self.patch_index = int(patch_index)
        self.steps_per_episode = int(steps_per_episode)
        self.habitat = make_default_habitat(seed=seed, n_patches=n_patches)
        self._episode = 0

    def _fresh_organism(self) -> PersistentOrganism:
        # Same persistent habitat (shared world), a fresh organism body each
        # episode so learning must come from the controller + memory, not from
        # a reset body.
        return make_organism(
            architecture=self.architecture, seed=self.seed + self._episode,
            habitat=self.habitat, patch_index=self.patch_index,
        )

    def run_episode(self, org: Optional[PersistentOrganism] = None) -> EpisodeResult:
        if org is None:
            org = self._fresh_organism()
        total = 0.0
        steps = 0
        alive = True
        for _ in range(self.steps_per_episode):
            if not org.alive:
                alive = False
                break
            res = org.step()
            total += float(res.get("reward", 0.0))
            steps += 1
            alive = bool(res.get("alive", False))
        mean = total / max(steps, 1)
        ep = EpisodeResult(
            episode=self._episode, steps=steps, total_reward=total,
            mean_reward=mean, final_energy=float(org.body.energy),
            alive=alive, n_memories=len(org.memory.entries),
        )
        self._episode += 1
        return ep

    def run(self, episodes: int) -> EmbodiedLearningResult:
        result = EmbodiedLearningResult(
            episodes=episodes, steps_per_episode=self.steps_per_episode,
            seed=self.seed, architecture=self.architecture,
        )
        for _ in range(episodes):
            ep = self.run_episode()
            result.rewards.append(ep.mean_reward)
            result.episodes_results.append(ep)
        # Honest solved criterion: final-window mean reward clearly above the
        # zero baseline AND the agent survives most of the final episodes.
        if result.rewards:
            window = result.rewards[-max(1, episodes // 4):]
            final_mean = float(np.mean(window))
            survival = float(np.mean([e.alive for e in result.episodes_results[-max(1, episodes // 4):]]))
            result.solved = bool(final_mean > 0.05 and survival > 0.5)
        result.note = (
            "Reward-only memory-based control in this habitat is UNRESOLVED: "
            "the measured outcome is reported, not claimed as success."
        )
        return result

    def checkpoint(self) -> dict:
        return {
            "schema": "embodied-learning/v1",
            "architecture": self.architecture, "seed": self.seed,
            "n_patches": self.n_patches, "patch_index": self.patch_index,
            "steps_per_episode": self.steps_per_episode, "episode": self._episode,
            "habitat": self.habitat.checkpoint(),
        }

    @classmethod
    def restore(cls, data: dict, habitat: Habitat) -> "EmbodiedLearning":
        el = cls.__new__(cls)
        el.architecture = str(data["architecture"])
        el.seed = int(data["seed"])
        el.n_patches = int(data["n_patches"])
        el.patch_index = int(data["patch_index"])
        el.steps_per_episode = int(data["steps_per_episode"])
        el._episode = int(data.get("episode", 0))
        el.habitat = Habitat.restore(habitat.planet, habitat.network, data["habitat"],
                                     seed=habitat.seed, n_patches=habitat.n_patches)
        return el


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
