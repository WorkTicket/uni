"""Evolutionary architecture search (Sprint 6, Research Program C6).

Allows candidate controllers and sensory/developmental traits to vary through
constrained evolutionary experiments. Maintains an archive of diverse
evolutionary lineages and measures novelty, diversity, and lineage histories.

No organism is rewarded for becoming humanoid. Intelligence levels are not
assigned by elapsed time. Mutations do not automatically improve organisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from ..environment.gridworld import (
    GridWorld, GridWorldConfig, ReactiveController, RecurrentWorldModelController,
)
from ..workspace.model import WorkspaceAgentController
from ..contracts import AgentAction, SensoryObservation, ExperimentManifest


@dataclass
class AgentGenome:
    """Heritable genotype encoding controller architecture and sensory traits.

    Modules:
    - "R": reactive controller
    - "W": recurrent world-model controller
    - "E": ECS-IW (self-model + workspace) controller
    - "S": enhanced sensing range
    - "M": metabolic efficiency
    - "D": defensive capability
    - "N": neural/controller complexity
    """
    modules: Tuple[str, ...]
    regulatory: Dict[int, float] = field(default_factory=dict)
    mutation_rate: float = 1e-3
    metadata: Dict[str, object] = field(default_factory=dict)

    ALPHABET = frozenset({"R", "W", "E", "S", "M", "D", "N"})

    def __post_init__(self) -> None:
        for m in self.modules:
            if m not in self.ALPHABET:
                raise ValueError(f"unknown module {m!r}; allowed {sorted(self.ALPHABET)}")

    @classmethod
    def random(cls, rng: np.random.Generator, min_len: int = 1, max_len: int = 6) -> "AgentGenome":
        n = int(rng.integers(min_len, max_len + 1))
        modules = tuple(rng.choice(sorted(cls.ALPHABET), size=n).tolist())
        reg = {i: float(rng.uniform(0.3, 1.0)) for i in range(n) if rng.random() < 0.4}
        return cls(modules=modules, regulatory=reg,
                   metadata={"origin": "random"})

    def copy(self) -> "AgentGenome":
        return AgentGenome(modules=tuple(self.modules),
                           regulatory=dict(self.regulatory),
                           mutation_rate=float(self.mutation_rate),
                           metadata=dict(self.metadata))

    def __hash__(self) -> int:
        return hash((self.modules, tuple(sorted(self.regulatory.items()))))


@dataclass
class AgentPhenotype:
    """Expressed traits developed from genome via genotype-to-phenotype map.

    Each trait carries a maintenance cost. No trait is inherently "good" --
    fitness emerges from the phenotype-environment relationship.
    """
    controller_type: str        # "reactive" | "recurrent" | "ecs_iw"
    sensing_range: int          # patches in each direction
    metabolic_efficiency: float # energy gain multiplier
    defence: float              # hazard resistance
    controller_complexity: float # hidden dimension size
    maintenance_cost: float     # per-step energy cost
    size: float                 # body size

    @classmethod
    def develop(cls, genome: AgentGenome) -> "AgentPhenotype":
        """Genotype-to-phenotype mapping (declared developmental rules)."""
        mod_counts: Dict[str, int] = {m: 0 for m in genome.ALPHABET}
        for m in genome.modules:
            mod_counts[m] += 1

        # Base traits
        controller_type = "reactive"
        if "E" in genome.modules:
            controller_type = "ecs_iw"
        elif "W" in genome.modules:
            controller_type = "recurrent"

        sensing_range = 1 + mod_counts["S"]
        metabolic_efficiency = 1.0 + 0.2 * mod_counts["M"]
        defence = 0.1 * mod_counts["D"]
        controller_complexity = 4.0 + 4.0 * mod_counts["N"]

        # Costs emerge from traits
        maintenance_cost = (
            0.01
            + 0.005 * sensing_range
            + 0.01 * defence
            + 0.002 * controller_complexity
            + (0.02 if controller_type == "ecs_iw" else 0.0)
            + (0.01 if controller_type == "recurrent" else 0.0)
        )
        size = 1.0 + 0.1 * len(genome.modules)

        return AgentPhenotype(
            controller_type=controller_type,
            sensing_range=sensing_range,
            metabolic_efficiency=metabolic_efficiency,
            defence=defence,
            controller_complexity=controller_complexity,
            maintenance_cost=maintenance_cost,
            size=size,
        )


@dataclass
class EvolvableAgent:
    """One agent in the evolutionary archive with genome, phenotype, controller."""
    agent_id: str
    genome: AgentGenome
    phenotype: AgentPhenotype
    controller: object
    birth_time: float
    generation: int
    parent_id: Optional[str]
    lifetime_reward: float = 0.0
    n_steps: int = 0
    alive: bool = True
    extinction_time: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def fitness(self) -> float:
        """Lifetime reproductive success proxy: reward per step while alive."""
        if self.n_steps == 0:
            return 0.0
        return self.lifetime_reward / self.n_steps


def make_controller(genome: AgentGenome, phenotype: AgentPhenotype,
                    seed: int = 0) -> object:
    """Factory: build a controller from genotype+phenotype."""
    rng = np.random.default_rng(seed)
    if phenotype.controller_type == "ecs_iw":
        return WorkspaceAgentController(
            agent_id="temp", seed=seed,
            workspace_capacity=1,
            lr=0.01,
        )
    elif phenotype.controller_type == "recurrent":
        return RecurrentWorldModelController(
            agent_id="temp", seed=seed,
            hidden_dim=int(phenotype.controller_complexity),
            lr=0.01,
        )
    else:
        return ReactiveController(agent_id="temp", seed=seed)


def mutate_genome(genome: AgentGenome, rng: np.random.Generator) -> AgentGenome:
    """Apply mutation operators. No mutation is beneficial by default."""
    if rng.random() > genome.mutation_rate:
        return genome.copy()

    modules = list(genome.modules)
    reg = dict(genome.regulatory)
    op = rng.choice(["substitute", "insert", "delete", "regulate"])

    if op == "substitute" and len(modules) > 0:
        pos = int(rng.integers(0, len(modules)))
        old = modules[pos]
        choices = [m for m in sorted(genome.ALPHABET) if m != old]
        modules[pos] = str(rng.choice(choices))
    elif op == "insert":
        pos = int(rng.integers(0, len(modules) + 1))
        new_mod = str(rng.choice(sorted(genome.ALPHABET)))
        modules.insert(pos, new_mod)
        # shift regulatory marks
        new_reg = {}
        for p, w in reg.items():
            new_reg[p + 1 if p >= pos else p] = w
        reg = new_reg
    elif op == "delete" and len(modules) > 1:
        pos = int(rng.integers(0, len(modules)))
        modules.pop(pos)
        new_reg = {}
        for p, w in reg.items():
            if p == pos:
                continue
            new_reg[p - 1 if p > pos else p] = w
        reg = new_reg
    elif op == "regulate" and len(modules) > 0:
        pos = int(rng.integers(0, len(modules)))
        old_w = reg.get(pos, 1.0)
        new_w = float(np.clip(old_w + rng.normal(0.0, 0.2), 0.05, 1.0))
        reg[pos] = new_w

    child = AgentGenome(
        modules=tuple(modules), regulatory=reg,
        mutation_rate=float(genome.mutation_rate),
        metadata=dict(genome.metadata, parent=genome.metadata.get("origin", "seed")),
    )
    return child


@dataclass
class LineageRecord:
    """Persistent lineage history for one genome."""
    genome_hash: int
    genome: AgentGenome
    parent_hash: Optional[int]
    birth_time: float
    generation: int
    n_descendants: int = 0
    max_fitness: float = 0.0
    extinction_time: Optional[float] = None


@dataclass
class EvolutionArchive:
    """Archive of diverse evolutionary lineages and candidate architectures.

    Maintains novelty measurements, lineage histories, and diversity metrics.
    No predetermined species progression.
    """
    lineages: Dict[int, LineageRecord] = field(default_factory=dict)
    population: Dict[str, EvolvableAgent] = field(default_factory=dict)
    n_generations: int = 0
    total_births: int = 0
    total_deaths: int = 0
    total_extinctions: int = 0

    def register_lineage(self, genome: AgentGenome, parent: Optional[AgentGenome],
                         generation: int, t: float) -> int:
        h = hash(genome)
        parent_hash = hash(parent) if parent is not None else None
        if h not in self.lineages:
            self.lineages[h] = LineageRecord(
                genome_hash=h, genome=genome, parent_hash=parent_hash,
                birth_time=t, generation=generation,
            )
        else:
            self.lineages[h].n_descendants += 1
            self.lineages[h].max_fitness = max(
                self.lineages[h].max_fitness,
                0.0,  # updated on death
            )
        return h

    def diversity(self) -> Dict[str, float]:
        """Diversity metrics: genome diversity, controller diversity, trait diversity."""
        if not self.lineages:
            return {"genome_diversity": 0.0, "controller_diversity": 0.0,
                    "trait_diversity": 0.0, "n_lineages": 0}
        controllers = set()
        sensing_ranges = set()
        for lr in self.lineages.values():
            phen = AgentPhenotype.develop(lr.genome)
            controllers.add(phen.controller_type)
            sensing_ranges.add(phen.sensing_range)
        n = len(self.lineages)
        return {
            "genome_diversity": float(n),
            "controller_diversity": float(len(controllers)),
            "trait_diversity": float(len(sensing_ranges)),
            "n_lineages": float(n),
        }

    def extinction(self, genome: AgentGenome, t: float) -> None:
        h = hash(genome)
        if h in self.lineages:
            self.lineages[h].extinction_time = t
        self.total_extinctions += 1


class EvolutionaryExperiment:
    """Constrained evolutionary experiment in GridWorld.

    Runs a population of evolvable agents over generations. Agents compete for
    limited resources (energy/food). Reproduction requires real energy. Death
    from starvation is possible. Lineages are tracked in an archive.

    The environment changes over time (food scarcity, hazard changes) to test
    adaptation under changing conditions.
    """

    def __init__(self, seed: int = 0, n_patches: int = 8,
                 n_founders: int = 10, dt: float = 0.1,
                 max_energy: float = 10.0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.max_energy = float(max_energy)
        self.t = 0.0
        self.generation = 0
        self.archive = EvolutionArchive()
        self.n_patches = int(n_patches)
        self.env = GridWorld(GridWorldConfig(
            width=n_patches, height=n_patches, seed=seed,
            food_regeneration=0.01, hazard_probability=0.1,
            dt=dt, max_energy=max_energy,
        ))
        self._founders: List[str] = []
        self._seed_founders(n_founders)

    def _seed_founders(self, n: int) -> None:
        for i in range(n):
            g = AgentGenome.random(self.rng)
            phen = AgentPhenotype.develop(g)
            ctrl = make_controller(g, phen, seed=int(self.rng.integers(0, 2**31)))
            aid = f"founder-{i}"
            self.env.register_agent(aid, (int(self.rng.integers(0, self.n_patches)),
                                          int(self.rng.integers(0, self.n_patches))),
                                     initial_energy=5.0)
            agent = EvolvableAgent(
                agent_id=aid, genome=g, phenotype=phen, controller=ctrl,
                birth_time=self.t, generation=0, parent_id=None,
            )
            self.archive.register_lineage(g, None, 0, self.t)
            self.archive.population[aid] = agent
            self._founders.append(aid)

    def step(self) -> Dict:
        """Advance one evolutionary step: act, metabolize, reproduce, die."""
        self.t += self.dt

        # 1. Each alive agent acts
        for aid, agent in list(self.archive.population.items()):
            if not agent.alive:
                continue
            obs = self.env.observe(aid)
            # Adjust sensory range based on phenotype
            obs = self._adjust_observation(obs, agent.phenotype)
            action = agent.controller.select_action(obs)
            result = self.env.act(aid, action)
            reward = float(result.get("reward", 0.0))
            next_obs = self.env.observe(aid)
            agent.controller.learn(obs, action, reward, next_obs)
            agent.lifetime_reward += reward
            agent.n_steps += 1

            # Metabolic cost
            state = self.env._agent_states[aid]
            state.energy -= agent.phenotype.maintenance_cost * self.dt
            state.energy += reward * agent.phenotype.metabolic_efficiency * self.dt
            state.energy = float(np.clip(state.energy, 0.0, self.max_energy))

            if state.energy <= 0:
                state.energy = 0.0
                agent.alive = False
                agent.extinction_time = self.t
                self.archive.total_deaths += 1
                self.archive.extinction(agent.genome, self.t)

        # 2. Environment step (food regeneration, hazard drift)
        self.env.step(self.dt)

        # 3. Reproduction: high-energy agents reproduce with mutation
        new_agents: Dict[str, EvolvableAgent] = {}
        for aid, agent in list(self.archive.population.items()):
            if not agent.alive:
                continue
            state = self.env._agent_states[aid]
            if state.energy < 2.0 * agent.phenotype.maintenance_cost:
                continue
            if self.rng.random() > 0.02:  # low reproduction probability per step
                continue

            # Cost of reproduction
            cost = agent.phenotype.maintenance_cost * 5.0 * self.dt
            if state.energy < cost:
                continue
            state.energy -= cost

            # Mutate genome
            child_genome = mutate_genome(agent.genome, self.rng)
            child_phen = AgentPhenotype.develop(child_genome)
            child_ctrl = make_controller(child_genome, child_phen,
                                         seed=int(self.rng.integers(0, 2**31)))
            child_aid = f"child-{self.archive.total_births}"
            # Place child near parent
            parent_pos = self.env._agent_positions[aid]
            child_pos = (
                int(parent_pos[0] + self.rng.integers(-1, 2)) % self.env.cfg.width,
                int(parent_pos[1] + self.rng.integers(-1, 2)) % self.env.cfg.height,
            )
            self.env.register_agent(child_aid, child_pos,
                                     initial_energy=2.0 * child_phen.maintenance_cost)
            child_agent = EvolvableAgent(
                agent_id=child_aid, genome=child_genome, phenotype=child_phen,
                controller=child_ctrl, birth_time=self.t,
                generation=agent.generation + 1, parent_id=aid,
            )
            self.archive.register_lineage(child_genome, agent.genome,
                                          child_agent.generation, self.t)
            self.archive.total_births += 1
            new_agents[child_aid] = child_agent

        self.archive.population.update(new_agents)

        # 4. Periodic environmental change (changing conditions)
        if int(self.t / 10.0) > int((self.t - self.dt) / 10.0):
            self._change_environment()

        self.generation += 1
        return self.stats()

    def _adjust_observation(self, obs: SensoryObservation,
                            phen: AgentPhenotype) -> SensoryObservation:
        """Adjust observation based on phenotype traits (e.g., sensing range)."""
        # Enhanced sensing gives better resource estimates
        if phen.sensing_range > 1:
            obs = SensoryObservation(
                time=obs.time, position=obs.position,
                energy_level=obs.energy_level, nutrient_level=obs.nutrient_level,
                local_temperature=obs.local_temperature, local_light=obs.local_light,
                nearby_resources=float(np.clip(obs.nearby_resources * 1.2, 0.0, 1.0)),
                nearby_hazards=obs.nearby_hazards,
                nearby_agents=obs.nearby_agents,
                social_signals=obs.social_signals,
                extra=obs.extra, noise_seed=obs.noise_seed,
                provenance=obs.provenance,
            )
        return obs

    def _change_environment(self) -> None:
        """Change environmental conditions to test adaptation."""
        # Vary food regeneration rate
        for p in self.env.patches.values():
            p.regeneration_rate = float(self.rng.uniform(0.005, 0.03))
        # Vary hazard levels
        for p in self.env.patches.values():
            if self.rng.random() < 0.1:
                p.hazard = 1.0 if p.hazard == 0.0 else 0.0

    def stats(self) -> Dict:
        alive = [a for a in self.archive.population.values() if a.alive]
        return {
            "t": self.t,
            "generation": self.generation,
            "n_alive": len(alive),
            "n_total": len(self.archive.population),
            "n_lineages": len(self.archive.lineages),
            "total_births": self.archive.total_births,
            "total_deaths": self.archive.total_deaths,
            "total_extinctions": self.archive.total_extinctions,
            "diversity": self.archive.diversity(),
            "mean_fitness": float(np.mean([a.fitness() for a in alive])) if alive else 0.0,
            "max_fitness": float(max([a.fitness() for a in alive])) if alive else 0.0,
        }

    def run(self, n_steps: int) -> List[Dict]:
        """Run n evolutionary steps, returning stats history."""
        history = []
        for _ in range(n_steps):
            history.append(self.step())
        return history

    def checkpoint(self) -> dict:
        return {
            "seed": self.seed, "t": self.t, "generation": self.generation,
            "dt": self.dt, "max_energy": self.max_energy,
            "archive": {
                "lineages": {str(k): {
                    "genome_hash": v.genome_hash,
                    "genome_modules": v.genome.modules,
                    "genome_regulatory": v.genome.regulatory,
                    "parent_hash": v.parent_hash,
                    "birth_time": v.birth_time,
                    "generation": v.generation,
                    "n_descendants": v.n_descendants,
                    "extinction_time": v.extinction_time,
                } for k, v in self.archive.lineages.items()},
                "population": {aid: {
                    "genome_modules": a.genome.modules,
                    "phenotype": {
                        "controller_type": a.phenotype.controller_type,
                        "sensing_range": a.phenotype.sensing_range,
                        "metabolic_efficiency": a.phenotype.metabolic_efficiency,
                        "defence": a.phenotype.defence,
                        "maintenance_cost": a.phenotype.maintenance_cost,
                    },
                    "birth_time": a.birth_time, "generation": a.generation,
                    "parent_id": a.parent_id, "alive": a.alive,
                    "lifetime_reward": a.lifetime_reward, "n_steps": a.n_steps,
                } for aid, a in self.archive.population.items()},
                "total_births": self.archive.total_births,
                "total_deaths": self.archive.total_deaths,
                "total_extinctions": self.archive.total_extinctions,
            },
        }
