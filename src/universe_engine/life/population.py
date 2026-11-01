"""Population: a collection of persistent Organism entities.

Handles reproduction (with heredity and mutation), death, resource-limited
growth, and basic accounting. The Population does not decide fitness -- that is
the environment's job via the MetabolicNetwork and EcologicalNetwork. It
enforces conservation: reproduction costs real material/energy drawn from each
parent and from a shared resource pool; death returns material through
decomposition.

Provenance: population dynamics are a bounded agent-based abstraction, not a
claim about any specific species.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .genome import Genome
from .genetics import Genetics, GeneticOperators
from .development import DevelopmentalProgram, Phenotype, TraitValue
from .organism import Organism, OrganismState, MemoryEntry


@dataclass
class PopulationStats:
    """Snapshot summary of a population."""
    size: int
    total_energy: float
    total_nutrients: float
    mean_age: float
    mean_stress: float
    n_species: int
    n_lineages: int
    births: int
    deaths: int
    generation_mean: float


@dataclass
class Population:
    """A collection of organisms sharing a resource pool and environment.

    Parameters
    ----------
    organisms:
        identity -> Organism.
    resource_pool:
        shared free nutrient/energy units available to the population.
    developmental:
        DevelopmentalProgram used to turn offspring genomes into phenotypes.
    genetics:
        GeneticOperators for mutation/recombination during reproduction.
    rng:
        random number generator.
    metadata:
        provenance and configuration.
    """

    organisms: Dict[str, Organism] = field(default_factory=dict)
    resource_pool: float = 0.0
    developmental: DevelopmentalProgram = field(default_factory=DevelopmentalProgram)
    genetics: Genetics = field(default_factory=Genetics)
    rng: Optional[np.random.Generator] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    # Counters
    births: int = 0
    deaths: int = 0
    total_births: int = 0
    total_deaths: int = 0
    next_id: int = 0

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(0)
        if not self.organisms and self.next_id == 0:
            self.next_id = 0

    # ---- identity ----------------------------------------------------------
    def _new_id(self) -> str:
        nid = self.next_id
        self.next_id += 1
        return f"org-{nid}"

    # ---- add / remove ------------------------------------------------------
    def add(self, organism: Organism) -> None:
        self.organisms[organism.identity] = organism

    def remove(self, identity: str) -> Optional[Organism]:
        return self.organisms.pop(identity, None)

    def get(self, identity: str) -> Optional[Organism]:
        return self.organisms.get(identity)

    def alive(self) -> List[Organism]:
        return [o for o in self.organisms.values() if o.state.alive]

    def size(self) -> int:
        return sum(1 for o in self.organisms.values() if o.state.alive)

    # ---- reproduction ------------------------------------------------------
    def reproduce(self, parent: Organism, dt: float,
                  mutation_rate_scale: float = 1.0,
                  energy_cost_share: float = 0.5) -> Tuple[Optional[Organism], List]:
        """Create one child organism from a parent.

        Costs energy from parent and resource_pool. Returns (child, events).
        Returns (None, []) if reproduction is not possible (insufficient
        resources or no reproductive trait).
        """
        events = []
        if not parent.state.alive:
            return None, events
        if parent.phenotype.reproduction_rate <= 0:
            return None, events
        cost = parent.phenotype.developmental_cost
        share = cost * energy_cost_share
        if parent.state.energy < share:
            return None, events
        if self.resource_pool < share:
            return None, events

        # Pay the cost.
        parent.state.energy -= share
        self.resource_pool -= share
        events.append({"kind": "reproduction_cost", "amount": share,
                       "parent": parent.identity, "time": parent.state.age})

        # Mutate the genome.
        child_genome = parent.genome.copy()
        ops = GeneticOperators(self.rng, self.genetics)
        child_genome, evs = ops.substitute(child_genome)
        for e in evs:
            events.append({"kind": "mutation", "operator": e.operator,
                           "position": e.position, "detail": e.detail,
                           "cost_delta": e.cost_delta})

        # Develop child phenotype.
        child_phen = self.developmental.develop(child_genome)
        child_energy = child_phen.developmental_cost  # child born with build cost energy
        if self.resource_pool < child_energy:
            # Refund parent; cannot afford child.
            parent.state.energy += share
            self.resource_pool += share
            return None, events
        self.resource_pool -= child_energy
        parent.reproduction_count += 1
        parent.offspring.append("placeholder")  # filled by caller with child id

        child = Organism(
            identity=self._new_id(),
            genome=child_genome,
            phenotype=child_phen,
            state=OrganismState(energy=child_energy, nutrients=share,
                                age=0.0),
            location=parent.location,
            parent=parent.identity,
            generation=parent.generation + 1,
            species_id=parent.species_id,
        )
        child.offspring = []
        self.add(child)
        self.births += 1
        self.total_births += 1
        parent.offspring[-1] = child.identity
        events.append({"kind": "birth", "child": child.identity,
                       "parent": parent.identity, "time": parent.state.age})
        return child, events

    # ---- death -------------------------------------------------------------
    def kill(self, identity: str, reason: str = "death",
             return_material: float = 0.0) -> Optional[Organism]:
        """Mark an organism dead and return its stored material to the pool."""
        org = self.organisms.get(identity)
        if org is None or not org.state.alive:
            return None
        org.state.alive = False
        org.state.stress = 1.0
        self.deaths += 1
        self.total_deaths += 1
        # Decomposition: return stored nutrients+energy to resource pool.
        reclaimed = return_material + org.state.nutrients + 0.5 * org.state.energy
        self.resource_pool += reclaimed
        org.state.energy = 0.0
        org.state.nutrients = 0.0
        return org

    # ---- step --------------------------------------------------------------
    def step(self, dt: float, energy_income: float = 0.0) -> List:
        """Advance all organisms by dt: maintenance, ageing, death.

        energy_income: energy added to the shared resource_pool this step
        (e.g. from environmental production). Returns events.
        """
        events = []
        self.resource_pool += energy_income
        for org in list(self.organisms.values()):
            if not org.state.alive:
                continue
            cost = org.maintenance_cost(dt)
            org.state.energy -= cost
            org.state.age += dt
            org.state.lifetime += dt
            # Stress accumulates when energy is low.
            if org.state.energy < 0:
                org.state.stress = min(1.0, org.state.stress + 0.1 * dt)
            if org.state.energy <= 0 or org.state.stress >= 1.0:
                self.kill(org.identity, reason="starvation" if org.state.energy <= 0 else "senescence")
                events.append({"kind": "death", "organism": org.identity,
                               "reason": org.state.stress >= 1.0 and "senescence" or "starvation",
                               "time": org.state.age})
        return events

    # ---- stats -------------------------------------------------------------
    def stats(self) -> PopulationStats:
        live = self.alive()
        if not live:
            return PopulationStats(0, 0.0, 0.0, 0.0, 0.0, 0, 0,
                                   self.births, self.deaths, 0.0)
        total_energy = sum(o.state.energy for o in live)
        total_nutrients = sum(o.state.nutrients for o in live)
        mean_age = sum(o.state.age for o in live) / len(live)
        mean_stress = sum(o.state.stress for o in live) / len(live)
        species = {o.species_id for o in live if o.species_id is not None}
        lineages = {o.genome for o in live}
        return PopulationStats(size=len(live), total_energy=total_energy,
                               total_nutrients=total_nutrients, mean_age=mean_age,
                               mean_stress=mean_stress, n_species=len(species),
                               n_lineages=len(lineages), births=self.births,
                               deaths=self.deaths, generation_mean=
                               sum(o.generation for o in live) / len(live))

    def checkpoint(self) -> dict:
        return {
            "resource_pool": self.resource_pool,
            "births": self.births, "deaths": self.deaths,
            "total_births": self.total_births, "total_deaths": self.total_deaths,
            "next_id": self.next_id,
            "organisms": {i: o.checkpoint() for i, o in self.organisms.items()},
        }

    @classmethod
    def restore(cls, data: dict, developmental: DevelopmentalProgram,
                genetics: Genetics, rng_seed: int = 0) -> "Population":
        pop = cls(resource_pool=float(data.get("resource_pool", 0.0)),
                  developmental=developmental, genetics=genetics,
                  rng=np.random.default_rng(rng_seed),
                  births=int(data.get("births", 0)), deaths=int(data.get("deaths", 0)),
                  total_births=int(data.get("total_births", 0)),
                  total_deaths=int(data.get("total_deaths", 0)),
                  next_id=int(data.get("next_id", 0)))
        for i, od in data.get("organisms", {}).items():
            gmods = tuple(od["genome_modules"])
            genome = Genome(modules=gmods,
                            regulatory={int(k): v for k, v in od.get("genome_regulatory", {}).items()},
                            metadata=od.get("genome_metadata", {}))
            phen = Phenotype(
                traits={k: TraitValue(name=k, value=v["value"], cost=v["cost"],
                                      modules=tuple(v.get("modules", [])))
                        for k, v in od.get("phenotype_traits", {}).items()},
                size=float(od.get("phenotype_size", 1.0)),
                motility=float(od.get("phenotype_motility", 0.0)),
                sensing_range=float(od.get("phenotype_sensing_range", 0.0)),
                reproduction_rate=float(od.get("phenotype_reproduction_rate", 0.0)),
                defence=float(od.get("phenotype_defence", 0.0)),
                metabolic_power=float(od.get("phenotype_metabolic_power", 0.0)),
                structure=od.get("phenotype_structure", {}),
                developmental_cost=float(od.get("phenotype_developmental_cost", 0.0)),
            )
            state = OrganismState(energy=float(od.get("state_energy", 0.0)),
                                  nutrients=float(od.get("state_nutrients", 0.0)),
                                  age=float(od.get("state_age", 0.0)),
                                  stress=float(od.get("state_stress", 0.0)),
                                  alive=bool(od.get("state_alive", True)))
            pop.add(Organism.restore(od, genome, phen, state))
        return pop
