"""Individual organism: a persistent simulation entity.

An Organism owns a Genome, a developed Phenotype, and an internal
energy/resource state. It acts on its environment through a behavioural
controller (stimulus-response by default; cognitive controllers optional) and
stores limited memory of perceptions. It is *not* omniscient: it only knows what
its modeled senses and memory provide.

Provenance: organism is a software agent abstraction of a living cell/organism,
not a claim about any particular biology. Subjective experience is not modeled
or claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .genome import Genome
from .development import DevelopmentalProgram, Phenotype


@dataclass
class OrganismState:
    """Mutable physiological state of one organism."""
    energy: float            # stored metabolic energy (units)
    nutrients: float         # stored material/nutrient units
    age: float = 0.0         # time since birth (simulation time units)
    lifetime: float = 0.0    # total time alive
    stress: float = 0.0      # accumulated damage/stress (0..1)
    alive: bool = True
    lifespan_used: float = 0.0

    def clone(self) -> "OrganismState":
        return OrganismState(energy=self.energy, nutrients=self.nutrients,
                             age=self.age, lifetime=self.lifetime,
                             stress=self.stress, alive=self.alive,
                             lifespan_used=self.lifespan_used)


@dataclass
class MemoryEntry:
    """One remembered perception/event."""
    time: float
    kind: str                # "food", "danger", "conspecific", "mate", "territory", "object"
    location: Tuple[float, ...]
    value: float             # salience / valence (positive=beneficial, negative=harmful)
    detail: Dict[str, object] = field(default_factory=dict)


@dataclass
class Organism:
    """One persistent organism.

    identity:
        globally unique id (within a run).
    genome:
        inherited genotype.
    phenotype:
        developed from genome via the DevelopmentalProgram.
    state:
        energy/nutrients/age/stress.
    memory:
        finite ring of recent perceptions.
    location:
        position in the habitat grid (tuple of floats).
    parent:
        id of parent organism, or None for a founder.
    generation:
        generation number (founder = 0).
    """

    identity: str
    genome: Genome
    phenotype: Phenotype
    state: OrganismState
    memory: List[MemoryEntry] = field(default_factory=list)
    location: Tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0))
    parent: Optional[str] = None
    generation: int = 0
    species_id: Optional[str] = None
    reproduction_count: int = 0
    offspring: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    # ---- lifecycle ---------------------------------------------------------
    def birth_energy(self) -> float:
        """Initial energy = developmental cost of phenotype."""
        return self.phenotype.developmental_cost

    def maintenance_cost(self, dt: float) -> float:
        """Energy/nutrient cost to stay alive for dt (incl. phenotype upkeep)."""
        base = 0.01 * dt
        upkeep = self.phenotype.total_maintenance_cost() * dt
        size_cost = 0.001 * self.phenotype.size * dt
        cognitive_cost = self.metadata.get("cognitive_cost", 0.0) * dt
        return base + upkeep + size_cost + cognitive_cost

    def can_reproduce(self) -> bool:
        """Can reproduce if energy covers developmental cost of a child."""
        return self.state.alive and self.state.energy >= self.phenotype.developmental_cost

    def add_memory(self, entry: MemoryEntry, capacity: int = 64) -> None:
        self.memory.append(entry)
        if len(self.memory) > capacity:
            self.memory = self.memory[-capacity:]

    def recent_memory(self, kind: Optional[str] = None, window: Optional[float] = None,
                      now: float = 0.0) -> List[MemoryEntry]:
        out = []
        for m in self.memory:
            if kind is not None and m.kind != kind:
                continue
            if window is not None and now - m.time > window:
                continue
            out.append(m)
        return out

    def checkpoint(self) -> dict:
        return {
            "identity": self.identity,
            "genome_modules": self.genome.modules,
            "genome_regulatory": self.genome.regulatory,
            "genome_metadata": self.genome.metadata,
            "phenotype_traits": {k: {"value": v.value, "cost": v.cost,
                                     "modules": v.modules}
                                 for k, v in self.phenotype.traits.items()},
            "phenotype_size": self.phenotype.size,
            "phenotype_motility": self.phenotype.motility,
            "phenotype_sensing_range": self.phenotype.sensing_range,
            "phenotype_reproduction_rate": self.phenotype.reproduction_rate,
            "phenotype_defence": self.phenotype.defence,
            "phenotype_metabolic_power": self.phenotype.metabolic_power,
            "phenotype_structure": self.phenotype.structure,
            "phenotype_developmental_cost": self.phenotype.developmental_cost,
            "state_energy": self.state.energy,
            "state_nutrients": self.state.nutrients,
            "state_age": self.state.age,
            "state_stress": self.state.stress,
            "state_alive": self.state.alive,
            "location": self.location,
            "parent": self.parent,
            "generation": self.generation,
            "species_id": self.species_id,
            "reproduction_count": self.reproduction_count,
            "offspring": self.offspring,
            "metadata": self.metadata,
            "memory": [{"time": m.time, "kind": m.kind, "location": m.location,
                        "value": m.value, "detail": m.detail} for m in self.memory],
        }

    @classmethod
    def restore(cls, data: dict, genome: Genome, phenotype: Phenotype,
                state: OrganismState) -> "Organism":
        mem = [MemoryEntry(t=m["time"], kind=m["kind"],
                           location=tuple(m["location"]), value=m["value"],
                           detail=m.get("detail", {})) for m in data.get("memory", [])]
        return cls(identity=data["identity"], genome=genome, phenotype=phenotype,
                   state=state, memory=mem,
                   location=tuple(data.get("location", (0.0, 0.0))),
                   parent=data.get("parent"), generation=int(data.get("generation", 0)),
                   species_id=data.get("species_id"),
                   reproduction_count=int(data.get("reproduction_count", 0)),
                   offspring=list(data.get("offspring", [])),
                   metadata=data.get("metadata", {}))
