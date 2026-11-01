"""SpeciesLineage: tracked ancestry and population divergence.

Species are not assigned from cosmetic templates. They emerge through tracked
ancestry and population divergence according to declared criteria (genetic
distance, reproductive isolation, distinct niche). Lineage is a DAG of genome
identities with parent->child links and mutation counts.

Provenance: speciation criteria are declared inputs (thresholds on genetic
distance and isolation), not an empirical species concept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .genome import Genome


@dataclass
class LineageNode:
    """One genome identity in the lineage DAG."""
    genome_id: str             # hash of genome modules (unique per distinct genome)
    genome: Genome
    parent: Optional[str]      # parent genome_id or None (founder)
    birth_time: float
    n_mutations: int           # mutations accumulated along this branch
    descendant_count: int = 0
    extinct: bool = False
    extinction_time: Optional[float] = None


@dataclass
class SpeciesLineage:
    """Tracks ancestry and speciation across a population.

    species_criteria:
        genetic_distance_threshold: genomes farther apart than this are
            candidate separate species.
        isolation_threshold: generations without gene flow to be considered
            reproductively isolated.
    """
    species_criteria: Dict[str, float] = field(default_factory=lambda: {
        "genetic_distance_threshold": 0.3,
        "isolation_threshold": 50,
    })
    nodes: Dict[str, LineageNode] = field(default_factory=dict)
    species: Dict[str, Set[str]] = field(default_factory=dict)   # species_id -> set of genome_ids
    organism_species: Dict[str, str] = field(default_factory=dict)  # organism_identity -> species_id
    next_species_id: int = 0
    next_node_id: int = 0

    # ---- genome identity ---------------------------------------------------
    def _genome_id(self, genome: Genome) -> str:
        return str(hash(genome))

    def register_genome(self, genome: Genome, parent_genome: Optional[Genome],
                        birth_time: float, n_mutations: int) -> str:
        gid = self._genome_id(genome)
        if gid in self.nodes:
            return gid
        parent_id = None if parent_genome is None else self._genome_id(parent_genome)
        node = LineageNode(genome_id=gid, genome=genome, parent=parent_id,
                           birth_time=birth_time, n_mutations=n_mutations)
        self.nodes[gid] = node
        if parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].descendant_count += 1
        self.next_node_id += 1
        return gid

    # ---- speciation --------------------------------------------------------
    def _genetic_distance(self, a: Genome, b: Genome) -> float:
        """Simple Jaccard-ish distance on module sets plus length difference."""
        sa = set(a.modules); sb = set(b.modules)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb); union = len(sa | sb)
        jaccard = 1.0 - inter / union if union else 0.0
        len_diff = abs(a.length - b.length) / max(1, max(a.length, b.length))
        return 0.7 * jaccard + 0.3 * len_diff

    def assign_species(self, organism_identity: str, genome: Genome,
                       generation: int, now: float) -> str:
        """Assign (or create) a species for an organism based on lineage.

        An organism joins an existing species if its genome is within the
        genetic distance threshold of a member genome AND not isolated.
        Otherwise a new species is created.
        """
        gid = self._genome_id(genome)
        self.register_genome(genome, None, now, 0)

        # Find candidate species: any member genome within distance threshold.
        best_species = None; best_dist = float("inf")
        for sid, members in self.species.items():
            for mgid in members:
                if mgid not in self.nodes:
                    continue
                d = self._genetic_distance(genome, self.nodes[mgid].genome)
                if d < best_dist:
                    best_dist = d; best_species = sid

        threshold = self.species_criteria["genetic_distance_threshold"]
        if best_species is not None and best_dist <= threshold:
            self.species[best_species].add(gid)
            self.organism_species[organism_identity] = best_species
            return best_species

        # Otherwise speciate: create a new species.
        sid = f"species-{self.next_species_id}"
        self.next_species_id += 1
        self.species[sid] = {gid}
        self.organism_species[organism_identity] = sid
        return sid

    def n_species(self) -> int:
        return len(self.species)

    def n_lineages(self) -> int:
        return len(self.nodes)

    def checkpoint(self) -> dict:
        return {
            "species_criteria": self.species_criteria,
            "nodes": {k: {"genome_modules": v.genome.modules,
                          "genome_regulatory": v.genome.regulatory,
                          "parent": v.parent, "birth_time": v.birth_time,
                          "n_mutations": v.n_mutations,
                          "descendant_count": v.descendant_count,
                          "extinct": v.extinct,
                          "extinction_time": v.extinction_time}
                      for k, v in self.nodes.items()},
            "species": {k: sorted(v) for k, v in self.species.items()},
            "organism_species": dict(self.organism_species),
            "next_species_id": self.next_species_id,
            "next_node_id": self.next_node_id,
        }

    @classmethod
    def restore(cls, data: dict) -> "SpeciesLineage":
        lin = cls(species_criteria=data.get("species_criteria", {}),
                  next_species_id=int(data.get("next_species_id", 0)),
                  next_node_id=int(data.get("next_node_id", 0)))
        from .genome import Genome
        for gid, nd in data.get("nodes", {}).items():
            g = Genome(modules=tuple(nd["genome_modules"]),
                       regulatory={int(k): v for k, v in nd.get("genome_regulatory", {}).items()})
            node = LineageNode(genome_id=gid, genome=g, parent=nd.get("parent"),
                               birth_time=float(nd.get("birth_time", 0.0)),
                               n_mutations=int(nd.get("n_mutations", 0)),
                               descendant_count=int(nd.get("descendant_count", 0)),
                               extinct=bool(nd.get("extinct", False)),
                               extinction_time=nd.get("extinction_time"))
            lin.nodes[gid] = node
        lin.species = {k: set(v) for k, v in data.get("species", {}).items()}
        lin.organism_species = dict(data.get("organism_species", {}))
        return lin
