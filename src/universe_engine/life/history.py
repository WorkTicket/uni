"""LifeHistoryDatabase: persistent histories and state summaries.

Stores per-organism and per-population snapshots across time, lineage records,
and species/diversity metrics. Provides versioned serialization for save/reload
and deterministic continuation of experiments.

Provenance: this is a data-recording layer; it makes no claims about the
reality of the recorded histories beyond what the simulation produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import time


@dataclass
class OrganismRecord:
    """One snapshot of an organism at a point in time."""
    t: float
    identity: str
    species_id: Optional[str]
    generation: int
    age: float
    energy: float
    nutrients: float
    alive: bool
    location: list
    genome_modules: tuple
    behavior: Optional[str] = None


@dataclass
class PopulationRecord:
    """One population snapshot."""
    t: float
    population_id: str
    size: int
    total_energy: float
    total_nutrients: float
    n_species: int
    n_lineages: int
    births: int
    deaths: int


@dataclass
class LifeHistoryDatabase:
    """Append-only history of organism and population snapshots.

    records:
        time-ordered list of snapshots (each a dict with schema tag).
    lineage:
        SpeciesLineage for ancestry tracking.
    metadata:
        run provenance (seed, config version, start time).
    """

    records: List[dict] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    max_records: int = 1_000_000

    def record_organism(self, t: float, org) -> None:
        self.records.append({
            "schema": "organism/v1",
            "t": float(t),
            "identity": getattr(org, "identity", None),
            "species_id": getattr(org, "species_id", None),
            "generation": getattr(org, "generation", 0),
            "age": float(getattr(org.state, "age", 0.0)),
            "energy": float(getattr(org.state, "energy", 0.0)),
            "nutrients": float(getattr(org.state, "nutrients", 0.0)),
            "alive": bool(getattr(org.state, "alive", False)),
            "location": list(getattr(org, "location", [])),
            "genome_modules": tuple(getattr(org.genome, "modules", ())),
        })
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

    def record_population(self, t: float, pop_id: str, pop) -> None:
        stats = pop.stats()
        self.records.append({
            "schema": "population/v1",
            "t": float(t),
            "population_id": pop_id,
            "size": stats.size,
            "total_energy": stats.total_energy,
            "total_nutrients": stats.total_nutrients,
            "n_species": stats.n_species,
            "n_lineages": stats.n_lineages,
            "births": stats.births,
            "deaths": stats.deaths,
        })
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

    def diversity(self) -> dict:
        """Compute biodiversity, biomass, population trajectory summary."""
        pops = [r for r in self.records if r.get("schema") == "population/v1"]
        if not pops:
            return {"n_species": 0, "max_biomass": 0.0, "n_extinct": 0}
        max_size = max(p["size"] for p in pops)
        n_extinct = sum(1 for p in pops if p["size"] == 0)
        species_seen = set()
        for p in pops:
            species_seen.update(range(p["n_species"]))
        return {"n_species": len(species_seen), "max_biomass": float(max_size),
                "n_extinct": int(n_extinct), "n_population_snapshots": len(pops)}

    def save(self, path: str) -> None:
        payload = {
            "schema": "life-history/v1",
            "metadata": self.metadata,
            "records": self.records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "LifeHistoryDatabase":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        db = cls(metadata=payload.get("metadata", {}),
                 max_records=int(payload.get("max_records", 1_000_000)))
        db.records = payload.get("records", [])
        return db
