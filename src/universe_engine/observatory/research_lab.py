"""Research laboratory: managed evolution experiments with full provenance.

Experiments run a `MicrobialWorld` (or a bare `Habitat`) under declared,
reproducible conditions. Each experiment gets a UUID, a config, a run history
(population snapshots, births/deaths, extinctions, lineage splits), and
checkpoint/restore. Nothing here claims abiogenesis: a world seeded with zero
organisms stays sterile, and all founders are explicitly introduced.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..life.simulation import MicrobialWorld, MicrobialWorldConfig
from ..habitat import Habitat, make_default_network
from ..planet.model import Planet, PlanetSpec

EXPERIMENT_SCHEMA = "research-lab/v1"
MAX_EXPERIMENTS = 256
MAX_STEPS_PER_REQUEST = 10_000


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ResearchLabError(ValueError):
    pass


@dataclass
class ExperimentRecord:
    id: str
    name: str
    created_at: str
    config: Dict[str, Any]
    status: str = "created"          # created | running | paused | finished | error
    t_gyr: float = 0.0
    steps: int = 0
    error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    schema: str = EXPERIMENT_SCHEMA
    last_saved: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchLab:
    """Registry of independent research experiments with persistent state.

    Each experiment owns its own mutable `MicrobialWorld`. Isolation is by
    in-memory record; no shared mutable simulation state between experiments.
    """

    def __init__(self, root: str = "runs/research_lab"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._worlds: Dict[str, MicrobialWorld] = {}

    # ---- creation ----------------------------------------------------------
    def create_experiment(self, name: str, config: Optional[Dict[str, Any]] = None,
                          seed: Optional[int] = None) -> ExperimentRecord:
        if not name or not name.strip() or len(name) > 128:
            raise ResearchLabError("name must be non-empty and <= 128 chars")
        if len(self._experiments) >= MAX_EXPERIMENTS:
            raise ResearchLabError(f"maximum {MAX_EXPERIMENTS} experiments reached")
        cfg = MicrobialWorldConfig(**(config or {}))
        if seed is not None:
            cfg.seed = int(seed)
        rec = ExperimentRecord(
            id=str(uuid.uuid4()), name=name.strip(), created_at=now_iso(),
            config={"n_patches": cfg.n_patches, "patch_spacing_m": cfg.patch_spacing_m,
                    "seed": cfg.seed, "n_founders": cfg.n_founders,
                    "founder_genome_len": cfg.founder_genome_len,
                    "initial_pool": cfg.initial_pool,
                    "nutrient_per_patch_mol": cfg.nutrient_per_patch_mol,
                    "dt_gyr": cfg.dt_gyr, "cognition_architecture": cfg.cognition_architecture},
        )
        self._experiments[rec.id] = rec
        self._worlds[rec.id] = MicrobialWorld(config=cfg)
        return rec

    def list_experiments(self) -> List[ExperimentRecord]:
        return sorted(self._experiments.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, eid: str) -> ExperimentRecord:
        if eid not in self._experiments:
            raise ResearchLabError(f"unknown experiment {eid!r}")
        return self._experiments[eid]

    def world(self, eid: str) -> MicrobialWorld:
        self.get(eid)  # validate
        return self._worlds[eid]

    # ---- control -----------------------------------------------------------
    def delete(self, eid: str) -> None:
        self.get(eid)
        self._experiments.pop(eid, None)
        self._worlds.pop(eid, None)

    def set_status(self, eid: str, status: str) -> ExperimentRecord:
        rec = self.get(eid)
        rec.status = status
        return rec

    def step(self, eid: str, n: int = 1, dt_gyr: Optional[float] = None) -> Dict[str, Any]:
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_STEPS_PER_REQUEST:
            raise ResearchLabError(f"n must be an integer in [1, {MAX_STEPS_PER_REQUEST}]")
        rec = self.get(eid)
        w = self.world(eid)
        last = None
        for _ in range(n):
            last = w.step(dt_gyr)
        rec.t_gyr = w.t_gyr
        rec.steps += n
        rec.status = "running"
        self._append_history(rec, w, last)
        return {"experiment": rec.to_dict(), "last": last}

    def run(self, eid: str, max_steps: Optional[int] = None, dt_gyr: Optional[float] = None) -> Dict[str, Any]:
        rec = self.get(eid)
        w = self.world(eid)
        done = 0
        last = None
        while max_steps is None or done < max_steps:
            last = w.step(dt_gyr)
            done += 1
            rec.t_gyr = w.t_gyr
            rec.steps += 1
            self._append_history(rec, w, last)
        rec.status = "finished"
        return {"experiment": rec.to_dict(), "last": last, "steps": done}

    def pause(self, eid: str) -> ExperimentRecord:
        rec = self.get(eid)
        rec.status = "paused"
        return rec

    # ---- observation -------------------------------------------------------
    def stats(self, eid: str) -> Dict[str, Any]:
        rec = self.get(eid)
        w = self.world(eid)
        s = w.stats()
        return {"t_gyr": w.t_gyr, "n_organisms": s.n_organisms, "n_species": s.n_species,
                "n_lineages": s.n_lineages, "total_energy": s.total_energy,
                "total_nutrients": s.total_nutrients, "resource_pool": s.resource_pool,
                "births": s.births, "deaths": s.deaths, "diversity": s.diversity,
                "sterile": w.is_sterile(), "steps": rec.steps, "status": rec.status}

    def history(self, eid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rec = self.get(eid)
        h = rec.history
        if limit is not None:
            h = h[-limit:]
        return h

    def trajectory(self, eid: str) -> Dict[str, List[float]]:
        """Return time series for plotting: t, organisms, species, lineages, births, deaths, resource_pool."""
        rec = self.get(eid)
        out = {"t": [], "organisms": [], "species": [], "lineages": [],
               "births": [], "deaths": [], "resource_pool": []}
        for h in rec.history:
            out["t"].append(h["t_gyr"])
            out["organisms"].append(h["n_organisms"])
            out["species"].append(h["n_species"])
            out["lineages"].append(h["n_lineages"])
            out["births"].append(h["births"])
            out["deaths"].append(h["deaths"])
            out["resource_pool"].append(h["resource_pool"])
        return out

    def lineage(self, eid: str) -> Dict[str, Any]:
        rec = self.get(eid)
        w = self.world(eid)
        return {"lineage": w.lineage.checkpoint() if hasattr(w.lineage, "checkpoint") else {},
                "n_lineages": rec.history[-1]["n_lineages"] if rec.history else 0}

    def extinction_events(self, eid: str) -> List[Dict[str, Any]]:
        rec = self.get(eid)
        return [e for e in rec.history if e.get("extinction")]

    # ---- persistence -------------------------------------------------------
    def checkpoint(self, eid: str) -> Dict[str, Any]:
        rec = self.get(eid)
        w = self.world(eid)
        cp = {"schema": "research-checkpoint/v1", "experiment": rec.to_dict(),
              "world": w.checkpoint()}
        rec.checkpoints.append({"created_at": now_iso(), "t_gyr": rec.t_gyr,
                                "steps": rec.steps, "schema": cp["schema"]})
        rec.last_saved = now_iso()
        return cp

    def restore(self, eid: str, cp: Dict[str, Any]) -> ExperimentRecord:
        rec = self.get(eid)
        w = MicrobialWorld.restore(cp["world"])
        self._worlds[eid] = w
        rec.t_gyr = float(cp["experiment"]["t_gyr"])
        rec.steps = int(cp["experiment"]["steps"])
        rec.status = "paused"
        rec.history = list(cp["experiment"].get("history", []))
        rec.last_saved = now_iso()
        return rec

    # ---- helpers -----------------------------------------------------------
    def _append_history(self, rec: ExperimentRecord, w: MicrobialWorld, last: Optional[Dict[str, Any]]):
        s = w.stats()
        entry = {"t_gyr": w.t_gyr, "steps": rec.steps, "n_organisms": s.n_organisms,
                 "n_species": s.n_species, "n_lineages": s.n_lineages,
                 "total_energy": s.total_energy, "total_nutrients": s.total_nutrients,
                 "resource_pool": s.resource_pool, "births": s.births, "deaths": s.deaths,
                 "diversity": s.diversity, "sterile": w.is_sterile()}
        if last and last.get("events"):
            entry["events"] = last["events"]
        rec.history.append(entry)
