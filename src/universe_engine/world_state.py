"""Authoritative world-state spine (PASS 4).

Stable identities and explicit relationships for the chain
universe -> star -> planet -> habitat -> organism. Entities carry explicit
physical units and provenance, a local clock, model version, and state
ownership. Versioned checkpoints capture the full chain so a restored object
reproduces the same scientific history.

This is an adapter/registry layer on top of the existing models (Universe,
Planet, Habitat, PersistentOrganism); it does not replace them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional
import time

import numpy as np

from .core.clocks import SECONDS_PER_GYR
from .planet.model import Planet, PlanetSpec, StarSpec

try:  # pragma: no cover - optional integration with the persistent organism backend
    from .research.persistent_organism import PersistentOrganism
    from .habitat import Habitat
except Exception:  # pragma: no cover
    PersistentOrganism = None  # type: ignore
    Habitat = None  # type: ignore


class Provenance(str):
    """Source model / provenance of a quantity (matches the state proposal)."""
    SIMULATED = "simulated"
    ANALYTIC = "analytic"
    RECONSTRUCTED = "reconstructed"
    ILLUSTRATIVE = "illustrative"
    SPECULATIVE = "speculative"

    def __new__(cls, value):
        return str.__new__(cls, value)


@dataclass(frozen=True)
class Quantity:
    """A physical quantity with explicit units and provenance.

    Numerically derived, reduced-order and illustrative properties remain
    distinguishable via `provenance` and `model_id`.
    """
    value: float
    unit: str
    provenance: str = Provenance.SIMULATED
    model_id: str = ""
    uncertainty: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Quantity":
        return cls(
            value=float(d["value"]), unit=str(d["unit"]),
            provenance=str(d.get("provenance", Provenance.SIMULATED)),
            model_id=str(d.get("model_id", "")),
            uncertainty=(None if d.get("uncertainty") is None else float(d["uncertainty"])),
        )


@dataclass
class WorldEntity:
    """One materialized scientific entity with a stable identity.

    parent_id gives the explicit physical relationship (e.g. planet -> star,
    habitat -> planet, organism -> habitat). fields carries only real modeled
    state as Quantity objects; derived/illustrative values carry their own
    provenance so the UI never promotes them into measurements.
    """
    id: str
    parent_id: Optional[str]
    universe_id: str
    kind: str            # universe | star | planet | habitat | organism
    fields: Dict[str, Quantity] = field(default_factory=dict)
    model_version: str = "0.1.0"
    updated_at_local_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "parent_id": self.parent_id, "universe_id": self.universe_id,
            "kind": self.kind, "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "model_version": self.model_version, "updated_at_local_time": self.updated_at_local_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldEntity":
        return cls(
            id=d["id"], parent_id=d.get("parent_id"), universe_id=d["universe_id"],
            kind=d["kind"],
            fields={k: Quantity.from_dict(v) for k, v in d.get("fields", {}).items()},
            model_version=d.get("model_version", "0.1.0"),
            updated_at_local_time=float(d.get("updated_at_local_time", 0.0)),
        )


@dataclass
class WorldClock:
    """Explicit local clock for one universe (distinct universes need no common time)."""
    universe_id: str
    integration_time_s: float = 0.0
    model_epoch: str = ""
    scale_factor: Optional[float] = None
    paused: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldClock":
        return cls(
            universe_id=d["universe_id"],
            integration_time_s=float(d.get("integration_time_s", 0.0)),
            model_epoch=str(d.get("model_epoch", "")),
            scale_factor=(None if d.get("scale_factor") is None else float(d["scale_factor"])),
            paused=bool(d.get("paused", False)),
        )


WORLD_SCHEMA = "world-state/v1"
WORLD_MODEL_VERSION = "0.1.0"


class WorldState:
    """Registry of entities, relationships, clocks, and versioned checkpoints.

    One complete, persistent connection: universe -> planet -> habitat ->
    organism. Restoring a checkpoint reproduces the same scientific history;
    a single physical object is never regenerated with a different history.
    """

    def __init__(self, universe_id: str):
        self.universe_id = universe_id
        self.entities: Dict[str, WorldEntity] = {}
        self.clocks: Dict[str, WorldClock] = {}
        self._owners: Dict[str, str] = {}   # entity id -> owning model id
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- entities / relationships -------------------------------------
    def add_entity(self, entity: WorldEntity, owner: Optional[str] = None) -> None:
        self.entities[entity.id] = entity
        self.clocks.setdefault(entity.universe_id, WorldClock(universe_id=entity.universe_id))
        if owner:
            self._owners[entity.id] = owner

    def link(self, child_id: str, parent_id: str) -> None:
        if child_id in self.entities:
            self.entities[child_id] = replace(self.entities[child_id], parent_id=parent_id)

    def entity(self, eid: str) -> Optional[WorldEntity]:
        return self.entities.get(eid)

    def children(self, parent_id: str) -> List[WorldEntity]:
        return [e for e in self.entities.values() if e.parent_id == parent_id]

    def chain(self, eid: str) -> List[WorldEntity]:
        out: List[WorldEntity] = []
        cur: Optional[str] = eid
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            e = self.entities.get(cur)
            if e is None:
                break
            out.append(e)
            cur = e.parent_id
        return out

    # ---- clock --------------------------------------------------------
    def clock(self, universe_id: Optional[str] = None) -> WorldClock:
        uid = universe_id or self.universe_id
        if uid not in self.clocks:
            self.clocks[uid] = WorldClock(universe_id=uid)
        return self.clocks[uid]

    def advance(self, dt_s: float, universe_id: Optional[str] = None) -> WorldClock:
        clk = self.clock(universe_id)
        clk.integration_time_s += dt_s
        return clk

    # ---- checkpoint / restore -----------------------------------------
    def checkpoint(self, code_sha: str = "unknown") -> dict:
        model_versions = {eid: e.model_version for eid, e in self.entities.items()}
        rng_states: Dict[str, Any] = {}
        return {
            "schema": WORLD_SCHEMA,
            "code_sha": code_sha,
            "model_versions": model_versions,
            "universe_id": self.universe_id,
            "created_at": self.created_at,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "clocks": {uid: c.to_dict() for uid, c in self.clocks.items()},
            "owners": dict(self._owners),
            "rng_states": rng_states,
        }

    @classmethod
    def restore(cls, data: dict) -> "WorldState":
        if data.get("schema") != WORLD_SCHEMA:
            raise ValueError(f"unsupported world-state schema {data.get('schema')!r}")
        ws = cls(universe_id=data["universe_id"])
        ws.created_at = data.get("created_at", ws.created_at)
        ws.entities = {eid: WorldEntity.from_dict(e) for eid, e in data.get("entities", {}).items()}
        ws.clocks = {uid: WorldClock.from_dict(c) for uid, c in data.get("clocks", {}).items()}
        ws._owners = dict(data.get("owners", {}))
        return ws


# ---- chain wiring -----------------------------------------------------
def _q(value, unit, provenance=Provenance.SIMULATED, model_id="", uncertainty=None) -> Quantity:
    return Quantity(value=float(value), unit=str(unit), provenance=str(provenance),
                    model_id=str(model_id), uncertainty=uncertainty)


def connect_chain(ws: WorldState, universe, planet: Planet, habitat: Habitat,
                  organism=None) -> dict:
    """Wire one complete, persistent connection: universe -> planet -> habitat -> organism.

    Registers stable IDs and explicit parent-child relationships, captures the
    mutable model state as Quantity fields (with units/provenance), and syncs
    the world clock with the planet/habitat clock. Returns the entity ids used.
    """
    uid = ws.universe_id
    uni_id = getattr(universe, "uuid", None) or "universe-" + uid
    planet_id = "planet-" + uid
    habitat_id = "habitat-" + uid

    ws.add_entity(WorldEntity(
        id=uni_id, parent_id=None, universe_id=uid, kind="universe",
        fields={
            "status": _q(0, "nominal", Provenance.SIMULATED, "core.universe"),
            "seed": _q(float(getattr(universe, "seed", 0)), "1", Provenance.SIMULATED, "core.universe"),
        },
        model_version="0.1.0", updated_at_local_time=float(getattr(universe, "_state", type("x", (), {"t": 0.0})()).t),
    ), owner="core.universe")
    ws.clock(uid).integration_time_s = float(planet.t_gyr) * SECONDS_PER_GYR

    env = planet.environment()
    ws.add_entity(WorldEntity(
        id=planet_id, parent_id=uni_id, universe_id=uid, kind="planet",
        fields={
            "t_gyr": _q(planet.t_gyr, "Gyr", Provenance.SIMULATED, "planet.model"),
            "T_global_K": _q(env.get("T_global_K"), "K", Provenance.SIMULATED, "planet.model"),
            "pCO2_bar": _q(env.get("pCO2_bar"), "bar", Provenance.SIMULATED, "planet.model"),
            "pH_ocean": _q(env.get("pH_ocean"), "1", Provenance.SIMULATED, "planet.model"),
            "liquid_water_fraction": _q(env.get("liquid_water_fraction"), "1", Provenance.SIMULATED, "planet.model"),
            "a_au": _q(planet.spec.a_au, "AU", Provenance.ANALYTIC, "planet.model"),
            "mass_mearth": _q(planet.spec.mass_mearth, "Mearth", Provenance.ANALYTIC, "planet.model"),
        },
        model_version="0.1.0", updated_at_local_time=float(planet.t_gyr),
    ), owner="planet.model")
    ws.link(planet_id, uni_id)

    ws.add_entity(WorldEntity(
        id=habitat_id, parent_id=planet_id, universe_id=uid, kind="habitat",
        fields={
            "t_gyr": _q(habitat.t_gyr, "Gyr", Provenance.SIMULATED, "habitat"),
            "n_patches": _q(habitat.n_patches, "1", Provenance.SIMULATED, "habitat"),
            "mass_ledger_error": _q(habitat._ledger_error, "1", Provenance.SIMULATED, "habitat"),
            "seed": _q(float(habitat.seed), "1", Provenance.SIMULATED, "habitat"),
        },
        model_version="0.1.0", updated_at_local_time=float(habitat.t_gyr),
    ), owner="habitat")
    ws.link(habitat_id, planet_id)

    org_id = None
    if organism is not None and PersistentOrganism is not None:
        org_id = getattr(organism.identity, "organism_id", None) or "organism-" + uid
        ws.add_entity(WorldEntity(
            id=org_id, parent_id=habitat_id, universe_id=uid, kind="organism",
            fields={
                "t_gyr": _q(organism.t_gyr, "Gyr", Provenance.SIMULATED, "persistent_organism"),
                "energy": _q(organism.body.energy, "1", Provenance.SIMULATED, "persistent_organism"),
                "nutrients": _q(organism.body.nutrients, "1", Provenance.SIMULATED, "persistent_organism"),
                "stress": _q(organism.body.stress, "1", Provenance.SIMULATED, "persistent_organism"),
                "alive": _q(1.0 if organism.alive else 0.0, "1", Provenance.SIMULATED, "persistent_organism"),
                "n_steps": _q(float(organism.n_steps), "1", Provenance.SIMULATED, "persistent_organism"),
                "lifetime_reward": _q(organism.lifetime_reward, "1", Provenance.SIMULATED, "persistent_organism"),
            },
            model_version="0.1.0", updated_at_local_time=float(organism.t_gyr),
        ), owner="persistent_organism")
        ws.link(org_id, habitat_id)

    return {"universe": uni_id, "planet": planet_id, "habitat": habitat_id, "organism": org_id}


def chain_state(ws: WorldState, habitat: Habitat, organism=None) -> dict:
    """Capture the full chain state (habitat checkpoint incl. planet + organism)."""
    out = {
        "schema": WORLD_SCHEMA,
        "universe_id": ws.universe_id,
        "t_gyr": float(habitat.t_gyr),
        "habitat": habitat.checkpoint(),
    }
    if organism is not None:
        out["organism"] = organism.checkpoint()
    return out