"""Multiverse manager: independent, persistent universe instances with versioned checkpoints.

Each universe owns its own mutable simulation state. Universes are isolated: two universes
never share state accidentally, and the same seed + config reproduces the same deterministic
result for the same engine version.

Serialization is versioned JSON metadata + NumPy binary arrays. Imported metadata is
validated and resource-limited; untrusted state is never unpickled or executed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .core.origin_events import OriginEventManager
from .core.clocks import (
    Clock, UniverseClockSet, SimScheduler, make_default_clock_set,
    FLWRCosmology,
)
from .core.universe import Universe, UniverseState, OriginEvent

SAVE_SCHEMA = "multiverse/v1"
CHECKPOINT_SCHEMA = "checkpoint/v1"
UNIVERSE_SCHEMA = "universe/v1"


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy arrays and other non-JSON types to JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj

# Observationally motivated (Planck 2018-like) reference flat LCDM.
REFERENCE_COSMOLOGY = {
    "H0_km_s_Mpc": 67.4,
    "Omega_m": 0.315,
    "Omega_r": 9.2e-5,
    "Omega_Lambda": 0.684908,
}

# Resource limits to defend against malformed/imported state.
MAX_CHECKPOINTS = 64
MAX_NAME_LEN = 128
# a_end is a model-specific validity bound, not a present-day clamp. The
# reference flat LCDM solver supports any a_end > 0; 10.0 remains a default
# sanity cap that can be raised per-universe.
MAX_A_END_DEFAULT = 10.0


class MultiverseError(ValueError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def classify_cosmology(c: Dict[str, float]) -> str:
    """Classify a cosmology as 'observational', 'hypothetical', or 'outside_validated_domain'.

    Does not claim arbitrary parameters yield a self-consistent Big Bang theory.
    """
    om = c.get("Omega_m", 0.0)
    ol = c.get("Omega_Lambda", 0.0)
    orr = c.get("Omega_r", 0.0)
    h0 = c.get("H0_km_s_Mpc", 0.0)
    flat = abs(om + orr + ol - 1.0)
    if (
        flat < 1e-6
        and 0.25 <= om <= 0.35
        and 0.60 <= ol <= 0.75
        and 0 <= orr <= 2e-4
        and 60 <= h0 <= 75
    ):
        return "observational"
    if flat < 1e-4 and 0.05 <= om <= 0.95 and 0.0 <= ol <= 0.95 and 0 <= orr <= 1e-3 and 40 <= h0 <= 90:
        return "hypothetical"
    return "outside_validated_domain"


def validate_cosmology(c: Dict[str, float]) -> None:
    for k in ("H0_km_s_Mpc", "Omega_m", "Omega_r", "Omega_Lambda"):
        if k not in c:
            raise MultiverseError(f"missing cosmology.{k}")
    if c["H0_km_s_Mpc"] <= 0 or c["Omega_m"] <= 0 or c["Omega_r"] < 0 or c["Omega_Lambda"] < 0:
        raise MultiverseError("invalid cosmological parameters")
    flat = abs(c["Omega_m"] + c["Omega_r"] + c["Omega_Lambda"] - 1.0)
    if flat > 1e-4:
        raise MultiverseError("cosmology must be flat (Omega sum = 1) within tolerance")
    if not (40 <= c["H0_km_s_Mpc"] <= 90):
        raise MultiverseError("H0 outside validated range [40, 90] km/s/Mpc")
    a_end = c.get("a_end", 1.0)
    max_a_end = c.get("max_a_end", MAX_A_END_DEFAULT)
    if a_end <= 0:
        raise MultiverseError("a_end must be positive")
    if a_end > max_a_end:
        raise MultiverseError(
            f"a_end {a_end} beyond the configured future domain (max_a_end={max_a_end})"
        )


@dataclass
class UniverseRecord:
    """Immutable identity + mutable simulation state for one universe."""
    uuid: str
    name: str
    created_at: str
    cosmology: Dict[str, float]
    seed: int
    model_settings: Dict[str, Any]
    status: str = "created"            # created | running | paused | finished
    cosmic_time_s: float = 0.0
    scale_factor: float = 1e-13
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    provenance: str = ""
    schema: str = SAVE_SCHEMA
    engine_version: str = "0.1.0"
    catalog: Dict[str, Any] = field(default_factory=dict)   # generated structure/object catalog
    last_saved: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UniverseRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class CheckpointRecord:
    id: str
    created_at: str
    cosmic_time_s: float
    scale_factor: float
    schema: str
    path: str          # relative file path within the universe directory
    size_bytes: int


class MultiverseManager:
    """Registry of independent universes with persistent checkpoint storage.

    Each universe is a full `Universe` instance with its own independent clocks,
    FLRW cosmology, and mutable state. Universes live under a root directory;
    each universe gets its own subdirectory containing metadata.json plus binary
    checkpoint arrays. Isolation is by directory and by in-memory record; no
    shared mutable simulation state.

    Causally disconnected universes do not share a unique physically meaningful
    global time. Any ordering of their creation events is a defined feature of
    the selected bulk model (or an administrative simulation ordering).
    """

    def __init__(self, root: str = "runs/multiverse"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._universes: Dict[str, UniverseRecord] = {}
        self._instances: Dict[str, Universe] = {}  # live universe instances
        self.scheduler = SimScheduler()
        self.origin_manager = OriginEventManager(self)
        self._load_index()

    # ---- index ----
    def _uni_dir(self, uid: str) -> Path:
        return self.root / uid

    def _load_index(self) -> None:
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir() and (entry / "metadata.json").is_file():
                try:
                    meta = json.loads((entry / "metadata.json").read_text())
                    uid = meta["uuid"]
                    self._universes[uid] = UniverseRecord.from_dict(meta)
                except Exception:
                    # skip corrupt universe directories rather than crash the manager
                    continue

    def _save_meta(self, rec: UniverseRecord) -> None:
        d = self._uni_dir(rec.uuid)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "metadata.json.tmp"
        tmp.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, d / "metadata.json")

    # ---- instance access ----
    def get_instance(self, uid: str) -> Universe:
        """Return the live Universe instance for uid, loading it if necessary."""
        if uid in self._instances:
            return self._instances[uid]
        rec = self.get(uid)
        # Reconstruct a Universe from the stored record.
        inst = Universe(
            universe_id=uid, name=rec.name, seed=rec.seed,
            cosmology=rec.cosmology, model_settings=rec.model_settings,
            origin_event=None,
        )
        inst.status = rec.status
        inst.created_at = rec.created_at
        inst.provenance = rec.provenance
        # Restore clocks and scale factor from the record.
        inst.clocks.cosmic.t = rec.cosmic_time_s
        inst.clocks.scale_factor.t = rec.scale_factor
        inst.clocks.bulk.t = rec.cosmic_time_s
        inst._state = inst.flrw.state_at(rec.scale_factor, rec.cosmic_time_s)
        self._instances[uid] = inst
        return inst

    def register_instance(self, uid: str, inst: Universe) -> None:
        self._instances[uid] = inst
        self.scheduler.register(uid, inst.clocks)   # atomic

    # ---- creation ----
    def create_universe(
        self,
        name: str,
        seed: Optional[int] = None,
        cosmology: Optional[Dict[str, float]] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        origin_event: Optional[OriginEvent] = None,
    ) -> UniverseRecord:
        if not name or not name.strip() or len(name) > MAX_NAME_LEN:
            raise MultiverseError(f"name must be non-empty and <= {MAX_NAME_LEN} chars")
        c = dict(REFERENCE_COSMOLOGY) if cosmology is None else dict(cosmology)
        c.setdefault("a_start", 1e-13)
        c.setdefault("a_end", 2.0)          # default to future evolution past present
        c.setdefault("steps", 500)
        c.setdefault("max_a_end", MAX_A_END_DEFAULT)
        validate_cosmology(c)
        ms = {
            "a_start": float(c["a_start"]),
            "a_end": float(c["a_end"]),
            "steps": int(c["steps"]),
            "max_a_end": float(c["max_a_end"]),
            "deterministic": True,
            **(model_settings or {}),
        }
        seed_val = int(seed if seed is not None else uuid.uuid4().int & (2**63 - 1))
        rec = UniverseRecord(
            uuid=str(uuid.uuid4()),
            name=name.strip(),
            created_at=now_iso(),
            cosmology=c,
            seed=seed_val,
            model_settings=ms,
            status="created",
            cosmic_time_s=0.0,
            scale_factor=float(c["a_start"]),
            provenance=classify_cosmology(c),
            catalog={},
        )
        self._universes[rec.uuid] = rec
        self._save_meta(rec)

        # Create the live instance and register with the scheduler.
        inst = Universe(
            universe_id=rec.uuid, name=rec.name, seed=rec.seed,
            cosmology=rec.cosmology, model_settings=rec.model_settings,
            origin_event=origin_event,
        )
        self.register_instance(rec.uuid, inst)
        return rec

    def list_universes(self) -> List[UniverseRecord]:
        return sorted(self._universes.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, uid: str) -> UniverseRecord:
        if uid not in self._universes:
            raise MultiverseError(f"unknown universe {uid!r}")
        return self._universes[uid]

    # ---- identity / isolation ----
    def universe_ids(self) -> List[str]:
        return list(self._universes.keys())

    # ---- mutation (switch time is a property of the active session, not the record) ----
    def rename(self, uid: str, name: str) -> UniverseRecord:
        rec = self.get(uid)
        if not name or len(name) > MAX_NAME_LEN:
            raise MultiverseError("invalid name")
        rec.name = name.strip()
        self._save_meta(rec)
        return rec

    def delete(self, uid: str) -> None:
        rec = self.get(uid)
        d = self._uni_dir(uid)
        import shutil
        if d.exists():
            shutil.rmtree(d)
        del self._universes[uid]

    # ---- time / status ----
    def set_time(self, uid: str, cosmic_time_s: float, scale_factor: float) -> UniverseRecord:
        rec = self.get(uid)
        rec.cosmic_time_s = float(cosmic_time_s)
        rec.scale_factor = float(scale_factor)
        self._save_meta(rec)
        return rec

    def set_status(self, uid: str, status: str) -> UniverseRecord:
        rec = self.get(uid)
        rec.status = status
        self._save_meta(rec)
        # Sync live instance if present.
        if uid in self._instances:
            if status == "paused":
                self._instances[uid].pause()
            elif status == "running":
                self._instances[uid].resume()
        return rec

    # ---- evolution ----
    def advance_universe(self, uid: str, dt_master_s: float) -> Dict[str, Any]:
        """Advance one universe by dt_master_s seconds of master time."""
        inst = self.get_instance(uid)
        result = inst.advance(dt_master_s, self.scheduler)
        # Sync record with instance state.
        rec = self.get(uid)
        rec.cosmic_time_s = inst._state.t
        rec.scale_factor = inst._state.a
        rec.status = inst.status

        # Check for model-dependent origin transitions.
        state = inst.state
        transitions = self.origin_manager.check_all_transitions(
            uid, state, inst._state.t)
        origin_events = []
        for transition in transitions:
            record = self.origin_manager.create_universe_from_transition(
                transition, inst, self)
            if record is not None and record.new_universe_id:
                origin_events.append(record.to_dict())

        self._save_meta(rec)
        result["origin_events"] = origin_events
        return result

    def advance_all(self, dt_master_s: float) -> Dict[str, Dict[str, Any]]:
        """Advance every registered universe by dt_master_s."""
        results: Dict[str, Dict[str, Any]] = {}
        for uid in list(self._universes.keys()):
            results[uid] = self.advance_universe(uid, dt_master_s)
        return results

    def pause_universe(self, uid: str) -> UniverseRecord:
        return self.set_status(uid, "paused")

    def resume_universe(self, uid: str) -> UniverseRecord:
        return self.set_status(uid, "running")

    def invalidate_universe(self, uid: str, reason: str) -> UniverseRecord:
        inst = self.get_instance(uid)
        inst.invalidate(reason)
        rec = self.get(uid)
        rec.status = "invalid"
        self._save_meta(rec)
        return rec

    # ---- checkpoints ----
    def checkpoint(self, uid: str, cid: Optional[str] = None,
                   arrays: Optional[Dict[str, np.ndarray]] = None) -> CheckpointRecord:
        """Backward-compatible alias for checkpoint_universe."""
        return self.checkpoint_universe(uid, cid, arrays)

    def checkpoint_universe(self, uid: str, cid: Optional[str] = None,
                            arrays: Optional[Dict[str, np.ndarray]] = None) -> CheckpointRecord:
        inst = self.get_instance(uid)
        cp_data = inst.checkpoint()
        # Convert any numpy arrays in cp_data to lists for JSON serialization.
        cp_data_json = _jsonify(cp_data)
        if arrays:
            cp_data_json["arrays"] = {k: np.asarray(v).tolist() for k, v in arrays.items()}
        rec = self.get(uid)
        if len(rec.checkpoints) >= MAX_CHECKPOINTS:
            raise MultiverseError(f"maximum {MAX_CHECKPOINTS} checkpoints reached for universe {uid}")
        cid = cid or f"cp-{int(time.time() * 1000)}"
        d = self._uni_dir(uid)
        d.mkdir(parents=True, exist_ok=True)
        cp_path = d / f"{cid}.npz"
        tmp = d / f"{cid}.tmp.npz"
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "universe": uid,
            "id": cid,
            "created_at": now_iso(),
            "cosmic_time_s": rec.cosmic_time_s,
            "scale_factor": rec.scale_factor,
            "config_sha": hashlib.sha256(json.dumps({"cosmology": rec.cosmology, "seed": rec.seed, "model_settings": rec.model_settings}, sort_keys=True).encode()).hexdigest()[:16],
        }
        payload: Dict[str, Any] = {}
        if arrays:
            for k, v in arrays.items():
                payload[k] = np.asarray(v)
        payload["meta"] = np.array(json.dumps(meta))
        payload["universe_state"] = np.array(json.dumps(cp_data_json))
        np.savez_compressed(str(tmp), **payload)
        os.replace(str(tmp), str(cp_path))
        cprec = CheckpointRecord(
            id=cid, created_at=meta["created_at"],
            cosmic_time_s=rec.cosmic_time_s, scale_factor=rec.scale_factor,
            schema=CHECKPOINT_SCHEMA, path=f"{uid}/{cid}.npz", size_bytes=cp_path.stat().st_size,
        )
        rec.checkpoints.append(asdict(cprec))
        rec.last_saved = now_iso()
        inst.last_saved = now_iso()
        self._save_meta(rec)
        return cprec

    def list_checkpoints(self, uid: str) -> List[CheckpointRecord]:
        rec = self.get(uid)
        return [CheckpointRecord(**c) for c in rec.checkpoints]

    def restore_checkpoint(self, uid: str, cid: str) -> Dict[str, Any]:
        rec = self.get(uid)
        cprecs = [c for c in rec.checkpoints if c["id"] == cid]
        if not cprecs:
            raise MultiverseError(f"checkpoint {cid!r} not found in universe {uid}")
        cp = self._uni_dir(uid) / f"{cid}.npz"
        if not cp.is_file():
            raise MultiverseError(f"checkpoint file missing for {cid!r}")
        with np.load(cp, allow_pickle=False) as z:
            if "meta" not in z.files:
                raise MultiverseError("checkpoint missing meta array")
            meta = json.loads(str(z["meta"]))
            if meta.get("schema") != CHECKPOINT_SCHEMA:
                raise MultiverseError(f"unsupported checkpoint schema {meta.get('schema')!r}")
            arrays = {k: z[k] for k in z.files if k != "meta" and k != "universe_state"}
            # Restore the live Universe instance from checkpoint.
            if "universe_state" in z.files:
                cp_data = json.loads(str(z["universe_state"]))
                inst = Universe.restore(cp_data)
                self._instances[uid] = inst
                self.scheduler.register(uid, inst.clocks)
                rec.cosmic_time_s = float(meta["cosmic_time_s"])
                rec.scale_factor = float(meta["scale_factor"])
                rec.status = "paused"
                rec.last_saved = now_iso()
                self._save_meta(rec)
                return {"meta": meta, "arrays": arrays, "universe": rec.to_dict()}
        # Fallback: restore record-level time only.
        rec.cosmic_time_s = float(meta["cosmic_time_s"])
        rec.scale_factor = float(meta["scale_factor"])
        rec.status = "paused"
        rec.last_saved = now_iso()
        self._save_meta(rec)
        return {"meta": meta, "arrays": arrays, "universe": rec.to_dict()}

    # ---- branching ----
    def branch(self, uid: str, new_name: str, from_checkpoint: Optional[str] = None, seed: Optional[int] = None) -> UniverseRecord:
        rec = self.get(uid)
        cosmo = dict(rec.cosmology)
        new_seed = int(seed if seed is not None else rec.seed)
        branch = self.create_universe(
            name=new_name, seed=new_seed, cosmology=cosmo,
            model_settings=dict(rec.model_settings),
        )
        if from_checkpoint:
            restored = self.restore_checkpoint(uid, from_checkpoint)
            branch.cosmic_time_s = restored["meta"]["cosmic_time_s"]
            branch.scale_factor = restored["meta"]["scale_factor"]
            branch.status = "paused"
            self._save_meta(branch)
        return branch

    def to_public(self) -> List[Dict[str, Any]]:
        out = []
        for r in self.list_universes():
            # Prefer live instance state when available.
            if r.uuid in self._instances:
                inst = self._instances[r.uuid]
                out.append(inst.to_public())
            else:
                out.append({
                    "uuid": r.uuid, "name": r.name, "created_at": r.created_at,
                    "seed": r.seed, "cosmic_age_Gyr": r.cosmic_time_s / 3.15576e16,
                    "cosmology": r.cosmology, "status": r.status,
                    "scale_factor": r.scale_factor, "last_saved": r.last_saved,
                    "checkpoints": len(r.checkpoints), "provenance": r.provenance,
                    "schema": r.schema,
                })
        return out

    # ---- origin events ----
    def list_origin_events(self) -> List[Dict]:
        """List all recorded universe-origin events."""
        return self.origin_manager.list_origins()
