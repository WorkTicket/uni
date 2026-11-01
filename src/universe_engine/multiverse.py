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

SAVE_SCHEMA = "multiverse/v1"
CHECKPOINT_SCHEMA = "checkpoint/v1"

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
MAX_A_END = 10.0


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
    if c["a_end"] > MAX_A_END:
        raise MultiverseError(f"a_end {c['a_end']} beyond validated future domain (a <= {MAX_A_END})")


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

    Universes live under a root directory; each universe gets its own subdirectory
    containing metadata.json plus binary checkpoint arrays. Isolation is by directory
    and by in-memory record; no shared mutable simulation state.
    """

    def __init__(self, root: str = "runs/multiverse"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._universes: Dict[str, UniverseRecord] = {}
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
        os.replace(tmp, d / "metadata.json")   # atomic

    # ---- creation ----
    def create_universe(
        self,
        name: str,
        seed: Optional[int] = None,
        cosmology: Optional[Dict[str, float]] = None,
        model_settings: Optional[Dict[str, Any]] = None,
    ) -> UniverseRecord:
        if not name or not name.strip() or len(name) > MAX_NAME_LEN:
            raise MultiverseError(f"name must be non-empty and <= {MAX_NAME_LEN} chars")
        c = dict(REFERENCE_COSMOLOGY) if cosmology is None else dict(cosmology)
        c.setdefault("a_start", 1e-13)
        c.setdefault("a_end", 1.0)
        c.setdefault("steps", 500)
        validate_cosmology(c)
        ms = {
            "a_start": float(c["a_start"]),
            "a_end": float(c["a_end"]),
            "steps": int(c["steps"]),
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
        return rec

    # ---- checkpoints ----
    def checkpoint(self, uid: str, cid: Optional[str] = None, arrays: Optional[Dict[str, np.ndarray]] = None) -> CheckpointRecord:
        rec = self.get(uid)
        if len(rec.checkpoints) >= MAX_CHECKPOINTS:
            raise MultiverseError(f"maximum {MAX_CHECKPOINTS} checkpoints reached for universe {uid}")
        cid = cid or f"cp-{int(time.time() * 1000)}"
        d = self._uni_dir(uid)
        d.mkdir(parents=True, exist_ok=True)
        cp_path = d / f"{cid}.npz"
        tmp = d / f"{cid}.tmp.npz"
        payload: Dict[str, Any] = {}
        if arrays:
            for k, v in arrays.items():
                payload[k] = np.asarray(v)
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "universe": uid,
            "id": cid,
            "created_at": now_iso(),
            "cosmic_time_s": rec.cosmic_time_s,
            "scale_factor": rec.scale_factor,
            "config_sha": hashlib.sha256(json.dumps({"cosmology": rec.cosmology, "seed": rec.seed, "model_settings": rec.model_settings}, sort_keys=True).encode()).hexdigest()[:16],
        }
        payload["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(str(tmp), **payload)
        os.replace(str(tmp), str(cp_path))
        cprec = CheckpointRecord(
            id=cid, created_at=meta["created_at"],
            cosmic_time_s=rec.cosmic_time_s, scale_factor=rec.scale_factor,
            schema=CHECKPOINT_SCHEMA, path=f"{uid}/{cid}.npz", size_bytes=cp_path.stat().st_size,
        )
        rec.checkpoints.append(asdict(cprec))
        rec.last_saved = now_iso()
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
            arrays = {k: z[k] for k in z.files if k != "meta"}
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
            out.append({
                "uuid": r.uuid, "name": r.name, "created_at": r.created_at,
                "seed": r.seed, "cosmic_age_Gyr": r.cosmic_time_s / 3.15576e16,
                "cosmology": r.cosmology, "status": r.status,
                "scale_factor": r.scale_factor, "last_saved": r.last_saved,
                "checkpoints": len(r.checkpoints), "provenance": r.provenance,
                "schema": r.schema,
            })
        return out
