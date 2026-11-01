"""Simulation session for the observatory: builds a verified 1D system from a preset,
advances it on STEP BOUNDARIES only (so pause/resume/save/load replay bitwise), and exposes
JSON-safe snapshots. This is a 1D laboratory simulation in dimensionless units, not a 3D universe."""
import hashlib
import json
import threading
import time

import numpy as np

from ..solvers.hydro import Euler1D
from ..solvers.nbody import GravitySystem1D

SAVE_SCHEMA = "observatory-save/v1"
DRIFT_WARNING_LEVEL = 1e-3          # benchmarks verify drift <= ~1e-3 in the regimes with exact references
SCIENTIFIC_STATUS = ("1D periodic laboratory simulation, dimensionless code units. Solvers verified by "
                     "`universe benchmark hydro|gravity|nbody` (T1 numerical verification). Not a 3D universe: "
                     "no cosmological expansion, thermochemistry, stars, galaxies or life.")

PRESETS = {
    "zeldovich": {"doc": "Cold collisionless sheets collapsing under self-gravity (exact solution before shell crossing).",
                  "defaults": {"n": 64, "ppc": 8, "A": 0.02, "G": 1.0},
                  "limits": {"n": (16, 512), "ppc": (1, 32), "A": (1e-6, 0.12), "G": (0.0, 10.0)}},
    "gas_particles": {"doc": "Pressure-supported gas plus collisionless matter in a Jeans-unstable box.",
                      "defaults": {"n": 64, "ppc": 8, "A": 0.05, "G": 1.0, "gas_fraction": 0.5},
                      "limits": {"n": (16, 512), "ppc": (1, 32), "A": (1e-6, 0.3), "G": (0.0, 10.0),
                                 "gas_fraction": (0.05, 0.95)}},
}


class ParamError(ValueError):
    pass


def resolve_params(preset, params):
    if preset not in PRESETS:
        raise ParamError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    spec = PRESETS[preset]
    out = dict(spec["defaults"])
    for k, v in (params or {}).items():
        if k not in out:
            raise ParamError(f"unknown parameter {k!r} for preset {preset!r}")
        out[k] = v
    for k, (lo, hi) in spec["limits"].items():
        v = out[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not lo <= v <= hi:
            raise ParamError(f"parameter {k}={v!r} must be a number in [{lo}, {hi}]")
    out["n"], out["ppc"] = int(out["n"]), int(out["ppc"])
    return out


def build_system(preset, p):
    n, ppc = p["n"], p["ppc"]
    if preset == "zeldovich":
        N = n * ppc
        q = (np.arange(N) + 0.5) / N
        return GravitySystem1D(x=q + p["A"] * np.sin(2 * np.pi * q), v=np.zeros(N), mass=1.0 / N,
                               box=(0.0, 1.0), n=n, G=p["G"], kick_cfl=0.1 * 32 / n)
    L, gam, k = 2.0, 1.4, np.pi
    rg, rp = p["gas_fraction"], 1.0 - p["gas_fraction"]
    x = (np.arange(n) + 0.5) * L / n
    d = p["A"] * np.sin(k * x)
    h = Euler1D(rg * (1 + d), np.zeros(n), (rg / gam) * (1 + gam * d), x0=0.0, x1=L, gamma=gam)
    N = n * ppc
    q = (np.arange(N) + 0.5) * L / N
    return GravitySystem1D(hydro=h, x=q + (p["A"] / k) * np.cos(k * q), v=np.zeros(N), mass=rp * L / N,
                           G=p["G"], kick_cfl=0.1 * 64 / n)


class Session:
    def __init__(self, preset="zeldovich", params=None):
        self.preset = preset
        self.params = resolve_params(preset, params)
        self.system = build_system(preset, self.params)
        self.lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()
        self.running = False
        self.version = 0                       # increments on every state change (for streaming)

    # ---------------------------------------------------------------- identity
    def config_hash(self):
        blob = json.dumps({"preset": self.preset, "params": self.params, "schema": SAVE_SCHEMA}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- control
    def step(self, n=1):
        """Advance exactly n whole steps (never clipped), so every pause point is a step boundary."""
        with self.lock:
            if self.running:
                raise RuntimeError("cannot step manually while running; pause first")
            for _ in range(int(n)):
                self.system.step()
            self.version += 1

    def run(self, max_steps=None, delay=0.0):
        with self.lock:
            if self.running:
                return
            self._stop.clear()
            self.running = True
        def loop():
            done = 0
            while not self._stop.is_set() and (max_steps is None or done < max_steps):
                with self.lock:
                    self.system.step()
                    self.version += 1
                done += 1
                if delay:
                    time.sleep(delay)
            with self.lock:
                self.running = False
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def pause(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        with self.lock:
            self.running = False
            self.version += 1

    # ---------------------------------------------------------------- observation
    def status(self):
        with self.lock:
            s = self.system
            gas_p, part_p = s.momentum()
            drift = s.energy_drift()
            warnings = []
            if drift > DRIFT_WARNING_LEVEL:
                warnings.append(f"energy drift {drift:.1e} exceeds {DRIFT_WARNING_LEVEL:.0e}: strongly nonlinear regime "
                                "beyond the verified benchmarks (exact-solution tests cover linear growth and pre-shell-crossing "
                                "collapse only). Conservation is checked; accuracy is not.")
            return {"preset": self.preset, "params": self.params, "config_hash": self.config_hash(), "warnings": warnings,
                    "t": s.t, "steps": s.steps, "running": self.running, "version": self.version,
                    "energy_drift": drift, "momentum_gas": gas_p, "momentum_particles": part_p,
                    "total_momentum": gas_p + part_p, "scientific_status": SCIENTIFIC_STATUS}

    def snapshot(self, max_particles=2000):
        with self.lock:
            s = self.system
            rho = s.density()
            phi, g = s.fields()
            snap = {"schema": "observatory-snapshot/v1", **self.status(), "x0": s.x0, "x1": s.x1, "n": s.n,
                    "density": rho.tolist(), "g": g.tolist(), "phi": phi.tolist()}
            if s.hydro is not None:
                r, u, p = s.hydro.primitives()
                snap.update(gas_density=r.tolist(), gas_velocity=u.tolist(), gas_pressure=p.tolist())
            if s.x is not None:
                idx = np.arange(s.x.size)
                if s.x.size > max_particles:
                    idx = idx[:: int(np.ceil(s.x.size / max_particles))]
                snap.update(particle_x=s.x[idx].tolist(), particle_v=s.v[idx].tolist(), particles_total=int(s.x.size))
            return snap

    def inspect(self, x):
        with self.lock:
            s = self.system
            if not s.x0 <= x < s.x1:
                raise ParamError(f"x must be in [{s.x0}, {s.x1})")
            i = min(int((x - s.x0) / s.dx), s.n - 1)
            rho = s.density()
            phi, g = s.fields()
            out = {"x": x, "cell": i, "cell_centre": s.x0 + (i + 0.5) * s.dx, "density_total": float(rho[i]),
                   "potential": float(phi[i]), "acceleration": float(g[i]), "t": s.t}
            if s.hydro is not None:
                r, u, p = s.hydro.primitives()
                out.update(gas_density=float(r[i]), gas_velocity=float(u[i]), gas_pressure=float(p[i]),
                           sound_speed=float(np.sqrt(s.hydro.gamma * p[i] / r[i])))
            if s.x is not None:
                lo, hi = s.x0 + i * s.dx, s.x0 + (i + 1) * s.dx
                m = (s.x >= lo) & (s.x < hi)
                out.update(particles_in_cell=int(m.sum()), mean_particle_velocity=float(s.v[m].mean()) if m.any() else None)
            return out

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        with self.lock:
            if self.running:
                raise RuntimeError("pause before saving")
            cp = self.system.checkpoint()
            arrays = {}
            if cp["hydro"] is not None:
                for k, v in cp["hydro"].items():
                    arrays[f"hydro_{k}"] = np.asarray(v)
            if cp["x"] is not None:
                arrays["x"], arrays["v"] = cp["x"], cp["v"]
            meta = {"schema": SAVE_SCHEMA, "preset": self.preset, "params": self.params,
                    "config_hash": self.config_hash(), "t": cp["t"], "steps": cp["steps"], "E0": cp["E0"],
                    "has_hydro": cp["hydro"] is not None}
            np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
            return meta

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("schema") != SAVE_SCHEMA:
                raise ParamError(f"unsupported save schema {meta.get('schema')!r}")
            ses = cls(meta["preset"], meta["params"])
            if ses.config_hash() != meta["config_hash"]:
                raise ParamError("config hash mismatch: save file does not match its own parameters")
            hydro = None
            if meta["has_hydro"]:
                hydro = {k[6:]: z[k] for k in z.files if k.startswith("hydro_")}
                for key in ("t", "steps", "rejected_steps", "fallback_interfaces"):
                    hydro[key] = hydro[key].item()
            cp = {"hydro": hydro, "t": meta["t"], "steps": meta["steps"], "E0": meta["E0"],
                  "x": z["x"] if "x" in z.files else None, "v": z["v"] if "v" in z.files else None}
            ses.system.restore(cp)
            ses.version += 1
            return ses
