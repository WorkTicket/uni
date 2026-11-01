"""FastAPI observatory service: create / step / run / pause / inspect / save / load a real simulation
and stream its snapshots over a WebSocket. One session per app instance.
Start:  universe-observatory --port 8000   then open http://127.0.0.1:8000/   (3D universe viewer; /lab = 1D laboratory viewer)"""
import argparse
import asyncio
import json
import os
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .session import PRESETS, ParamError, Session
from ..multiverse import MultiverseManager, MultiverseError
from .research_lab import ResearchLab, ResearchLabError
from ..habitat import Habitat, make_default_network
from ..planet.model import Planet, PlanetSpec
from ..core.clocks import FLWRCosmology
from ..research.persistent_organism import (
    PersistentOrganism, make_default_habitat, make_organism,
)
from ..geometry.geometry_core import (
    flat_minkowski_4d, flat_minkowski_5d, circle_compactification,
    GeometrySpec,
)
from ..frontier.compact_dimensions import (
    CircleCompactification, ToroidalCompactification, SphereCompactification,
    CompactDimensionExplorer, verify_kk_circle_spectrum, verify_kk_torus_spectrum,
    verify_sphere_spectrum,
)
from ..frontier.planck_scale import PLANCK_LAYER, PlanckUnits
from ..frontier.brane_world import BraneWorldModel, BraneWorldParameters
from ..solvers.pm3d import Cosmology, ParticleMesh3D

STATIC = Path(__file__).parent / "static"
MAX_STEPS_PER_REQUEST = 10_000
DEFAULT_COSMOLOGY = {"H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_r": 0.000092, "Omega_Lambda": 0.684908, "spectral_index": 0.965}
SAVE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
UNIVERSE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
HABITAT_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _pm3d_lattice(n_ppax: int, seed: int, amplitude: float):
    """Particles on a perturbed lattice for PM3D initial conditions."""
    rng = np.random.default_rng(seed)
    n = n_ppax
    grid = np.stack(np.meshgrid(
        *([np.arange(n) / n + 0.5 / n] * 3), indexing="ij"), -1).reshape(-1, 3)
    if amplitude > 0:
        grid = grid + amplitude * rng.normal(size=grid.shape)
        grid = grid % 1.0
    mass = np.full(grid.shape[0], 1.0 / grid.shape[0])
    return grid, mass


def _pm3d_snapshot_json(sim, max_points: int = 2000) -> dict:
    """Convert a PM3D snapshot to bounded JSON-safe form.

    Large arrays (positions, velocities, density field) are decimated to a
    bounded number of points so frames can be streamed without converting
    enormous numerical arrays to JSON. Full-resolution state is available via
    the dedicated get endpoint when needed.
    """
    s = sim.snapshot()
    n_particles = int(s["n_particles"])

    def decimate(arr: np.ndarray) -> list:
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] <= max_points:
            return arr.tolist()
        idx = np.linspace(0, arr.shape[0] - 1, max_points).round().astype(int)
        return arr[idx].tolist()

    delta = np.asarray(s["density_contrast"], dtype=float)
    out = {
        "schema": s["schema"],
        "a": float(s["a"]),
        "t": float(s["t"]),
        "t_Gyr": float(s["t_Gyr"]),
        "H_s": float(s["H_s"]),
        "z": float(s["z"]),
        "box_size_Mpc": float(s["box_size_Mpc"]),
        "n_mesh": int(s["n_mesh"]),
        "n_particles": n_particles,
        "returned_points": int(len(decimate(np.asarray(s["x_comoving"])))),
        "x_comoving": decimate(np.asarray(s["x_comoving"])),
        "x_comoving_Mpc": decimate(np.asarray(s["x_comoving_Mpc"])),
        "x_proper_Mpc": decimate(np.asarray(s["x_proper_Mpc"])),
        "v_pec_km_s": decimate(np.asarray(s["v_pec_km_s"])),
        "density_stats": {
            "mean": float(delta.mean()),
            "rms": float(np.std(delta)),
            "min": float(delta.min()),
            "max": float(delta.max()),
        },
        "mass_is_physical": bool(s.get("mass_is_physical", False)),
        "comoving_mean_density_Msun_Mpc3": float(s.get("comoving_mean_density_Msun_Mpc3", 0.0)),
        "proper_mean_density_Msun_Mpc3": float(s.get("proper_mean_density_Msun_Mpc3", 0.0)),
        "units": s.get("units", {}),
    }
    return out


def load_cosmology():
    """Background cosmology for the 3D viewer: the engine's config/reference.yaml when it can be found, else Planck-like defaults."""
    import sys
    cands = [Path("config/reference.yaml"), Path(__file__).parents[3] / "config" / "reference.yaml"]
    if getattr(sys, "_MEIPASS", None): cands.insert(0, Path(sys._MEIPASS) / "config" / "reference.yaml")     # PyInstaller bundle
    for cand in cands:
        try:
            import yaml
            cfg = yaml.safe_load(cand.read_text())
            c, f = cfg["cosmology"], cfg.get("field", {})
            return {"H0_km_s_Mpc": float(c["H0_km_s_Mpc"]), "Omega_m": float(c["Omega_m"]), "Omega_r": float(c["Omega_r"]),
                    "Omega_Lambda": float(c["Omega_Lambda"]), "spectral_index": float(f.get("spectral_index", 0.965)), "source": str(cand)}
        except Exception:
            continue
    return dict(DEFAULT_COSMOLOGY, source="built-in defaults")


def create_app(save_dir="runs/observatory"):
    app = FastAPI(title="Universe Observatory", version="0.1")
    holder = {"session": Session("zeldovich")}
    save_root = Path(save_dir)
    multiverse = MultiverseManager(save_root.parent / "multiverse")
    research_lab = ResearchLab(save_root.parent / "research_lab")
    habitat_holder: Dict[str, Habitat] = {}
    organism_holder: Dict[str, PersistentOrganism] = {}
    # Scientific 3D PM3D cosmology simulation (actual numerical state, not procedural).
    pm3d_holder: Dict[str, ParticleMesh3D] = {}
    pm3d_playback: Dict[str, dict] = {}
    pm3d_universe: Dict[str, str] = {}   # pm3d sid -> multiverse universe uid
    pm3d_lock = threading.RLock()

    def _pm3d_advance(sid: str, st: dict) -> None:
        """Advance one playing PM3D sim by a small d ln a (continuous playback)."""
        with pm3d_lock:
            sim = pm3d_holder.get(sid)
        if sim is None:
            return
        speed = float(st.get("speed", 1.0))
        a_max = float(st.get("a_max", 10.0))
        d_ln_a = 0.002 * speed
        a_end = sim.a * math.exp(d_ln_a)
        if a_end > a_max:
            st["playing"] = False
            return
        sim.run_to_a(a_end, max_d_ln_a=0.01)
        st["a_current"] = float(sim.a)
        st["t_current"] = float(sim.t)

    def _pm3d_playback_worker() -> None:
        """Background thread: advance playing PM3D sims continuously."""
        while True:
            time.sleep(0.05)
            with pm3d_lock:
                items = list(pm3d_playback.items())
            for sid, st in items:
                if not st.get("playing"):
                    continue
                try:
                    _pm3d_advance(sid, st)
                except Exception:
                    pass

    threading.Thread(target=_pm3d_playback_worker, daemon=True).start()

    @app.exception_handler(ParamError)
    async def _param_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(RuntimeError)
    async def _runtime_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(MultiverseError)
    async def _multiverse_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(ResearchLabError)
    async def _research_lab_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    def ses():
        return holder["session"]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/lab")
    def lab():
        return FileResponse(STATIC / "lab.html")

    @app.get("/research-lab")
    def research_lab_page():
        return FileResponse(STATIC / "research_lab.html")

    @app.get("/hyperspace")
    def hyperspace_page():
        return FileResponse(STATIC / "hyperspace.html")

    @app.get("/api/cosmology")
    def cosmology():
        return load_cosmology()

    @app.get("/api/presets")
    def presets():
        return {k: {"doc": v["doc"], "defaults": v["defaults"], "limits": v["limits"]} for k, v in PRESETS.items()}

    @app.post("/api/session")
    def new_session(body: dict = Body(default={})):
        old = holder["session"]
        if old.running:
            old.pause()
        holder["session"] = Session(body.get("preset", "zeldovich"), body.get("params"))
        return holder["session"].status()

    @app.get("/api/state")
    def state():
        return ses().status()

    @app.get("/api/snapshot")
    def snapshot():
        return ses().snapshot()

    @app.post("/api/step")
    def step(body: dict = Body(default={})):
        n = body.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_STEPS_PER_REQUEST:
            raise ParamError(f"n must be an integer in [1, {MAX_STEPS_PER_REQUEST}]")
        ses().step(n)
        return ses().status()

    @app.post("/api/run")
    def run(body: dict = Body(default={})):
        ms, delay = body.get("max_steps"), body.get("delay", 0.0)
        if ms is not None and (isinstance(ms, bool) or not isinstance(ms, int) or ms < 1):
            raise ParamError("max_steps must be a positive integer or omitted")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 1:
            raise ParamError("delay must be a number in [0, 1] seconds")
        ses().run(ms, delay)
        return ses().status()

    @app.post("/api/pause")
    def pause():
        ses().pause()
        return ses().status()

    @app.get("/api/inspect")
    def inspect(x: float = Query(...)):
        return ses().inspect(x)

    @app.post("/api/save")
    def save(body: dict = Body(default={})):
        sid = body.get("id") or time.strftime("save-%Y%m%dT%H%M%S")
        if not SAVE_ID.match(str(sid)):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        save_root.mkdir(parents=True, exist_ok=True)
        meta = ses().save(save_root / f"{sid}.npz")
        return {"id": sid, "meta": meta}

    @app.get("/api/saves")
    def saves():
        return {"saves": sorted(p.stem for p in save_root.glob("*.npz"))} if save_root.exists() else {"saves": []}

    # ---- multiverse ----
    @app.get("/api/multiverse")
    def multiverse_list():
        return {"universes": multiverse.to_public()}

    @app.post("/api/multiverse")
    def multiverse_create(body: dict = Body(default={})):
        name = body.get("name", "")
        seed = body.get("seed")
        cosmology = body.get("cosmology")
        model_settings = body.get("model_settings")
        rec = multiverse.create_universe(name=name, seed=seed, cosmology=cosmology, model_settings=model_settings)
        return {"universe": rec.to_dict(), "provenance": rec.provenance}

    @app.get("/api/multiverse/{uid}")
    def multiverse_get(uid: str):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        rec = multiverse.get(uid)
        return {"universe": rec.to_dict(), "provenance": rec.provenance, "checkpoints": multiverse.list_checkpoints(uid)}

    @app.patch("/api/multiverse/{uid}/name")
    def multiverse_rename(uid: str, body: dict = Body(...)):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        rec = multiverse.rename(uid, body.get("name", ""))
        return {"universe": rec.to_dict()}

    @app.delete("/api/multiverse/{uid}")
    def multiverse_delete(uid: str):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        multiverse.delete(uid)
        return {"ok": True}

    @app.post("/api/multiverse/{uid}/time")
    def multiverse_set_time(uid: str, body: dict = Body(...)):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        rec = multiverse.set_time(uid, float(body.get("cosmic_time_s", 0)), float(body.get("scale_factor", 1e-13)))
        return {"universe": rec.to_dict()}

    @app.post("/api/multiverse/{uid}/checkpoint")
    def multiverse_checkpoint(uid: str, body: dict = Body(default={})):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        cp = multiverse.checkpoint(uid, body.get("id"), body.get("arrays"))
        return {"checkpoint": asdict(cp)}

    @app.get("/api/multiverse/{uid}/checkpoints")
    def multiverse_checkpoints(uid: str):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        return {"checkpoints": [asdict(c) for c in multiverse.list_checkpoints(uid)]}

    @app.post("/api/multiverse/{uid}/checkpoints/{cid}/restore")
    def multiverse_restore(uid: str, cid: str):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        return multiverse.restore_checkpoint(uid, cid)

    @app.post("/api/multiverse/{uid}/branch")
    def multiverse_branch(uid: str, body: dict = Body(default={})):
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        rec = multiverse.branch(uid, body.get("name", f"branch of {uid}"), body.get("from_checkpoint"), body.get("seed"))
        return {"universe": rec.to_dict(), "provenance": rec.provenance}

    # ---- bulk geometry (Requirement 12: Bulk geometry view) ----
    @app.get("/api/geometry/flat4d")
    def geometry_flat4d():
        spec = flat_minkowski_4d()
        return spec.to_dict()

    @app.get("/api/geometry/flat5d")
    def geometry_flat5d():
        spec = flat_minkowski_5d()
        return spec.to_dict()

    @app.get("/api/geometry/circle/{radius}")
    def geometry_circle(radius: float):
        spec = circle_compactification(radius)
        return spec.to_dict()

    @app.post("/api/geometry/circle")
    def geometry_circle_create(body: dict = Body(default={})):
        radius = float(body.get("radius", 1.0))
        spec = circle_compactification(radius)
        return spec.to_dict()

    @app.get("/api/kk/circle/{radius}")
    def kk_circle(radius: float, max_mode: int = 10):
        result = verify_kk_circle_spectrum(radius, max_mode=max_mode)
        return result

    @app.post("/api/kk/torus")
    def kk_torus(body: dict = Body(default={})):
        radii = body.get("radii", [1.0, 2.0])
        modes = body.get("modes", [[1, 0], [0, 1], [1, 1]])
        result = verify_kk_torus_spectrum(radii, modes)
        return result

    @app.get("/api/kk/sphere/{radius}")
    def kk_sphere(radius: float, max_l: int = 5):
        result = verify_sphere_spectrum(radius, max_l=max_l)
        return result

    @app.get("/api/explorer/compact/{radius}")
    def compact_explorer(radius: float, scales: str = "0.1,1.0,5.0"):
        scale_list = [float(s) for s in scales.split(",")]
        explorer = CompactDimensionExplorer(radius)
        return explorer.visibility_vs_scale(scale_list)

    # ---- brane dynamics (Requirement 12: Brane dynamics view) ----
    @app.get("/api/brane/parameters")
    def brane_parameters():
        params = BraneWorldParameters()
        params.derive_4d_constants()
        return {
            "Lambda_5": params.Lambda_5,
            "G_5": params.G_5,
            "k": params.k,
            "lambda_brane": params.lambda_brane,
            "G_4": params.G_4,
            "Lambda_4": params.Lambda_4,
            "classification": "speculative_research",
        }

    @app.post("/api/brane/simulate")
    def brane_simulate(body: dict = Body(default={})):
        params = BraneWorldParameters(
            Lambda_5=float(body.get("Lambda_5", -6.0)),
            G_5=float(body.get("G_5", 1.0)),
            lambda_brane=float(body.get("lambda_brane", 10.0)),
            junction_tol=float(body.get("junction_tol", 1e12)),
        )
        params.derive_4d_constants()
        model = BraneWorldModel(params)
        a0 = float(body.get("a_start", 1e-10))
        rho_r = float(body.get("rho_radiation", 1e8))
        state = model.initialize(t=0.0, a_brane=a0, rho_radiation=rho_r)
        steps = int(body.get("steps", 10))
        dt = float(body.get("dt", 0.01))
        history = []
        for _ in range(steps):
            state = model.step(dt)
            history.append({
                "t": state.t, "a": state.a_brane, "H": state.H_brane,
                "rho": state.rho_matter + state.rho_radiation,
                "validity": state.validity,
                "junction_residual": state.junction_residual,
            })
        return {
            "parameters": {"Lambda_5": params.Lambda_5, "G_5": params.G_5,
                          "lambda_brane": params.lambda_brane, "G_4": params.G_4},
            "initial": {"a": state.a_brane, "H": state.H_brane},
            "history": history,
            "classification": "speculative_research",
        }

    # ---- Planck-scale / quantum geometry laboratory (Requirement 12) ----
    @app.get("/api/planck/units")
    def planck_units():
        return PLANCK_LAYER.to_dict()

    @app.get("/api/planck/length")
    def planck_length():
        return {"planck_length_m": PLANCK_LAYER.planck_length()}

    @app.get("/api/planck/time")
    def planck_time():
        return {"planck_time_s": PLANCK_LAYER.planck_time()}

    @app.get("/api/planck/mass")
    def planck_mass():
        return {"planck_mass_kg": PLANCK_LAYER.planck_mass()}

    @app.get("/api/planck/energy")
    def planck_energy():
        return {"planck_energy_J": PLANCK_LAYER.planck_energy()}

    @app.post("/api/planck/discrete/step")
    def planck_discrete_step(body: dict = Body(default={})):
        model = PLANCK_LAYER.models["discrete_geometry"]
        state = body.get("state")
        if state is None:
            state = model.initial_state()
        dt = float(body.get("dt", 1e-43))
        state = model.step(state, dt)
        obs = model.observables(state)
        return {"state": state, "observables": obs}

    @app.post("/api/planck/spinnetwork/step")
    def planck_spinnetwork_step(body: dict = Body(default={})):
        model = PLANCK_LAYER.models["spin_network_inspired"]
        state = body.get("state")
        if state is None:
            state = model.initial_state()
        dt = float(body.get("dt", 1e-43))
        state = model.step(state, dt)
        obs = model.observables(state)
        return {"state": state, "observables": obs}

    # ---- origin events (Requirement 12: Multiverse overview) ----
    @app.get("/api/origin/events")
    def origin_events():
        return {"origins": multiverse.list_origin_events()}

    # ---- research laboratory ----
    EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

    @app.get("/api/lab")
    def lab_list():
        return {"experiments": [r.to_dict() for r in research_lab.list_experiments()]}

    @app.post("/api/lab")
    def lab_create(body: dict = Body(default={})):
        name = body.get("name", "")
        config = body.get("config")
        seed = body.get("seed")
        rec = research_lab.create_experiment(name=name, config=config, seed=seed)
        return {"experiment": rec.to_dict()}

    @app.get("/api/lab/{eid}")
    def lab_get(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        rec = research_lab.get(eid)
        return {"experiment": rec.to_dict()}

    @app.delete("/api/lab/{eid}")
    def lab_delete(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        research_lab.delete(eid)
        return {"ok": True}

    @app.post("/api/lab/{eid}/step")
    def lab_step(eid: str, body: dict = Body(default={})):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        n = body.get("n", 1)
        dt_gyr = body.get("dt_gyr")
        return research_lab.step(eid, n=int(n), dt_gyr=float(dt_gyr) if dt_gyr is not None else None)

    @app.post("/api/lab/{eid}/run")
    def lab_run(eid: str, body: dict = Body(default={})):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        max_steps = body.get("max_steps")
        dt_gyr = body.get("dt_gyr")
        return research_lab.run(eid, max_steps=int(max_steps) if max_steps is not None else None,
                                dt_gyr=float(dt_gyr) if dt_gyr is not None else None)

    @app.post("/api/lab/{eid}/pause")
    def lab_pause(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return {"experiment": research_lab.pause(eid).to_dict()}

    @app.get("/api/lab/{eid}/stats")
    def lab_stats(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return research_lab.stats(eid)

    @app.get("/api/lab/{eid}/history")
    def lab_history(eid: str, limit: int = 0):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return {"history": research_lab.history(eid, limit=limit or None)}

    @app.get("/api/lab/{eid}/trajectory")
    def lab_trajectory(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return research_lab.trajectory(eid)

    @app.get("/api/lab/{eid}/lineage")
    def lab_lineage(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return research_lab.lineage(eid)

    @app.get("/api/lab/{eid}/extinctions")
    def lab_extinctions(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return {"extinctions": research_lab.extinction_events(eid)}

    @app.post("/api/lab/{eid}/checkpoint")
    def lab_checkpoint(eid: str):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return research_lab.checkpoint(eid)

    @app.post("/api/lab/{eid}/restore")
    def lab_restore(eid: str, body: dict = Body(...)):
        if not EXPERIMENT_ID.match(eid):
            raise ParamError("invalid experiment id")
        return {"experiment": research_lab.restore(eid, body).to_dict()}

    @app.post("/api/load")
    def load(body: dict = Body(...)):
        sid = str(body.get("id", ""))
        if not SAVE_ID.match(sid):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        path = save_root / f"{sid}.npz"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no save named {sid!r}")
        if holder["session"].running:
            holder["session"].pause()
        holder["session"] = Session.load(path)
        return holder["session"].status()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        last = None
        try:
            while True:
                s = ses()
                if s.version != last or id(s) != getattr(ws, "_sid", None):
                    last = s.version
                    await websocket.send_json(s.snapshot())
                await asyncio.sleep(0.03)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ---- read-only planetary habitat API ----
    HABITAT_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

    @app.post("/api/habitat")
    def habitat_create(body: dict = Body(default={})):
        """Create a new planetary habitat (read-only observation thereafter via GET endpoints).

        Body: { "id": str, "seed": int, "n_patches": int, "t_gyr": float,
                "has_water": bool, "a_au": float }
        """
        hid = body.get("id") or time.strftime("hab-%Y%m%dT%H%M%S")
        if not HABITAT_ID.match(str(hid)):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        if hid in habitat_holder:
            raise ParamError(f"habitat {hid!r} already exists")
        seed = int(body.get("seed", 0))
        n_patches = int(body.get("n_patches", 8))
        t_gyr = float(body.get("t_gyr", 4.5))
        has_water = bool(body.get("has_water", True))
        a_au = float(body.get("a_au", 1.0))
        spec = PlanetSpec(mass_mearth=1.0, a_au=a_au, ocean_fraction=0.71,
                          has_surface_water=has_water, tectonics="mobile", seed=seed)
        planet = Planet(spec, t_gyr=t_gyr)
        network = make_default_network()
        habitat_holder[hid] = Habitat(planet, network, n_patches=n_patches, seed=seed)
        return {"id": hid, "status": "created", "state": habitat_holder[hid].state()}

    @app.get("/api/habitat/{hid}")
    def habitat_get(hid: str):
        if not HABITAT_ID.match(hid):
            raise ParamError("invalid habitat id")
        if hid not in habitat_holder:
            raise HTTPException(status_code=404, detail=f"no habitat named {hid!r}")
        return habitat_holder[hid].state()

    @app.get("/api/habitat/{hid}/chemistry")
    def habitat_chemistry(hid: str):
        if not HABITAT_ID.match(hid):
            raise ParamError("invalid habitat id")
        if hid not in habitat_holder:
            raise HTTPException(status_code=404, detail=f"no habitat named {hid!r}")
        h = habitat_holder[hid]
        return {
            "t_gyr": h.t_gyr,
            "species": h.species_names,
            "patches": [{
                "index": p.index,
                "c": p.c.tolist(),
                "temperature_K": p.temperature_K,
                "pH": p.pH,
                "liquid_water": p.liquid_water,
                "nutrient_mol": p.nutrient_mol,
                "waste_mol": p.waste_mol,
                "volume_L": p.volume_L,
            } for p in h.patches],
        }

    @app.get("/api/habitat/{hid}/population")
    def habitat_population(hid: str):
        if not HABITAT_ID.match(hid):
            raise ParamError("invalid habitat id")
        if hid not in habitat_holder:
            raise HTTPException(status_code=404, detail=f"no habitat named {hid!r}")
        h = habitat_holder[hid]
        st = h.state()
        rep = h.replicators
        return {
            "t_gyr": h.t_gyr,
            "sterile": st.sterile,
            "replicators": st.replicators,
            "pool_monomer": st.pool_monomer,
            "total_replicators": sum(st.replicators.values()),
            "lineage": rep.lineage if rep is not None else {},
        }

    @app.get("/api/habitat/{hid}/lineage")
    def habitat_lineage(hid: str):
        if not HABITAT_ID.match(hid):
            raise ParamError("invalid habitat id")
        if hid not in habitat_holder:
            raise HTTPException(status_code=404, detail=f"no habitat named {hid!r}")
        h = habitat_holder[hid]
        rep = h.replicators
        if rep is None:
            return {"t_gyr": h.t_gyr, "lineages": {}, "n_lineages": 0}
        return {
            "t_gyr": h.t_gyr,
            "lineages": rep.lineage,
            "n_lineages": len(rep.lineage),
        }

    @app.get("/api/habitat/{hid}/resources")
    def habitat_resources(hid: str):
        if not HABITAT_ID.match(hid):
            raise ParamError("invalid habitat id")
        if hid not in habitat_holder:
            raise HTTPException(status_code=404, detail=f"no habitat named {hid!r}")
        h = habitat_holder[hid]
        return {
            "t_gyr": h.t_gyr,
            "pool_monomer": h.replicators.pool if h.replicators is not None else 0,
            "patches": [{
                "index": p.index,
                "nutrient_mol": p.nutrient_mol,
                "waste_mol": p.waste_mol,
                "volume_L": p.volume_L,
            } for p in h.patches],
            "mass_ledger_error": h._ledger_error,
        }

    # ---- persistent organism + Mind Inspector ----
    ORGANISM_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

    @app.post("/api/organism")
    def organism_create(body: dict = Body(default={})):
        """Create a persistent organism in a habitat (or create a habitat).

        Body: { "id": str, "habitat_id": str | null, "seed": int,
                "n_patches": int, "patch_index": int, "architecture": str }
        If habitat_id is omitted a fresh default habitat is created.
        """
        oid = body.get("id") or time.strftime("org-%Y%m%dT%H%M%S")
        if not ORGANISM_ID.match(str(oid)):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        if oid in organism_holder:
            raise ParamError(f"organism {oid!r} already exists")

        hid = body.get("habitat_id")
        if hid is not None:
            if hid not in habitat_holder:
                raise HTTPException(status_code=404, detail=f"no habitat named {hid!r}")
            habitat = habitat_holder[hid]
        else:
            habitat = make_default_habitat(
                seed=int(body.get("seed", 0)), n_patches=int(body.get("n_patches", 4)))

        org = make_organism(
            architecture=str(body.get("architecture", "ecs_iw")),
            seed=int(body.get("seed", 0)),
            habitat=habitat,
            patch_index=int(body.get("patch_index", 0)),
        )
        organism_holder[oid] = org
        # Register the habitat too so its chemistry stays addressable.
        if hid is None:
            hid = f"hab-for-{oid}"
            habitat_holder[hid] = habitat
        return {"id": oid, "habitat_id": hid, "state": org.inspector_state()}

    @app.get("/api/organism/saves")
    def organism_saves():
        org_dir = save_root / "organisms"
        if not org_dir.exists():
            return {"saves": []}
        return {"saves": sorted(p.stem for p in org_dir.glob("*.json"))}

    @app.get("/api/organism/{oid}")
    def organism_get(oid: str):
        if not ORGANISM_ID.match(oid):
            raise ParamError("invalid organism id")
        if oid not in organism_holder:
            raise HTTPException(status_code=404, detail=f"no organism named {oid!r}")
        return organism_holder[oid].inspector_state()

    @app.post("/api/organism/{oid}/step")
    def organism_step(oid: str, body: dict = Body(default={})):
        if not ORGANISM_ID.match(oid):
            raise ParamError("invalid organism id")
        if oid not in organism_holder:
            raise HTTPException(status_code=404, detail=f"no organism named {oid!r}")
        n = body.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1_000:
            raise ParamError("n must be an integer in [1, 1000]")
        org = organism_holder[oid]
        last = None
        for _ in range(n):
            last = org.step()
        return {"id": oid, "steps": n, "last": last,
                "state": org.inspector_state()}

    @app.get("/api/organism/{oid}/inspect")
    def organism_inspect(oid: str):
        """Mind Inspector view: full organism + habitat inspector state."""
        if not ORGANISM_ID.match(oid):
            raise ParamError("invalid organism id")
        if oid not in organism_holder:
            raise HTTPException(status_code=404, detail=f"no organism named {oid!r}")
        return organism_holder[oid].inspector_state()

    @app.post("/api/organism/{oid}/checkpoint")
    def organism_checkpoint(oid: str):
        if not ORGANISM_ID.match(oid):
            raise ParamError("invalid organism id")
        if oid not in organism_holder:
            raise HTTPException(status_code=404, detail=f"no organism named {oid!r}")
        return {"id": oid, "checkpoint": organism_holder[oid].checkpoint()}

    # ---- persistent organism save / load (disk-backed, process-restart safe) ----
    ORGANISM_SAVE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
    MAX_ORGANISM_CHECKPOINT_BYTES = 50 * 1024 * 1024
    ORGANISM_SCHEMA = "persistent-organism/v1"
    ORGANISM_REQUIRED_KEYS = ("identity", "t_gyr", "habitat", "body",
                              "memory", "controller", "self_model")

    @app.post("/api/organism/{oid}/save")
    def organism_save(oid: str, body: dict = Body(default={})):
        """Persist an organism checkpoint to disk (process-restart safe).

        Body: { "id": str }  (optional stable checkpoint id)
        """
        if not ORGANISM_ID.match(oid):
            raise ParamError("invalid organism id")
        if oid not in organism_holder:
            raise HTTPException(status_code=404, detail=f"no organism named {oid!r}")
        cid = str(body.get("id") or time.strftime("org-%Y%m%dT%H%M%S"))
        if not ORGANISM_SAVE_ID.match(cid):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        org = organism_holder[oid]
        cp = org.checkpoint()
        if cp.get("schema") != ORGANISM_SCHEMA:
            raise ParamError(f"unsupported organism checkpoint schema {cp.get('schema')!r}")
        payload = json.dumps(cp).encode("utf-8")
        if len(payload) > MAX_ORGANISM_CHECKPOINT_BYTES:
            raise ParamError("organism checkpoint exceeds size limit")
        org_dir = save_root / "organisms"
        org_dir.mkdir(parents=True, exist_ok=True)
        path = org_dir / f"{cid}.json"
        tmp = org_dir / f"{cid}.tmp.json"
        tmp.write_text(json.dumps(cp, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return {"id": cid, "schema": cp["schema"], "size_bytes": len(payload),
                "path": f"organisms/{cid}.json", "t_gyr": cp["t_gyr"],
                "organism_id": oid}

    @app.post("/api/organism/load")
    def organism_load(body: dict = Body(...)):
        """Restore an organism from a disk-backed checkpoint.

        Body: { "id": str, "organism_id": str | null }
        Rebuilds the shared habitat once from the checkpoint, then reconnects
        the organism to it by stable ID. Safe across a service restart.
        """
        cid = str(body.get("id", ""))
        if not ORGANISM_SAVE_ID.match(cid):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        org_dir = save_root / "organisms"
        path = org_dir / f"{cid}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no organism checkpoint named {cid!r}")
        raw = path.read_text(encoding="utf-8")
        if len(raw.encode("utf-8")) > MAX_ORGANISM_CHECKPOINT_BYTES:
            raise ParamError("organism checkpoint exceeds size limit")
        try:
            cp = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParamError(f"organism checkpoint is not valid JSON: {exc}")
        if not isinstance(cp, dict) or cp.get("schema") != ORGANISM_SCHEMA:
            raise ParamError(f"unsupported organism checkpoint schema {cp.get('schema')!r}")
        for key in ORGANISM_REQUIRED_KEYS:
            if key not in cp:
                raise ParamError(f"organism checkpoint missing required key {key!r}")
        hab_cp = cp.get("habitat", {})
        if not isinstance(hab_cp, dict) or "patches" not in hab_cp:
            raise ParamError("organism checkpoint habitat is incomplete")
        seed = int(hab_cp.get("seed", 0))
        n_patches = int(hab_cp.get("n_patches", len(hab_cp.get("patches", [])) or 8))
        network = make_default_network()
        fallback_planet = Planet(PlanetSpec(), t_gyr=float(hab_cp.get("t_gyr", 4.5)))
        habitat = Habitat.restore(fallback_planet, network, hab_cp,
                                  seed=seed, n_patches=n_patches)
        hid = f"hab-for-{cid}"
        habitat_holder[hid] = habitat
        oid = str(body.get("organism_id") or cp.get("identity", {}).get("organism_id") or cid)
        if not ORGANISM_ID.match(oid):
            oid = cid
        if oid in organism_holder:
            raise ParamError(f"organism {oid!r} already exists")
        org = PersistentOrganism.restore(cp, habitat)
        organism_holder[oid] = org
        return {"id": oid, "habitat_id": hid, "state": org.inspector_state()}

    # ---- scientific 3D PM3D cosmology simulation (actual numerical state) ----
    PM3D_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

    @app.post("/api/pm3d")
    def pm3d_create(body: dict = Body(default={})):
        """Create a 3D particle-mesh cosmology simulation from actual numerical state.

        Body: { "id": str, "seed": int, "n_mesh": int, "n_particles_per_axis": int,
                "a_start": float, "a_end": float, "amplitude": float }
        The simulation evolves cold dark matter under comoving PM gravity in an
        expanding FLRW background. State is the authoritative numerical result;
        the renderer displays it without fabricating structure.
        """
        sid = body.get("id") or time.strftime("pm3d-%Y%m%dT%H%M%S")
        if not PM3D_ID.match(str(sid)):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        if sid in pm3d_holder:
            raise ParamError(f"pm3d simulation {sid!r} already exists")
        seed = int(body.get("seed", 42))
        n_mesh = int(body.get("n_mesh", 16))
        n_ppax = int(body.get("n_particles_per_axis", 8))
        a_start = float(body.get("a_start", 0.2))
        a_end = float(body.get("a_end", 2.0))
        amplitude = float(body.get("amplitude", 0.02))
        box_size_Mpc = float(body.get("box_size_Mpc", 100.0))
        if not (8 <= n_mesh <= 64):
            raise ParamError("n_mesh must be in [8, 64]")
        if not (2 <= n_ppax <= 16):
            raise ParamError("n_particles_per_axis must be in [2, 16]")
        if not (0 < a_start < a_end):
            raise ParamError("require 0 < a_start < a_end")
        if not (0 < amplitude <= 0.3):
            raise ParamError("amplitude must be in (0, 0.3]")
        if not (0 < box_size_Mpc):
            raise ParamError("box_size_Mpc must be positive")
        cosmo = Cosmology(H0_km_s_Mpc=67.4, Omega_m=0.315, Omega_r=9.2e-5,
                          Omega_Lambda=0.684908)
        x, m = _pm3d_lattice(n_ppax, seed, amplitude)
        sim = ParticleMesh3D(n=n_mesh, x=x, u=np.zeros_like(x), mass=m,
                             cosmology=cosmo, a=a_start, box_size_Mpc=box_size_Mpc)
        # t(a) is initialized in the constructor via the background integral.
        pm3d_holder[sid] = sim
        # Register the PM3D volume as a persistent multiverse universe so its
        # cosmology, seed, box size, clock and checkpoints are owned by a
        # universe record rather than only a global in-memory holder.
        try:
            urec = multiverse.create_universe(
                name=sid, seed=seed,
                cosmology={"H0_km_s_Mpc": cosmo.H0_km_s_Mpc, "Omega_m": cosmo.Omega_m,
                           "Omega_r": cosmo.Omega_r, "Omega_Lambda": cosmo.Omega_Lambda,
                           "a_start": a_start, "a_end": a_end, "steps": 500,
                           "box_size_Mpc": box_size_Mpc},
                model_settings={"n_mesh": n_mesh, "n_particles_per_axis": n_ppax,
                                "amplitude": amplitude, "model": "pm3d"},
                origin_event=None,
            )
            pm3d_universe[sid] = urec.uuid
        except MultiverseError:
            # Universe may already exist (e.g. re-create with same id); link to it.
            existing = [u for u in multiverse.list_universes() if u.name == sid]
            if existing:
                pm3d_universe[sid] = existing[0].uuid
        return {"id": sid, "state": _pm3d_snapshot_json(sim),
                "classification": "SIMULATED", "universe": pm3d_universe.get(sid)}

    @app.get("/api/pm3d/{sid}")
    def pm3d_get(sid: str):
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        return {"id": sid, "state": _pm3d_snapshot_json(pm3d_holder[sid]),
                "classification": "SIMULATED"}

    @app.post("/api/pm3d/{sid}/step")
    def pm3d_step(sid: str, body: dict = Body(default={})):
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        a_end = float(body.get("a_end", pm3d_holder[sid].a * 1.1))
        max_d_ln_a = float(body.get("max_d_ln_a", 0.01))
        result = pm3d_holder[sid].run_to_a(a_end, max_d_ln_a=max_d_ln_a)
        return {"id": sid, "state": _pm3d_snapshot_json(pm3d_holder[sid]),
                "classification": "SIMULATED", "result": result}

    @app.post("/api/pm3d/{sid}/reset")
    def pm3d_reset(sid: str, body: dict = Body(default={})):
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        a_start = float(body.get("a_start", 0.2))
        amplitude = float(body.get("amplitude", 0.02))
        seed = int(body.get("seed", pm3d_holder[sid].mass.size and 42))
        cosmo = Cosmology(H0_km_s_Mpc=67.4, Omega_m=0.315, Omega_r=9.2e-5,
                          Omega_Lambda=0.684908)
        x, m = _pm3d_lattice(int(pm3d_holder[sid].n_mesh ** (1 / 3)), seed, amplitude)
        sim = ParticleMesh3D(n=pm3d_holder[sid].n_mesh, x=x, u=np.zeros_like(x),
                             mass=m, cosmology=cosmo, a=a_start)
        # t(a) is initialized in the constructor via the background integral.
        pm3d_holder[sid] = sim
        return {"id": sid, "state": _pm3d_snapshot_json(sim), "classification": "SIMULATED"}

    # ---- PM3D continuous playback control ----
    @app.post("/api/pm3d/{sid}/play")
    def pm3d_play(sid: str, body: dict = Body(default={})):
        """Start continuous playback of a PM3D simulation.

        Body: { "speed": float, "a_max": float }
        """
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        speed = float(body.get("speed", 1.0))
        a_max = float(body.get("a_max", 10.0))
        if speed <= 0:
            raise ParamError("speed must be positive")
        if a_max <= 0:
            raise ParamError("a_max must be positive")
        with pm3d_lock:
            pm3d_playback[sid] = {"playing": True, "speed": speed, "a_max": a_max,
                                  "a_current": float(pm3d_holder[sid].a),
                                  "t_current": float(pm3d_holder[sid].t)}
        return {"id": sid, "playing": True, "speed": speed, "a_max": a_max}

    @app.post("/api/pm3d/{sid}/pause")
    def pm3d_pause(sid: str):
        """Pause continuous playback of a PM3D simulation."""
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        with pm3d_lock:
            st = pm3d_playback.get(sid, {"playing": False})
            st["playing"] = False
            pm3d_playback[sid] = st
        return {"id": sid, "playing": False}

    @app.post("/api/pm3d/{sid}/speed")
    def pm3d_speed(sid: str, body: dict = Body(default={})):
        """Set the playback speed of a PM3D simulation."""
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        speed = float(body.get("speed", 1.0))
        if speed <= 0:
            raise ParamError("speed must be positive")
        with pm3d_lock:
            st = pm3d_playback.get(sid, {"playing": False, "speed": 1.0, "a_max": 10.0})
            st["speed"] = speed
            pm3d_playback[sid] = st
        return {"id": sid, "speed": speed}

    @app.get("/api/pm3d/{sid}/status")
    def pm3d_status(sid: str):
        """Return playback status and current state for a PM3D simulation."""
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        with pm3d_lock:
            st = dict(pm3d_playback.get(sid, {"playing": False, "speed": 1.0,
                                              "a_max": 10.0}))
        sim = pm3d_holder[sid]
        st["a"] = float(sim.a)
        st["t"] = float(sim.t)
        st["z"] = float(1.0 / sim.a - 1.0)
        return {"id": sid, **st}

    # ---- PM3D checkpoints (persisted via the multiverse, not a global holder) ----
    @app.post("/api/pm3d/{sid}/checkpoint")
    def pm3d_checkpoint(sid: str, body: dict = Body(default={})):
        """Save a PM3D simulation checkpoint tied to its multiverse universe."""
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        cid = body.get("id") or time.strftime("cp-%Y%m%dT%H%M%S")
        if not PM3D_ID.match(str(cid)):
            raise ParamError("id must match [A-Za-z0-9_-]{1,64}")
        sim = pm3d_holder[sid]
        cp = sim.checkpoint()
        arrays = {k: np.asarray(v) for k, v in cp.items()
                  if k in ("x", "u", "mass", "a", "t")}
        uid = pm3d_universe.get(sid)
        if uid is None:
            raise HTTPException(status_code=409, detail=f"pm3d {sid!r} is not bound to a universe")
        cprec = multiverse.checkpoint_universe(uid, cid, arrays)
        return {"id": sid, "universe": uid, "checkpoint_id": cprec.id,
                "a": float(sim.a), "t": float(sim.t),
                "path": f"{uid}/{cprec.id}.npz"}

    @app.post("/api/pm3d/{sid}/checkpoints/{cid}/restore")
    def pm3d_restore(sid: str, cid: str):
        """Restore a PM3D simulation from its multiverse universe checkpoint."""
        if not PM3D_ID.match(sid):
            raise ParamError("invalid pm3d id")
        if sid not in pm3d_holder:
            raise HTTPException(status_code=404, detail=f"no pm3d simulation {sid!r}")
        uid = pm3d_universe.get(sid)
        if uid is None:
            raise HTTPException(status_code=409, detail=f"pm3d {sid!r} is not bound to a universe")
        result = multiverse.restore_checkpoint(uid, cid)
        arrays = result.get("arrays", {})
        sim = pm3d_holder[sid]
        sim.restore({k: (v.item() if v.ndim == 0 else v) for k, v in arrays.items()
                     if k in ("x", "u", "mass", "a", "t")})
        from ..core.clocks import cosmic_time_from_scale_factor
        sim.t = cosmic_time_from_scale_factor(
            sim.a, sim.cosmology.H0_km_s_Mpc, sim.cosmology.Omega_m,
            sim.cosmology.Omega_r, sim.cosmology.Omega_Lambda,
        )
        with pm3d_lock:
            pm3d_playback[sid] = {"playing": False, "speed": 1.0, "a_max": 10.0,
                                  "a_current": float(sim.a), "t_current": float(sim.t)}
        return {"id": sid, "universe": uid, "state": _pm3d_snapshot_json(sim),
                "classification": "SIMULATED", "checkpoint_id": cid}

    @app.post("/api/pm3d/switch/{uid}")
    def pm3d_switch(uid: str, body: dict = Body(default={})):
        """Switch the active PM3D scientific state to another universe.

        Restores the selected universe's actual PM3D state. Creating another
        universe never overwrites the first; each universe owns its own
        cosmology, seed, box size, clock, model configuration and checkpoints.
        """
        if not UNIVERSE_ID.match(uid):
            raise ParamError("invalid universe id")
        urec = multiverse.get(uid)
        # Find or create a PM3D sim bound to this universe.
        sid = None
        for s, u in pm3d_universe.items():
            if u == uid:
                sid = s
                break
        if sid is None:
            # Create a new PM3D sim for this universe from its stored config.
            ms = urec.model_settings or {}
            n_mesh = int(ms.get("n_mesh", 16))
            n_ppax = int(ms.get("n_particles_per_axis", 8))
            seed = int(urec.seed)
            a_start = float(urec.cosmology.get("a_start", 0.2))
            amplitude = float(ms.get("amplitude", 0.02))
            box_size_Mpc = float(urec.cosmology.get("box_size_Mpc", 100.0))
            cosmo = Cosmology(H0_km_s_Mpc=float(urec.cosmology["H0_km_s_Mpc"]),
                              Omega_m=float(urec.cosmology["Omega_m"]),
                              Omega_r=float(urec.cosmology["Omega_r"]),
                              Omega_Lambda=float(urec.cosmology["Omega_Lambda"]))
            x, m = _pm3d_lattice(n_ppax, seed, amplitude)
            sid = "pm3d-" + uid
            if sid in pm3d_holder:
                raise ParamError(f"pm3d simulation {sid!r} already exists")
            sim = ParticleMesh3D(n=n_mesh, x=x, u=np.zeros_like(x), mass=m,
                                 cosmology=cosmo, a=a_start, box_size_Mpc=box_size_Mpc)
            pm3d_holder[sid] = sim
            pm3d_universe[sid] = uid
        # Restore the latest checkpoint if one exists.
        cps = multiverse.list_checkpoints(uid)
        if cps:
            latest = cps[-1]
            result = multiverse.restore_checkpoint(uid, latest.id)
            arrays = result.get("arrays", {})
            sim = pm3d_holder[sid]
            sim.restore({k: (v.item() if v.ndim == 0 else v) for k, v in arrays.items()
                         if k in ("x", "u", "mass", "a", "t")})
            from ..core.clocks import cosmic_time_from_scale_factor
            sim.t = cosmic_time_from_scale_factor(
                sim.a, sim.cosmology.H0_km_s_Mpc, sim.cosmology.Omega_m,
                sim.cosmology.Omega_r, sim.cosmology.Omega_Lambda,
            )
        with pm3d_lock:
            pm3d_playback[sid] = {"playing": False, "speed": 1.0, "a_max": 10.0,
                                  "a_current": float(pm3d_holder[sid].a),
                                  "t_current": float(pm3d_holder[sid].t)}
        return {"id": sid, "universe": uid,
                "state": _pm3d_snapshot_json(pm3d_holder[sid]),
                "classification": "SIMULATED",
                "provenance": urec.provenance,
                "restored_checkpoint": cps[-1].id if cps else None}

    app.mount("/js", StaticFiles(directory=STATIC / "js"), name="js")
    app.mount("/css", StaticFiles(directory=STATIC / "css"), name="css")
    return app


def main():
    p = argparse.ArgumentParser(description="Universe Observatory (3D universe viewer + 1D laboratory)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--save-dir", default="runs/observatory")
    a = p.parse_args()
    import uvicorn
    uvicorn.run(create_app(a.save_dir), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
