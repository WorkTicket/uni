"""FastAPI observatory service: create / step / run / pause / inspect / save / load a real simulation
and stream its snapshots over a WebSocket. One session per app instance.
Start:  universe-observatory --port 8000   then open http://127.0.0.1:8000/   (3D universe viewer; /lab = 1D laboratory viewer)"""
import argparse
import asyncio
import re
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


def _pm3d_snapshot_json(sim) -> dict:
    """Convert a PM3D snapshot to JSON-safe form (numpy arrays -> lists)."""
    s = sim.snapshot()
    out = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
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
    # Scientific 3D PM3D cosmology simulation (actual numerical state, not procedural).
    pm3d_holder: Dict[str, ParticleMesh3D] = {}

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
        if not (8 <= n_mesh <= 64):
            raise ParamError("n_mesh must be in [8, 64]")
        if not (2 <= n_ppax <= 16):
            raise ParamError("n_particles_per_axis must be in [2, 16]")
        if not (0 < a_start < a_end):
            raise ParamError("require 0 < a_start < a_end")
        if not (0 < amplitude <= 0.3):
            raise ParamError("amplitude must be in (0, 0.3]")
        cosmo = Cosmology(H0_km_s_Mpc=67.4, Omega_m=0.315, Omega_r=9.2e-5,
                          Omega_Lambda=0.684908)
        x, m = _pm3d_lattice(n_ppax, seed, amplitude)
        sim = ParticleMesh3D(n=n_mesh, x=x, u=np.zeros_like(x), mass=m,
                             cosmology=cosmo, a=a_start)
        # t(a) is initialized in the constructor via the background integral.
        pm3d_holder[sid] = sim
        return {"id": sid, "state": _pm3d_snapshot_json(sim), "classification": "SIMULATED"}

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
