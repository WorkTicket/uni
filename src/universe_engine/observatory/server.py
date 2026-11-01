"""FastAPI observatory service: create / step / run / pause / inspect / save / load a real simulation
and stream its snapshots over a WebSocket. One session per app instance.
Start:  universe-observatory --port 8000   then open http://127.0.0.1:8000/   (3D universe viewer; /lab = 1D laboratory viewer)"""
import argparse
import asyncio
import re
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .session import PRESETS, ParamError, Session
from ..multiverse import MultiverseManager, MultiverseError

STATIC = Path(__file__).parent / "static"
MAX_STEPS_PER_REQUEST = 10_000
DEFAULT_COSMOLOGY = {"H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_r": 0.000092, "Omega_Lambda": 0.684908, "spectral_index": 0.965}
SAVE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
UNIVERSE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


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
    multiverse = MultiverseManager(save_dir.parent / "multiverse")

    @app.exception_handler(ParamError)
    async def _param_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(RuntimeError)
    async def _runtime_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(MultiverseError)
    async def _multiverse_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    def ses():
        return holder["session"]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/lab")
    def lab():
        return FileResponse(STATIC / "lab.html")

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
