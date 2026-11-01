# Universe Engine — Verified Starting Foundation

This is a **small runnable baseline**, not a complete Big Bang-to-life simulator. The repository includes a consolidated master blueprint plus a step-by-step implementation manual for expansion into a scientifically defensible multiscale universe engine.

## Run (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
pytest -q
universe validate --config config/reference.yaml
universe run --config config/reference.yaml
```

If packages are already installed but editable installation is unavailable:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m universe_engine.cli run --config config/reference.yaml
```

Outputs: `runs/reference-linear-demo/manifest.json`, `background.npz`, `initial_overdensity.npy`, `final_linear_overdensity.npy`.

### Scientific status

The code integrates a **spatially flat ΛCDM background** and an approximate **scale-independent linear matter growth** equation. A random *illustrative* Gaussian density contrast is filtered using an arbitrary power law and scaled with the growth factor. **It is not a physical CMB/CLASS/CAMB initial condition, not an N-body solver, not an early Big Bang solver, and does not form galaxies, stars or planets.** The `elapsed_Gyr` output counts only elapsed time since its chosen starting scale factor. All scientifically necessary later modules are described in the documents, not implemented here.

Read `docs/IMPLEMENTATION_MANUAL.md` **first**, then `docs/MASTER_BLUEPRINT.md`. Integrate into your existing project only after auditing the existing ZIP and its tests; this starter was produced without access to that ZIP.

## Integrated experimental universe — NEW

This new command runs **actual nonlinear (particle-mesh) gravitational evolution** in a small, periodic, comoving volume and then attaches **reduced-order, phenomenological** galaxies, stellar systems and planets. A **separate, explicitly seeded artificial-evolution experiment** demonstrates heritable traits, mutation, selection and limited resources. It is not descended from the simulated planet chemistry.

```bash
# From repo root, after pip install -e '.[dev]'
python -m pytest -q
universe-evolve --grid 16 --steps 140 --seed 4242 --amplitude 0.09 --output runs/integrated-toy
# Or without editable install:
PYTHONPATH=src python -m universe_engine.evolve_cli --grid 16 --steps 140 --seed 4242 --amplitude 0.09
```

Expected deterministic output on reference settings: `12` local density peak candidates, `12` galaxy *proxies*, `14` sampled stellar systems, `23` sampled planets, and `200` generations in a separate ecology model. Output: `nbody.npz`, `integrated_results.json`. Small numerical differences may occur between platforms.

### Crucial scientific distinction

| Feature | Implementation | NOT implemented |
|---|---|---|
| Dark matter | Periodic CIC particle-mesh, FFT Poisson, numerical advancement in ln(a) | precision ΛCDM transfer functions, calibrated halo masses, bound halo finder, convergence-proven production solver |
| Halo candidates | Thresholded local density peaks | friends-of-friends / spherical-overdensity bound halos |
| Galaxies | Gas and stellar-mass **proxy** assigned to density peaks | hydrodynamic cooling, star formation physics, feedback, galaxies emergent from gas |
| Stars | Statistical mass and lifetime samples | stellar hydrostatic structure, actual nuclear reaction network, stellar tracks |
| Planets | Probabilistic planetary system samples with estimated equilibrium temperature | actual disk accretion, migration, atmospheric modeling or habitability proof |
| Evolution | Seeded artificial individuals, trait heredity, selection, mutations, resources | origin of life, cells, molecular genetics, emergence from the planetary simulation |

This is a **working educational integration prototype**, not full scientific implementation of all outstanding subsystems. The simulation's ecology is independent of its planets; no result claims life spontaneously originated. The reference cosmology demo remains available under `universe run`.

### Next upgrades before research claims

Replace synthetic initial perturbations with power spectra from CLASS/CAMB, check timestep and grid convergence, add physical units and particle masses, implement a genuine halo finder, integrate baryonic hydro/MHD and radiative cooling, connect an established stellar population synthesis model and disk/planet formation model, build chemically explicit reaction–diffusion and protocell dynamics, and validate each stage against published benchmark suites. See docs/IMPLEMENTATION_MANUAL.md.

## Executable physical extensions

```bash
universe-physics --gas-grid 128 --gas-steps 300 --output runs/physical-demo
```

Implements a separate periodic 1D finite-volume Euler gas model with explicit radiative energy removal and a local gas-to-star source; analytic stellar main-sequence track estimates; conservative local annular dust coagulation; and a driven mass-action chemical network. These physical demonstrators are **not yet physically coupled to the 3D cosmological simulation**. Refer to `docs/FULL_BUILD_PLAYBOOK.md` for exact contracts, data requirements, release gates and executable setup commands for advanced work. No claim of emergent galaxies, self-formed planets, or abiogenesis is made from these demonstrators.


## Verification commands (added in this development session)

```bash
python -m pytest -q                 # 146 tests: 24 original + unit + benchmark + sabotage + observatory
universe doctor                     # environment diagnostics (read-only)
universe benchmark hydro            # 106 checks vs exact Riemann solution; writes runs/benchmarks/hydro/<run_id>/
universe benchmark gravity          # 40 checks (Jeans growth, Poisson, conservation); writes runs/benchmarks/gravity/<run_id>/
```
Status and resumption: `docs/CURRENT_MILESTONE.md`, `docs/IMPLEMENTATION_STATUS.json`, `docs/DECISIONS.md`, `docs/SCIENTIFIC_LIMITS.md`.

```bash
universe benchmark nbody            # 37 checks: CIC, nonlinear Zel'dovich, two-fluid Jeans, momentum exchange
pip install -e ".[observatory]"     # (already in the dev extra)
universe-observatory --port 8000    # then open http://127.0.0.1:8000/  (1D laboratory viewer)
python scripts/write_status.py      # regenerate docs/IMPLEMENTATION_STATUS.json from real runs
```

## 3D Universe viewer (observatory)

`universe-observatory --port 8000`, then open http://127.0.0.1:8000/ and press **Begin**. (The original 1D laboratory viewer is at `/lab`.)

A WebGL2 universe simulator: the whole history from the Planck time through inflation, the plasma era, recombination (CMB), the dark ages, cosmic dawn and the growth of the cosmic web, then galaxies. Zoom continuously from supercluster to cluster, galaxy, star, planetary system and planet surface. There are no presets: each universe is generated from a seed (`?seed=123`). Quality: `?quality=low|medium|high`.

Controls: scroll zooms into whatever is under the cursor, drag orbits, double-click focuses, **Dive** autopilots down to a planet, the timeline scrubs cosmic time in both directions (state is a pure function of seed, time and camera). Only what the camera observes is generated and drawn (view-driven procedural detail).

Honest scope: see `docs/SCIENTIFIC_LIMITS.md` ("3D viewer").

## Launching without a terminal (Windows)

- **Double-click `Launch_Universe.bat`**: first run creates a local virtual environment and installs dependencies (needs Python 3.10+ on PATH), then opens the viewer in your browser. Later runs start immediately.
- **Single-file `.exe`**: double-click `build_exe.bat` once (needs Python 3.10+). It produces `dist\UniverseEngine.exe`, which starts the server and opens your browser by itself. The same packaging recipe was built and run on Linux; the Windows build has not been run by the author.
