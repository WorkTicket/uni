# UNIVERSE ENGINE — CLAUDE FULL IMPLEMENTATION AND VERIFICATION SPECIFICATION

**Document status:** Execution contract, not a claim of completed research capabilities.  
**Starting release:** The packaged `universe-starter/` Python code.  
**Main user goal:** Start an evolving universe, observe its true calculated state in an interactive 3D UI, and advance a validated multiscale simulation from cosmological conditions toward stars, planets, chemistry, and experimental evolution.  
**Non-negotiable scientific boundary:** No currently validated method deterministically simulates the literal Big Bang singularity, full observable universe atom-by-atom, or guaranteed origin of life. Resolve those using labeled assumptions, derived models and clearly bounded research experiments, never silent invented outcomes.

## A. General engineering contract

**Before each phase:** read only the relevant source/tests; run the last known passing regression; identify versioned interfaces and units; write an executable design note under `docs/decisions/` if a numerical discretization changes; work on a small, testable vertical slice.

**For every implemented solver**, record:
- Governing equations, independent variables, conserved variables, source terms, coordinates (comoving/physical), timestep stability criteria, numerical order, solver choice, positivity strategy and validity domain.
- All inputs/outputs with shape, units, dtype, boundary conditions, interpolation rules, uncertainties, provenance, runtime cost and failure modes.
- Property/unit tests, analytic or published reference benchmarks, convergence vs grid/timestep, conservation budget, reference command, expected result tolerance and actual measured error.
- Config version, run ID, random seed, dependencies, source version/commit, scientific status: documented / implemented / verified / integrated / observationally validated / speculative.

**Never:** convert failed tests into skips without justification; fake validation; introduce guessed stellar/planet properties as simulated fine-scale state; hide mass/energy loss inside clamping; alter scientific variables to fit visual art; run unaudited external code; make network-dependent operations mandatory for local baseline; destroy original repo; claim an asynchronous session will continue without an actual persistent agent or scheduler.

**Architecture:** keep numerical simulation separate from visualization and API. Adopt adapters around existing numerical components. The physics process owns state. Observer APIs read immutable versioned snapshots; control commands are serialized to avoid racing the solver. Use a dimensional-units policy (e.g., SI at boundaries, documented solver-native units internally). Treat cosmic expansion energy bookkeeping carefully: in expanding spacetime there is not generally a globally conserved Newtonian total energy to assert. Fixing every numerical domain to one global timestep is forbidden; allow multi-rate/subcycling with conservative handoffs.

### Target repository topology (adapt instead of gratuitous restructuring)

```text
universe-starter/
  README.md  AGENTS.md  pyproject.toml
  config/reference.yaml  config/benchmarks/*.yaml  config/experiments/*.yaml
  src/universe_engine/
    cli.py physics.py field.py evolution.py physical_models.py
    core/{state,units,clock,registry,interfaces,checkpoint,random,diagnostics}.py
    solvers/{hydro,gravity,cooling,radiation,chemistry}.py
    cosmos/{background,initial_conditions,structure,halos}.py
    astro/{stars,feedback,galaxies,black_holes}.py
    planets/{disks,orbits,interiors,climate}.py
    life/{prebiotic,replication,evolution,ecology}.py
    frontier/{registry,quantum,dark_sector,baryogenesis}.py
    observatory/{api,schemas,snapshots,control,streaming}.py
  tests/{unit,benchmark,integration,regression}/
  observatory-web/ [Next.js frontend, independently installable]
  docs/{CURRENT_MILESTONE.md,IMPLEMENTATION_STATUS.json,DECISIONS.md,SCIENTIFIC_LIMITS.md}
  runs/ [generated; gitignored]
```

Create directories only when adding actual working code. Package already has flat source; preserve its entrypoints.

### Verification tiers

1. **T0 — runs:** CLI starts, outputs parse, no unhandled exception.
2. **T1 — numerical:** benchmark/reproducibility/conservation/positivity pass at specified tolerances.
3. **T2 — integrated:** connected modules exchange state coherently, budgets close, restart reproduces.
4. **T3 — physically validated:** independent published or observed constraints matched within stated uncertainty and validation range.
5. **T4 — research hypothesis:** tested candidate alternative, never reclassified as established because visually plausible.

## B. Initial setup and repo preservation (Phase 0)

1. Inspect source ZIP safely. If `big-bang-universe.zip` is supplied separately, inventory all files, run its tests independently first; only then plan an additive merge. Make backup/branch; do not overwrite working original files.
2. Install Python dependencies in isolated venv, not global; test on supported Windows and optionally Linux.
3. Run `python -m pytest -q`, `universe validate`, `universe run`, `universe-evolve`, `universe-physics` using commands in `START_HERE_CLAUDE.md`. Report actual exit codes and outputs, not expected results as observations.
4. Add CI covering Python 3.11 and 3.12, lint/typecheck when available, import/API smoke tests, fast numerical unit suite and separate expensive benchmarks. Pin dependencies through lock/constraints and document platform limitations.
5. Create `docs/CURRENT_MILESTONE.md` and `docs/IMPLEMENTATION_STATUS.json`. Record code version, scientific fidelity per subsystem, exact next action, failing tests and benchmark data.
6. Add user-facing `universe doctor` diagnostics checking Python version, dependencies, free disk/memory, output write permissions, optional GPU presence and benchmark prerequisites.
**Exit:** baseline runs; unchanged tests pass; documented audit and exact command outputs; no destructive merge.

## C. R3 — Hydrodynamics: finish the solver and benchmark suite

**Equations:** 1D ideal-gas Euler in conservative variables `U=[rho,rho*u,E]`, `E=p/(gamma-1)+0.5*rho*u^2`, `F=[rho*u, rho*u^2+p, u*(E+p)]`. Positive rho and p; derive primitive variables consistently.

**Implementation:** robust first-order Godunov baseline using HLL, then MUSCL finite-volume reconstruction using minmod/MC limiter and SSP-RK2 or MUSCL-Hancock with consistent half-time states. CFL `dt <= C*dx/max(|u|+cs)` with e.g. C<=0.4 for default benchmarks. Ghost-cell periodic/outflow/reflective conditions; reflecting momentum sign. Conservative updates; positivity fall back on first-order interface flux when reconstruction invalid, or reject timestep and retry with smaller dt; log lost conservation budget. No hidden clipping of conserved variables. Phase-control API should accept solver state, dt and diagnostics; write output snapshots.

**Benchmarks (implement as executable pytest cases + CLI `universe benchmark hydro`):**
- Uniform periodic flow: exact preservation to roundoff.
- Sod tube: [0,1], x0=.5, gamma=1.4, left `(rho,u,p)=(1,0,1)`, right `(0.125,0,0.1)`, t=.2, noninterfering outflow; compare against exact Riemann reference, compute L1 density/velocity/pressure at 128/256/512 cells; tolerance justified and stored.
- Acoustic periodic wave: amplitude about 1e-4, compare analytic linear wave before nonlinear steepening; 64/128/256/512; verify observed convergence order consistent with implemented scheme.
- Periodic conservative advection: total mass/momentum/energy drift and wave profiles; use scale-aware absolute tolerance near zero reference quantities.
- Strong-shock/low-pressure robustness; no nonfinite, negative rho/p; emit explicit solver failure if impossible.
- Boundary tests: mass transport periodic, reflected momentum and pressure response, outflow ghost behavior.
- Optional blast wave (Sedov for higher dimension later), Shu–Osher, contact advection.
Outputs: `runs/benchmarks/hydro/<run_id>/{metrics.json,profiles.csv,metadata.json}` and plots if matplotlib installed. Include exact reference equations/source for any analytic solver. **Exit:** all above pass at documented numerical tolerances; results generated from live code; no regressions.

## D. R4 — Gravity–hydro coupling

Build coherent state for collisionless particles, baryonic gas and gravitational potential. Choose one grid/geometry per validated vertical slice. For Newtonian solver: Poisson `∇²Φ=4πG(ρ-ρbar)` under periodic cosmological boundary conditions with appropriate scale factors; for 1D laboratory tests use a separately appropriate self-gravity setup. Add mass deposition (CIC), mesh Poisson FFT with k=0 treatment, acceleration interpolation, gravity momentum source `-rho grad Φ`, energy source `-rho v·grad Φ`, time-centered kick/drift/kick or validated operator split; cosmological comoving factors required.
Tests: uniform field and free fall, hydrostatic equilibrium, Jeans perturbation growth and stable regime, force symmetry, force-grid convergence, particle/gas mass and momentum exchange, coordinate/units checks, combined checkpoint replay. **Exit:** actual hydro and gravity update same state and timestamps without mixing dimensions or units.

## E. R5 — Primordial thermochemistry and radiation interfaces

State vector H, H+, He, He+, He++, e-, internal energy, metals optional; electron charge closure. Load versioned sourced rate tables and EOS/mean molecular weight; collisional ionization, recombination, H/He excitation, bremsstrahlung, Compton, H2 network when available; UV photoionization optional. Rates depend on density, T, composition and radiation, never frozen global constants. Implement stiff integration (`scipy.integrate.solve_ivp(method="BDF")` optionally added dependency), operator splitting compatible with hydro, positivity and per-species element budget. Radiation transport starts with explicit optically thin or background approximation and honest transport interface; don't claim radiation hydrodynamics until photon transport coupled.
Benchmarks: analytic cooling relaxation, chemical equilibrium, primordial species conservation and electron balance, reaction-network stiff limit, energy cooling budget, timestep convergence. **Exit:** thermochemical gas cells evolve consistently with fluid state.

## F. R6 — True cosmological perturbation and nonlinear structure

Start at a supported post-primordial cosmological epoch; document initial epoch. Flat ΛCDM Friedmann baseline; comoving coordinates, redshift, scale factor, appropriate integration. Source linear matter power spectrum from CLASS or CAMB (optional externally installed scientific tool), documented fallback transfer approximation for demos, seed 3D Gaussian modes with Hermitian symmetry and correct normalization, Zel'dovich/2LPT displacement and peculiar velocity, k modes and Nyquist cutoff; no arbitrary density field described as precision cosmology. Periodic particle-mesh gravity + tested kick/drift/kick in `a` or `ln a`. Halo finder friends-of-friends or spherical overdensity; virial or bound criteria; track merger trees. Compute P(k), halo mass function, convergence across grid/particle counts and boxes; reproducible seeds; avoid tiny-box resolution claims. **Exit:** halo metrics and matter clustering numerically benchmarked, not just local peaks.

## G. R7 — Baryonic collapse and star formation

Choose actual 3D hydro integration strategy: extend solver to tested 3D finite-volume (dimensional splitting then unsplit if warranted) **or** interface to a validated mature framework, keeping input/output contracts and scientific provenance. Gravity, gas, cooling, shocks, pressure, Jeans mass/length and free-fall times; prevent artificial fragmentation by proper Jeans resolution (document criterion); sink or subgrid stellar population formation from gas, with local density, temperature, converging flow and resolution requirements. Strict gas-to-star mass transfer and metal bookkeeping. Benchmarks: stable cloud, collapse/Jeans unstable mode, density/pressure positivity, mass conservation, sensitivity to refinement and SF thresholds. **Exit:** stars formed from solved gas conditions rather than being injected by run age.

## H. R8 — Stellar evolution, nuclear yields and feedback

Implement stellar population objects and interpolation from validated tracks or licensed published grids; MESA integration optional and not assumed trivial. Track mass/age/metallicity, radius, Teff, luminosity, evolutionary phase, mass-loss/remnant and ejecta. Input grids must state versions and interpolation limits; disallow invalid extrapolation by default. Nuclear network reduced or track-derived yields must be distinguished. Couple SN, winds, ionizing radiation and metals to nearby fluid cells with conservative deposition and feedback timing. Tests: source-track interpolation, mass and elemental closure, SN yield totals, simple lifetime comparisons; integrated star birth–death–eject cycle. **Exit:** stellar objects age and feed back into gas.

## I. R9 — Galaxy assembly and black-hole astrophysics

Persistent galaxy IDs mapped to bound halos/merger tree; cooled baryonic gas, stellar mass buildup, metal mixing, mergers, satellite substructure, population statistics. Black-hole seeding is an explicit astrophysical subgrid hypothesis; accretion, bounded mass growth, AGN feedback, mergers; do not confuse Newtonian subgrid BH with a numerical relativity solve. Validate aggregate cosmic star-formation history, stellar-mass function, gas fractions, mass–metallicity and galaxy properties against separate calibration/test datasets, with volume/resolution limitations. **Exit:** evolving populations tied to gas/halo state and meaningful statistics.

## J. R10 — Disk physics and planetary formation

For selected stellar systems, initialize disk conditions from recorded stellar/gas histories; evolve gas/dust reservoir, dust coagulation/fragmentation, drift, planetesimal formation prescription, embryo accretion, collisional mergers, migration, disk dispersal. Couple to a tested orbital N-body library such as REBOUND if available; internal simple symplectic 2-body/limited N-body validated independently. Maintain disk/planet/ejected mass and angular momentum budgets; collision and ejection events explicitly registered. Kepler, two-body symplectic drift, resonances, angular momentum, mass closure and timestep tests. **Exit:** planets arise from bounded material reservoirs; if sampled components remain, label them proxies.

## K. R11 — Planet environment and long-term geochemistry

For selected planets simulate interiors, layering, radiogenic/tidal heat, differentiation, outgassing, surface boundary conditions, atmospheres, spectral absorption/radiation balance (tested gray model initially), escape, liquid water phase, ocean reservoir, weathering/carbon cycling, stellar flux history and climate stability. Couple volatile and mass/energy budgets and atmospheric chemistry. Validate Earth/Mars/Venus benchmark properties within model applicability; sensitivity to opacity, insolation and coarse resolution. **Exit:** planets have calculated time-dependent environment histories, not arbitrary habitability badges.

## L. R12 — Non-equilibrium prebiotic chemistry

Build a chemically explicit reactor driven by environmental temperature, solvents, species and energy flow from selected planetary simulations. Use stoichiometric matrix, atom/charge conservation, reversible rate laws and thermodynamic constraints, mass-action ODE, diffusion/reaction in selected spatial domain, external feed reservoirs, polymerization/hydrolysis, catalysis and compartments as explicit alternate models. Research modes: RNA world, metabolism first, hydrothermal gradients, lipid/protocell models. Test against known chemical reference rates, equilibrium, atom conservation, stiffness convergence and compartment budgets. Distinguish initialized replicators from truly emergent chemistry. **Exit:** reproducible chemistry experiments; no guaranteed abiogenesis.

## M. R13 — Evolution, ecosystems and behavioral experiments

After an explicitly justified replication model exists or user selects artificial-life initialization, evolve genomes/traits, mutation, selection, reproduction, finite resources, population death, drift, species differentiation, ecological networks, nutrient recycling, symbiosis, predation, migration. Couple species resource intake to simulated environmental reservoirs; ensure ecosystem mass/energy budgets. Validate selection under analytic simplified models, inheritance statistics, drift, extinction, food-web budgets. Advanced modules (multicellularity/neurons/learning) are standalone research components with bounded claims. **Exit:** heritable evolving ecology reproducible from declared assumptions.

## N. R14 — Frontier physics experiment registry

Keep physically established reference baseline. Version alternative theories with: equations, initial/boundary conditions, params/priors, compatibility constraints, observational tests, computational cost, predicted differences, provenance and status. Candidates include warm/self-interacting dark matter, axion effective models, dynamic dark energy, baryogenesis/leptogenesis, quantum measurement/CHSH/entanglement, objective collapse, many-worlds interpretation, Bohmian mechanics, black-hole GR/accretion and semiclassical Hawking approximations, speculative QG-inspired cases. Many-worlds interpretations cannot be treated as distinct observed predictions merely through visualization. Small quantum subsystems: density matrices, unitary/Lindblad evolution, multiple measurement bases, CHSH S (not just Bell-pair correlation). Model comparisons require held-out observations and uncertainty. **Exit:** valid, repeatable hypothesis comparisons; no claims of unsolved theories established.

## O. R15 — Operations, reproducibility, safety and experiment management

Version configs/schemas/scientific datasets, seeded RNG streams per subsystem, snapshots/checkpoint format with checksums and solver version, restart consistency, resource budgets, step-rejection and recoverable failures, progress observability, CPU profiling and optional GPU acceleration after correctness. Never silently truncate simulation physics to fit memory. Automated CI and slow-science nightly suite. Build run manifest tying source revision, parameters, physical models, seed, experiment branch, output hashes and benchmark results. **Exit:** restart from checkpoint, parameter fork and independently reproducible results at stated tolerances.

## P. R16 — Interactive 3D observatory (begin in parallel once R3/R4 streaming is stable)

**Goal:** user launches simulation via GUI and visually watches actual numerical state; no dependence on manually reading JSON. Preferred architecture: FastAPI local app with typed REST controls + WebSocket telemetry, immutable snapshot queue, Next.js + React Three Fiber/Three.js separate UI. The simulation solver owns state, UI can't change its equations except via validated configurations and explicit queued controls.

Backend endpoints, adapt to existing API style:
- `GET /api/health` (version, solver running, ready)
- `GET /api/models` (available validated solvers, fidelity and constraints)
- `POST /api/runs` (config ID, seed, resource budget; returns run_id)
- `POST /api/runs/{id}/start`, `/pause`, `/resume`, `/stop`, `/checkpoint`
- `GET /api/runs/{id}/status` (cosmic time, a, redshift, uptime, step, errors)
- `GET /api/runs/{id}/snapshots` (list time, bounds, fidelity, schema)
- `GET /api/runs/{id}/snapshots/{snapshot_id}` (manifest/binary refs)
- `GET /api/runs/{id}/objects/{object_id}` (actual catalog properties only)
- `WS /api/runs/{id}/stream` (live decimated field/particle snapshots with backpressure)
- `POST /api/runs/{id}/fork` (new assumptions, new run ID, immutable parent)

Security: default bind `127.0.0.1`, CORS allowlist, API validation, no arbitrary file paths, safe output root, max particle count and memory limit, safe termination and restart; never expose unauthenticated control server to public internet by default.

Observer UI layout: full-size WebGL 3D view with particle positions, density slices and gas overlays; camera orbit/pan/fly, axis, scale bar and units; left panel run/start/pause/resume/speed/computational budget; bottom timeline with saved snapshots and cosmic time/redshift; right inspector selected halo/galaxy/star/planet with properties, provenance and `not simulated` where missing; toggle layers, filter by species, color map legend, GPU draw-call budget and LOD. Dynamic downsampling preserves statistical integrity and labels its strategy. Playback speed affects visualization cadence, never physics timestep validity. Save local snapshots and support time scrub of recorded states. Change resolution by spawning a separate local region simulation; do not fabricate atom-level resolution from macro snapshots.

Implementation order: (1) FastAPI stream real `nbody.npz` snapshots without simulation control, (2) live existing gravity loop, (3) start/pause/resume/stop/checkpoint, (4) R3/R4 hydro fields, (5) halo object selection, (6) galaxy/star/planet catalogs with proxy badges, (7) local refinement experiments, (8) performance/reliability.

Frontend dependencies installed only inside `observatory-web/`, not mixed with Python package; add reproducible `package.json` and lockfile when implemented. Commands: `uvicorn universe_engine.observatory.api:app --host 127.0.0.1 --port 8000` and `npm run dev` in frontend, once endpoints are actually built. Document Windows scripts in README. Test API contract, WebSocket ordering, pause/resume, empty arrays, corrupt snapshots, oversized payload rejection, object provenance, browser connect/reconnect and long run. **Exit:** one click starts actual physics and displays real changing state, with saved timeline and functioning inspector.

## Q. R17 — End-to-end release qualification

Run scenarios: background/structure through halo statistics; gas gravity/cooling through star formation; stars into feedback/metals; disk through orbital dynamics; environment to chemistry; explicitly initialized evolution; alternate dark-sector branch; start/pause/checkpoint/restart/fork; GUI live visualization and science metadata. Each scenario has reference inputs, expected invariants, numerical tolerances, limited scale and outcome logs. Do not require all processes to occur in same cell/time resolution; use linked regional simulations and record boundaries. On any unsupported domain, result reports `not simulated` or `experimental` rather than faking continuity. Create release manifest, installer/setup script, end-user quick start, CLI and GUI smoke tests. **Exit:** working tested research simulator with honest coverage map. No "exact full universe" claim.

## R. Dependency and execution order

0 audit → R3 hydro → R4 gravity/hydro → R5 thermochemistry → R6 cosmological IC/halos → R7 gas collapse/star birth → R8 stellar physics → R9 galaxies/BHs → R10 disks/orbits → R11 environments → R12 prebiotic chemistry → R13 artificial biology → R14 frontier theories → R15 operations → R17 release.

**In parallel** after completing first stable snapshot schema, R16 observatory: first get actual N-body scene on screen, then add newer fields as modules arrive. Avoid holding off all visualization until R15. Support independent small component benchmark development while an external dataset is unavailable. If a critical dependency cannot be obtained, record precisely what is blocked and continue other independent work; don't prematurely declare entire phase complete.

## S. Exact session-level work protocol

At session start:
1. `git status`; inspect `docs/CURRENT_MILESTONE.md`/`docs/IMPLEMENTATION_STATUS.json` and last verified commit. If missing, create after running baseline.
2. Read relevant phase section, determine smallest unverified acceptance criterion. State the concrete immediate edit and test in one or two sentences; then write the code.
3. Run focused test, inspect failures, fix; run entire fast suite before completion. Generate benchmark files from real execution.
4. Update status files with date, SHA, exact commands, pass/fail/skip counts, physics limits and next smallest action; checkpoint changes or commit if authorized.
5. If time/compute limit is reached, stop at a recoverable checkpoint and say which task must resume. No request to "choose next phase".

At phase completion: publish concise table of milestone, changed files, equations/solver, tests, measured errors, scientific status and next milestone. A phase with source code but no validation must be marked **implemented/unverified**, not complete.

### Minimum JSON status contract

```json
{
  "project": "big-bang-universe",
  "repo_revision": null,
  "current_phase": "R3",
  "current_task": "Implement and benchmark finite-volume Euler solver",
  "phase_status": "in_progress",
  "test_command": "python -m pytest -q",
  "test_counts": {"passed": null, "failed": null, "skipped": null},
  "last_verified_benchmarks": [],
  "scientific_limitations": [],
  "next_action": "Inspect existing 1D hydro and implement Sod reference benchmark",
  "next_phase": "R4"
}
```

The fields with `null` must be replaced with measured values after a run, not invented.

## T. Acceptance summary / What "done" means

- User can launch a valid physical simulation, not just a screen animation.
- Physics solvers form a coherent documented multiscale pipeline with explicit model boundaries.
- Dark matter clustering, gas flow, star formation, stellar enrichment, disk/orbit dynamics, local chemistry and ecology all have **working bounded implementations** and scientific integrity checks; where no predictive first-principles model exists, present transparent experiment mode, not false certainty.
- Observatory renders real numerical fields/objects, with provenance and physical units, start/pause/checkpoint/restore and local zoom experiments.
- Release tests pass; numerical error documented; no physically unsupported outcomes silently generated.
- Ultimate fundamental-physics and abiogenesis uncertainties remain explicit rather than invented away.
