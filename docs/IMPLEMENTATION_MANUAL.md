# UNIVERSE ENGINE — FULL IMPLEMENTATION PLAYBOOK

**Edition:** 1.0 (2026-10-30)  
**Use with:** `MASTER_BLUEPRINT.md` (full domain inventory), repository source and tests.  
**Important:** This is an executable development specification and stage-by-stage plan, **not** a claim that all described physics exists or has been solved. A literal Big Bang singularity, actual abiogenesis and subjective consciousness remain outside validated first-principles computational science.

## 0. Outcome, constraints, and definition of success

### Target
Given a configuration and random seed, run a periodic representative universe volume from a scientifically defensible starting epoch. Advance an expanding background and self-gravitating matter, couple gas and radiation, form stars and chemical elements with state-dependent prescriptions, generate galaxies and selected planetary systems, and expose reproducible zoom simulations for chemistry, biology and quantum experiments. Compare observables against data. Never manually fabricate structures to satisfy a scene. Keep frontier alternatives separate from the reference baseline.

### Explicit boundaries
1. Do **not** promise a simulation of every atom in the observable universe. It is a finite periodic volume plus regional high-resolution models.
2. Do **not** start at mathematical time `t=0` under standard physics. The first executable stage here begins at `a=0.02` (`z=49`), and is **not** a reconstruction of primordial nucleosynthesis. Other early-universe modules start with independently supported equations and hand off validated state.
3. The current included runnable `0.1.0` is **FLRW expansion + a synthetic Gaussian field scaled using approximate linear growth**. There is no N-body collapse, radiation transfer, stellar physics, or biological emergence yet.
4. Multiscale feedback must conserve physically applicable quantities across transitions. In an expanding spacetime, do not enforce a naive globally conserved total Newtonian energy for the whole universe.
5. The simulation output must never be labeled observationally validated until compared against independent reference data and convergence tests.

### End-to-end release definition
A release is 'Reference Cosmic Evolution Ready' only when: (a) a pinned, validated initial power spectrum is loaded; (b) periodic gravity integrates collisionless matter with converged force and timestep accuracy; (c) a gas and cooling solver produces coupled collapse; (d) a documented subgrid star-formation model and feedback deposit matter, momentum, energy and metals consistently; (e) the run completes autonomously with checkpoint/restart; (f) measured statistics agree with selected observational/research baselines within explicitly specified tolerance and resolution limits; and (g) all model assumptions are inspectable. Planet/chemistry/biology are subsequent scoped research milestones, not prerequisites for this release.

## 1. Environment: Windows, local development, reproducibility

**Suggested development environment:** Windows 11 + WSL2 (Ubuntu LTS), Python for orchestration and prototyping, NumPy/SciPy for early solvers, compiled CPU code and optional CUDA for heavy kernels as needed, HDF5 or Zarr for scientific data, pytest for tests. No dedicated GPU is needed for the provided starter.

**Windows PowerShell, one-time:**

```powershell
wsl --install -d Ubuntu
```

After reboot if requested, open Ubuntu. Use your home directory for code rather than `/mnt/c` to avoid slow cross-filesystem I/O. Install the Python venv support and development tools:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git cmake pkg-config
mkdir -p ~/research
cd ~/research
# Extract universe-starter.zip here and enter universe-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
universe validate --config config/reference.yaml
universe run --config config/reference.yaml
cat runs/reference-linear-demo/manifest.json
```

This install needs internet access for dependency downloads unless a compatible Python environment already contains them. In a preconfigured Python environment:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m universe_engine.cli validate --config config/reference.yaml
PYTHONPATH=src python -m universe_engine.cli run --config config/reference.yaml
```

**Expected outputs:** `background.npz` containing arrays `a`, `z`, `H`, `D`, `elapsed_Gyr`; `initial_overdensity.npy`; `final_linear_overdensity.npy`; and `manifest.json`. The elapsed time is from `z=49`, not the absolute age of the universe. `initial_overdensity.npy` is synthetic, not normalized by a Boltzmann code.

**Windows-only Python alternative:** use PowerShell with `py -3 -m venv .venv`, `.venv\Scripts\Activate.ps1`, `python -m pip install -e '.[dev]'`; the same `pytest`/`universe` commands then apply.

**Repository rule:** pin exact Python/dependency versions and platform/compiler details in the CI and release lockfiles before declaring reproducible scientific results. The supplied `pyproject.toml` uses compatibility ranges for developer accessibility, not a research-grade frozen environment.

## 2. Existing 43-file archive integration (mandatory before merge)

The reported `big-bang-universe.zip` was **not provided** with these instructions. Therefore do not replace or assume its source, and do not claim its 22 tests pass locally. Extract your archive into a separate directory; create a Git branch; retain its historical tests and files; run existing tests before and after every integration.

```bash
cd ~/research
mkdir -p big-bang-universe-audit
unzip -q big-bang-universe.zip -d big-bang-universe-audit
cd big-bang-universe-audit
git init
git add .
git commit -m 'Baseline before integration'
find . -maxdepth 4 -type f | sort > baseline-files.txt
```

Create `docs/AUDIT.md` with one line per existing module: path, role, implementation vs description, equations, validity, input/output units, current tests, missing datasets, risks, target phase. Run the existing test command documented by that repository; log the exact command and results. If its configuration or language differs, adapt the starter into that repository **without** wholesale replacing its structure. Reconcile names and shared state contracts first.

## 3. Canonical folder structure for the long-term repository

```text
big-bang-universe/
  README.md
  AGENTS.md                           agent constraints, mandatory checks, scoped reads
  pyproject.toml / dependency locks
  config/
    reference.yaml                   conservative ΛCDM starting preset
    experiment/                     explicit speculative variants
    hardware/                       laptop, workstation, cluster budgets
  docs/
    MASTER_BLUEPRINT.md
    IMPLEMENTATION_MANUAL.md
    ARCHITECTURE.md
    PHYSICS_CONTRACTS.md
    DATA_MANIFEST.md
    VALIDATION_MATRIX.md
    OBSERVATIONAL_BENCHMARKS.md
    ASSUMPTIONS_AND_UNKNOWNS.md
    CURRENT_TASK.md
    DECISIONS.md
  src/universe_engine/
    cli.py                            run/validate/inspect/benchmark/resume/branch commands
    kernel/                          clocks, units, state, scheduler, rng, checkpoint
    numerics/                        ODE/PDE/gravity/flux/AMR/FFT/linear algebra
    cosmology/                       FLRW, perturbations, Boltzmann bridge, growth
    initial_conditions/              calibrated transfer and power spectra, realizations
    particles/                       collisionless dark matter and particle properties
    fluids/                          hydro, MHD, cooling, conduction, turbulence
    radiation/                       opacity, emission, absorption, transport
    nuclear/                         nuclides, rates, reactions, energy yields
    atomic/                          ionization, spectra, chemistry inputs
    galaxies/                        halo finder, mergers, stars, feedback
    stellar/                         evolution tracks, remnants, yields
    black_holes/                     accretion, feedback, geodesics, frontier
    planetary/                       disk, N-body, atmosphere, surface, climate
    chemistry/                       kinetics, constraints, reaction networks
    biology/                         conditional experimental replication/evolution
    frontier/                        alternative theories and compatibility rules
    observation/                     synthetic catalogs, lightcones, instruments
    validation/                      tests, baselines, statistics, error budgets
    operations/                      run registry, telemetry, scheduling, storage
  tests/
    unit/                             small exact checks
    integration/                      coupled conservation and restart
    physics/                          analytical solutions
    convergence/                      resolution and timestep scans
    benchmark/                        calibrated research benchmarks
  data/
    manifests/                        pinned URLs, releases, checksums, licenses
    raw/                              immutable acquired source data
    processed/                        validated versioned tables
  runs/                                ignored generated outputs
  scripts/                             download, verify, convert, benchmark
  viewer/                              optional read-only interactive UI
```

Use packages/modules that exist and are tested; create directories when entering a phase. Empty placeholder packages must not count as implementation.

## 4. First principles of data and state

### UniverseRun metadata
Required: `run_id`, UTC creation timestamp, code Git commit, dirty-tree flag, config file checksum, data-manifest checksum, random seed, physics preset, scientific status, solver versions, machine and precision metadata, box size and boundary conditions, cosmological parameters, target time/redshift range, resolved scale limits, output cadence, error tolerances, and prior checkpoint parent if resumed/branched.

### Coordinates and units
Use **SI** at system boundaries or a documented canonical cosmological internal unit system. Store unit metadata with every externally serialized array. Explicitly distinguish proper vs comoving positions; peculiar vs Hubble-flow velocity; proper vs conformal time; physical vs comoving density; frequency/wavelength in vacuum vs air; particle vs volume density; gravitational vs rest mass. Never transform variables by naming alone.

The FLRW baseline is `H(a)^2 = H0^2[Ωr a^-4 + Ωm a^-3 + Ωk a^-2 + ΩΛ]` for Λ dark energy, or replace ΩΛ with an explicitly integrated `w(a)` component. In the provided starter Ωk=0; `H(1)=H0`. Radiation is a background approximation, not coupled Boltzmann evolution. For linear growth, the basic ODE used by the starter neglects scale-dependent radiation/baryon effects; do not run it backwards into the radiation-dominated regime and call the result exact.

### Domain state schemas
Cosmological particles: `id`, `x_comoving[3]`, `peculiar_momentum_or_velocity[3]` (declare exactly which), `mass`, `species`, `softening`, `rng_stream`. Mesh fields: cell geometry, physical density, velocity/momentum, internal energy, temperature where derived, species fractions, ionization state, magnetic field, radiation groups, conserved species and validity flags. Star particles: birth time, mass, initial metallicity/composition, IMF/track ID, remnant state, feedback reservoirs. Galaxy/halo catalogs are **derived analysis objects**, not conservation-breaking independent gravitating doubles of their constituent particles.

### Physics module contract
For every solver, implement a declared schema:

```yaml
id: unique_module_name
version: semantic_version
scientific_status: established_or_approximate_or_speculative
reference: DOI_or_dataset_release
input_fields: [field_name_and_units]
output_fields: [field_name_and_units]
coordinate_frame: comoving_or_proper_or_local_relativistic
validity_domain: {temperature: ..., density: ..., redshift: ..., resolution: ...}
dependencies: [module_ids]
equations: explicit_reference
integrator: explicit_numerical_method
conserved_quantities: [mass, momentum, charge, ...]
constraints: [positivity, divergence_free, ...]
error_estimator: numeric_and_model_uncertainty
activation: predicate
fallback: refine_or_switch_or_raise
required_tests: [named_test_ids]
```

Runtime API: `prepare`, `propose_step`, `advance`, `validate`, `observe`, `checkpoint`, `restore`. Avoid mutation outside declared ownership; pass coupling fluxes explicitly; log constraint residuals. A model invalid beyond its temperature/density/resolution must fail closed or invoke a validated fallback. Never silently extrapolate tabulated reaction rates outside their domain.

### Randomness
Use a master seed plus stable, **independent** streams derived from immutable region/module identifiers, not `global_rng` whose execution order changes with parallelism. Record per-domain RNG state in checkpoints. A deterministic mode should reproduce bitwise on a pinned machine/toolchain where supported, while scientific reproducibility across platforms can use numerical tolerances.

## 5. Numerical infrastructure: build these before advanced features

1. Implement named dimensional units and strict conversions; reject unit mismatches.
2. Define a single monotonic universe clock (`a` or conformal time), with invertible proper-time conversions.
3. Implement ODE integrator with analytic convergence fixtures, adaptive error controller and bounded steps.
4. Add a collisionless particle gravity solver using periodic Poisson solution (FFT particle-mesh is a sensible initial method). Cosmological peculiar equations must be derived and tested in chosen variables; **never** use an unmodified isolated Newtonian N-body solver in comoving coordinates.
5. Implement particle-to-mesh mass assignment and mesh-to-particle force interpolation consistently (e.g., CIC), with periodic wrap, mean density subtraction, Poisson zero-mode policy, density normalization and force-softening control.
6. Use kick-drift-kick with expansion-dependent factors, and test momentum bookkeeping / growing-mode behavior with analytic linear perturbations.
7. Add adaptive timesteps and force-resolution convergence; do not infer precision from one high-resolution picture.
8. Add conservative gas solver with finite-volume Godunov fluxes and a verified Riemann problem test; enforce positivity and CFL bound.
9. Add operator-split cooling/chemistry, with stiff integrators and error controls, after verifying gas conservation in isolation.
10. Introduce AMR and local zoom only after base global gravity/gas correctness; log mass/momentum fluxes across refinement edges.
11. Implement checkpoint and restart before long experiments; serialization must include RNG, solver substate, integrator stage and data hashes.
12. Add physical and numerical monitoring: conservation residuals, time-step statistics, NaN detection, maximum overdensity, energy equation residuals, memory/VRAM use and wall-clock speed.

## 6. Build phases with strict dependencies and exit criteria

### Phase 0 — Audit, reproducibility and baseline
**Implement:** inspect existing archive; preserve every test; add Git, CI, environment lock, data manifest, assumptions registry, failure codes and config validation.  
**Pass:** clean install; all pre-existing tests pass; audit covers every extant file; manifests list data sources/versions; identical seed produces matching hashes on the reference platform.  
**Do not:** write new black-hole or life solvers yet.

### Phase 1 — FLRW + seed field (included starter)
**Implement:** background `H(a)`, cosmological clocks, approximate linear growth, reproducible Gaussian synthetic density contrast, save results.  
**Pass:** H(a=1)=configured H0; H positive; H decreases over this model's history; growing-mode amplitude increases; timestep refinement changes results negligibly; run artifacts hash against config; no unsupported model is advertised.  
**Known shortcoming:** synthetic field lacks a physical matter transfer function, radiation-dependent scale growth and nonlinear dynamics.  
**Deliverable:** `universe run ...` produces diagnostic arrays and manifest in seconds on a typical desktop.

### Phase 2 — Validated cosmological initial conditions
**Implement:** ingest outputs from a trusted Boltzmann code (CLASS or CAMB) or consistent published reference transfer-function data; match gauge, normalization, redshift, power spectrum conventions, baryon/dark-matter species and velocity assignments. Generate Fourier phases deterministically with proper Hermitian symmetry. Use a proper 2LPT implementation for particle starts when required; start redshift chosen for the method and box resolution.  
**Pass:** measured sampled power spectrum agrees with input within finite-volume/cosmic-variance tolerance; displacement distributions satisfy periodicity; particle mass accounts for configured Ω components; test different seeds; transfer data checksum and cosmology match the run.  
**No shortcut:** do not relabel the starter's arbitrary filtered white noise as a Planck-normalized primordial field.

### Phase 3 — Collisionless nonlinear structure
**Implement:** periodic comoving particle-mesh gravity, self-consistent background, cosmological time integration, proper kick/drift factors, halo finder. Add optional tree/PM hybrid only if convergence calls for it.  
**Pass:** force tests with known particle configurations; zero force for homogeneous density; expanding linear mode evolution; mass conservation; stable periodic crossing; reproducible checkpoint/resume; halo mass function / nonlinear power comparisons with a trusted benchmark at multiple resolutions.  
**Deliverable:** first real dark matter cosmic web. **Do not** declare galaxy formation yet.

### Phase 4 — Gas physics and first stars
**Implement:** compressible gas dynamics, primordial H/He cooling and ionization, radiation backgrounds, hydrodynamic gravity coupling, star formation as a transparent calibrated subgrid model where Jeans scales cannot be resolved.  
**Pass:** Sod shock tube, Sedov blast, Kelvin–Helmholtz test where appropriate, Jeans instability, radiative cooling equilibrium, conservative gas-to-star transfer and demonstrated timestep/mesh convergence.  
**Deliverable:** collapse to gas structures and stellar population particles; these are subgrid stellar populations, not atom-by-atom emergent individual stars.

### Phase 5 — Stellar evolution, enrichment, feedback, galaxies
**Implement:** pinned mass/metallicity-dependent stellar evolution tracks (e.g., evaluated MESA-derived grids), IMF and stellar population synthesis, yields, delayed mass return, supernova feedback, gas metal advection, black hole accretion/AGN feedback with documented calibration.  
**Pass:** total mass including ejecta/remnants; SN energy and event-rate accounting; metallicity limits; stellar lifetimes versus model grid; calibrated galaxy stellar-mass and metallicity statistics, with held-out observations and uncertainty.  
**Deliverable:** galaxies, star-formation histories, chemical enrichment and black-hole populations under declared subgrid assumptions.

### Phase 6 — Radiation, reionization, synthetic observables
**Implement:** absorption/emission/scattering with opacity/cooling tables, radiation transport method valid for chosen optical depths, photon groups, cosmic reionization model, observational light cones with light-travel time, redshift-space effects, synthetic spectra/images.  
**Pass:** static-source propagation benchmark, photon/energy accounting with boundary terms, Strömgren sphere, comparison against observed luminosity functions and reionization constraints.  
**Deliverable:** scientifically interpretable telescope-like outputs, not merely artistic renderings.

### Phase 7 — Compact objects and strong-field zooms
**Implement:** validated Schwarzschild/Kerr geodesic and lensing tools; compact binary approximations where appropriate; localized relativistic simulations if computing resources and validation allow. Model Hawking radiation only within meaningful scales; separate unresolved black-hole interiors and information hypotheses.  
**Pass:** photon orbits, weak-field limit, gravitational redshift, ray-tracing and binary tests against numerical/analytical references.  
**Deliverable:** black-hole observations and experimental strong-gravity research, not unverified quantum-gravity claims.

### Phase 8 — Planets and geophysical environments
**Implement:** select a parent star/galaxy with known environment; disk accretion, planetesimal populations, orbital N-body, impact evolution, interior/atmospheric/climate models; use statistical formation models if full direct simulation is unaffordable.  
**Pass:** Kepler orbits and orbital energy tests in supported regime; disk mass accounting; atmosphere energy balance and known solar-system benchmark comparisons; consistent parent-star irradiation/metallicity/tidal boundary conditions.  
**Deliverable:** modeled planetary populations and selected detailed planets. 'Habitable' is a hypothesis about environment, not proof of life.

### Phase 9 — Chemistry and origins-of-life experiments
**Implement:** bounded reaction networks with accurate elemental/charge balance, non-equilibrium kinetics, energy flow, transport, mineral surfaces, polymerization, membranes, protocells and competing origin hypotheses. Run controlled ensembles; include chemical extinction and failure as valid outcomes.  
**Pass:** mass/charge atom balance, nonnegative concentrations, equilibrium/detailed-balance tests where applicable, reproducible statistics and parameter sensitivity.  
**Deliverable:** **experimental** prebiotic model behavior; never label abiogenesis as solved.

### Phase 10 — Evolution and ecology
**Implement:** conditional heredity and metabolism models in bounded populations, genotype-to-phenotype mapping, mutation, selection, ecological constraints, environmental feedback. More detailed cellular/neural models only where needed for a targeted experiment.  
**Pass:** reproduction and mutation statistics, resource accounting, selection benchmarks, ecological stability tests, parameter uncertainty and repeatability across independent seeds.  
**Deliverable:** evolving computational organisms under stated assumptions. Intelligence or subjective experience are not guaranteed.

### Phase 11 — Frontier theory laboratory
**Implement:** model plugins for warm/self-interacting/axion-like dark matter, evolving dark energy, baryogenesis hypotheses, objective-collapse models, early-universe alternatives and black-hole/quantum-gravity models when mathematically computable. Add compatibility matrix; branch runs and generate likelihood comparisons.  
**Pass:** standard/reference limit returns reference behavior; model-specific predictions are reproducible; incompatible theories cannot be combined by accident; each alternative declares its evidence and possible falsification tests. Everett/many-worlds vs standard operational quantum mechanics is generally an interpretation choice, not a different observable cosmology.

### Phase 12 — Viewer and long-running operations
**Implement:** detached render viewer, query/index system, saved lightcones, selectable halo/galaxy/planet state, branch/restart controls, scientific labels, data integrity and retention limits.  
**Pass:** simulation headless remains unchanged by viewer presence; UI does not 'collapse' physical state by inspection; tests compare snapshots across resume/branch operations; corrupted checkpoint cannot silently resume; long-run memory stays bounded and alarms fire on solver failure.

### Order note
Some Phase 11 interfaces should be designed early, but detailed speculative models should not displace validation of Phases 0–6. Scientific tests, storage, observability, and checkpointing begin in Phase 0 and are strengthened continuously—never postponed to the end.

## 7. External data: acquisition and admissibility

Use published resources as data, not hidden scripts that create desired outcomes. Recommended primary sources: NIST/CODATA constants; Particle Data Group; IAEA evaluated nuclear records; NIST Atomic Spectra Database; CLASS/CAMB-calculated cosmological transfer functions; NASA LAMBDA/Planck data; DESI cosmological measurements; IllustrisTNG simulation products for comparison; MESA or independently evaluated stellar track datasets; appropriate opacity and chemical rate libraries. Check current release/version before ingesting and review licensing terms. Do not automatically ingest files of unknown provenance.

**Each source must have** `dataset_id`, version, URL/DOI, access date, license, SHA-256, citation, coordinate/unit conventions, valid parameter domain, interpolation method, uncertainties, permitted uses, and preprocessing source commit. Store raw immutable; build processed tables with schema validation. Implement monotone/interpolant choice deliberately; table extrapolation must be disabled by default. CI uses a tiny checked-in fixture, not multi-GB downloads.

**Order:** constants → cosmological transfer → validation benchmarks → nuclear/atomic/opacity → stellar tracks/yields → planet/chemistry references. Data import must not block the first working gravity milestone if its data is unnecessary.

## 8. Exact scientific tests and release evidence

| Suite | Minimal fixture | What it proves | What it does NOT prove |
|---|---|---|---|
| Config | reject negative Ω, inconsistent flatness, nonmonotone a | input validation | cosmological accuracy |
| FLRW | H(1), radiation/matter dependence, time integral | background arithmetic | full early universe |
| Linear | known growth case and timestep convergence | basic integrator/growth scheme | nonlinear structure |
| Random field | identical seed, mean zero, finite spectrum | reproducibility | correct primordial spectrum |
| Gravity PM | uniform field, single mode, periodic symmetry | force implementation | cosmological accuracy |
| Gravity N-body | convergence and reference power spectra | solver behavior | galaxy baryon physics |
| Hydro | Riemann shock, conserved fluxes, positivity | fluid solver | realism of star formation |
| Chemistry | reaction balance, stiff test, charge conservation | chemistry integrator | origin of life |
| Quantum | CHSH multi-basis test, no-signaling, Born frequencies | local quantum simulator | many-worlds truth |
| Stellar | lifetime, luminosity, yield interpolation | table handling | direct fusion of every star |
| Galaxy | held-out statistics with error bars | population plausibility | unique model truth |
| Planet | orbit and thermal energy benchmarks | local solver accuracy | inhabited planet |
| Resume | uninterrupted vs restart, hashes/metric tolerance | state integrity | physical fidelity |

Use separate test labels: `specified`, `implemented`, `numerically_verified`, `observation_validated`, `speculative`, `unsupported`. A green code test cannot automatically upgrade a model to observationally validated.

**CI:** lint and static-type checks if configured; unit tests on each commit; integration, conservation and small convergence tests on pull requests; expensive stochastic/scientific comparisons in scheduled or release workflows. CI must pin external datasets. In a validation failure, record offending parameter, solver, coordinate frame, timestep, max error and checkpoint; stop instead of painting a fake result.

## 9. Run lifecycle, snapshots, failure recovery

1. `validate`: parse config; reject unknown/invalid physical fields; check physics compatibility, data versions and resource budget.
2. `prepare`: prefetch/check dataset hashes; allocate arrays; assign deterministic RNG streams; initialize fields and document start epoch.
3. `run`: iterate time in the declared independent variable. Every active module proposes its admissible timestep. Scheduler chooses safe minimum; prepare boundary fluxes; advance; validate; commit successful step or roll back and reduce timestep.
4. `observe`: capture source fields at checkpoint epoch; never alter world state when generating observer output.
5. `checkpoint`: atomic write temporary file → validate checksum → rename; include physics, integrator, mesh hierarchy, RNG, schema/model/version metadata.
6. `resume`: compare code/data compatibility, restore all state, verify checksum and monotone clock, then continue from exact stage.
7. `branch`: freeze parent manifest, assign new run ID, record changed theories/parameters and when the branch began. For incompatible initial histories, regenerate initial conditions instead of pretending the old state remains valid.
8. `report`: produce scientific validation matrix, statistics and uncertainty alongside timings/memory, plus conspicuous missing-physics notices.

**Failure modes:** negative density/internal energy → timestep rollback or bounded positivity-preserving correction with error report; NaN/Inf → hard stop and recovery checkpoint; extrapolated lookup table → explicit error/refinement/fallback; CFL violation → reduce dt; bad mesh flux → stop and inspect; insufficient VRAM → reduce domain/resolution or CPU fallback if explicitly validated; failed checkpoint hash → refuse resume; observational disagreement → record model failure, do NOT force desired features into existence.

## 10. Resource and performance engineering

For a desktop workstation with a mid-range GPU and about 12 GB VRAM: begin 32³–128³ density/particle meshes and lightweight CPU prototypes; benchmark actual RAM/VRAM before increasing. N-body particle-mesh memory depends on density, force arrays, FFT workspaces, particle attributes, buffers and precision, not merely `N³ * particle_size`. Start with 64³ to test methodology, not to claim resolved galaxies. Longer high-resolution baryonic runs may require HPC.

**Profiling order:** (a) correct physics and units; (b) quantify memory/cost per physical step; (c) vectorize Python and reduce allocations; (d) move hottest kernels to compiled CPU/GPU; (e) partition domains and overlap communication; (f) consider AMR; (g) run scaling and deterministic checks again. Never optimize by disabling conservation validation silently. Render independently from simulation state.

**Run budget fields:** RAM cap, VRAM cap, per-run wall-time cap, disk cap, checkpoint interval, snapshot cadence, max steps, min timestep, maximum retry count, maximum refinement level and supported precision. Exceeding a cap must produce a clear incomplete-run status, not fabricated output.

## 11. Unknown physics configuration rules

Each frontier model must state: mathematical equations, free parameters and priors, known bounds, alternative outcomes, affected modules, compatible solver/initial conditions, scientific status, testable predictions, current evidence and known discriminators. Do not allow every alternative at once. Quantum entanglement is part of established quantum mechanics; Bell violations require multiple measurement settings. Matter-antimatter asymmetry is observed, but baryogenesis mechanisms are not settled. Dark matter can be represented phenomenologically without choosing a particle identity. Dark energy can be Λ as a reference while evolving-energy alternatives are tested. Black-hole horizons/astrophysics can be modeled separately from unresolved singularity interiors.

**Experimental setup:** choose reference config → pin seed and transfer inputs → fork compatible model variant → rerun relevant physics → collect common observables with uncertainties → compare against independent observations. A model that yields no halos, unstable disks, or no biochemistry may be a valid negative result.

## 12. Agent-driven implementation protocol (for a coding assistant)

Keep `AGENTS.md` minimal and point to `docs/CURRENT_TASK.md`, this manual and the blueprint. Give the agent exactly one phase and one independently testable vertical slice at a time. It must inspect existing files and code symbols before editing; preserve existing passing tests; never claim an unimplemented physics system is complete; avoid placeholder passes; cite equations/dataset releases in code comments and docs; show the changed file list, commands run, numerical diagnostic results, remaining risks and next smallest task.

**Copy-paste task template:**

```text
You are implementing ONE verified milestone in an existing multiscale
universe simulation repository. Read docs/IMPLEMENTATION_MANUAL.md,
docs/MASTER_BLUEPRINT.md, docs/CURRENT_TASK.md and the relevant existing
code only. First inspect the repository and preserve all existing tests.
Do not introduce scientifically unsupported behavior or fabricate emergent
objects. Use physically explicit units and reference frames. Implement the
smallest vertical slice with numerical verification, data provenance,
convergence tests, deterministic seeds, failure handling and README updates.
Do not label specified work as implemented. Run the relevant tests and
report exact commands/results, the model's validity domain, known gaps,
and the next dependency. Never silently replace existing project architecture.
```

**Definition of done for each task:** real implementation exists, tests demonstrate claimed behavior, data/equations documented, validity boundaries enforced, reproducible output, resource measurement, CI passing, integration dependencies updated, and validation matrix truthfully marked. A rendered demo alone does not meet definition of done.

## 13. Immediate task tickets (exact sequence)

**T00** Inventory ZIP, run baseline 22 reported tests, record exact environment and current repo layout.  
**T01** Add config/schema/unit library and strict reference-frame conversions; check existing constants.  
**T02** Integrate reference FLRW and verified cosmological clock; compare to independent trusted background output.  
**T03** Add run manifests, content hashes, deterministic RNG, checkpoint/resume.  
**T04** Pin and ingest physical transfer functions from CLASS/CAMB; validate spectrum and conventions.  
**T05** Generate correct cosmological particle initial conditions (and velocities) using a tested LPT method.  
**T06** Implement periodic PM Poisson/FFT gravity, CIC and comoving integrator; force fixtures.  
**T07** Add stable multistep cosmology runs and tests for linear growth/nonlinear structure.  
**T08** Add power spectra, halo finder, and scientific report with uncertainty.  
**T09** Add Godunov hydro isolated benchmarks before gravity coupling.  
**T10** Couple gas+gravity, primordial cooling, conservative star sinks.  
**T11** Add stellar tracks and yields, delayed feedback, enrichment and galaxy statistics.  
**T12** Add radiation/reionization and synthetic observables, then local compact/planet/chemistry modules.  
**T13** Add hypothesis laboratory with strict model compatibility and falsifiable outputs.  
**T14** Add detached viewer with scientifically labeled zoom and timeline.

Do not start T04 until T00–T03 and their tests meet acceptance criteria. Do not introduce galaxy visuals or star-formation subgrid code as substitutes for T06–T08 gravity validation.

## 14. What to do today

1. Extract the runnable starter in a **separate** project folder; verify `pytest` and the CLI output.
2. Open the existing universe ZIP (when available) in its own directory. Run and record its tests, then commit baseline.
3. Read `MASTER_BLUEPRINT.md` for the complete ~222-capability checklist; compare it to real source and datasets, not file names.
4. Enter T00 in `docs/CURRENT_TASK.md` and have a coding agent perform only the audit and minimal integration mapping.
5. After resolving state/unit mismatches, migrate the working FLRW baseline, or retain the existing stronger implementation if it is already validated.
6. Work toward T04–T08 to obtain a physically credible dark-matter cosmic web. That is the first major end-to-end astronomy milestone.

**Final caution:** no finite manual contains every future scientific discovery, and no program can be guaranteed to spontaneously create life. The way to keep this project moving smoothly is to preserve a truthful boundary between implemented, numerically correct, observationally supported, approximated and speculative modules.
