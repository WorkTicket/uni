# UNIVERSAL ENGINE — Consolidated Master Architecture & Implementation Plan

**Version:** 1.0 · **Purpose:** A single source of truth for building an autonomous, scientifically testable, multiscale universe simulator, from a defensible early cosmological state through large-scale structure, stars, planetary environments, and bounded chemistry/biology experiments.

**Inputs consolidated:** the two supplied universe plans, including their 15-domain/222-capability inventory, frontier-physics proposals, additional missing infrastructure, sample validation targets, suggested documentation modules, and development milestones. This document consolidates and improves the plans; it **does not audit** the reported `big-bang-universe.zip`, 43 files, or 22 passing tests (the ZIP was not supplied).

## 0. The mission and non-negotiable reality checks

**Mission:** Configure a reference universe; press **Start Universe**; advance it autonomously through supported epochs; let modeled structures emerge through physics instead of author-created events; branch experimental universes; observe any epoch or region through scientifically consistent synthetic instruments; refine selected regions as computational resources permit.

**Three distinct claims:**
1. **Reference reproduction:** match established laboratory and astronomical observations within stated uncertainty and numerical error.
2. **Plausible emergent universe:** produce structures from modeled dynamics without scripting outcomes.
3. **Frontier experiments:** explore mathematically specified alternatives that are labeled speculative and compared with evidence.

**Never imply:** that the literal Big Bang singularity can be calculated from established physics; that a continuum of length scales is fully modeled because a registry covers it; that passing unit tests demonstrates agreement with nature; that microscopic quantum simulation can span all particles; that complex life or consciousness must emerge; or that a simulation validates a disputed physical interpretation.

**Foundational design decisions:**
- Reference baseline: spatially flat ΛCDM with measured cosmological parameters and uncertainties; allow a well-defined starting redshift / epoch appropriate to the selected solver. Earlier epochs use dedicated, separately validated cosmology modules; unknown Planck-scale physics stays explicitly speculative.
- Computation: use a **hierarchical state**, not a global atom-by-atom world. Cosmological volumes use fields and collisionless particles; stars use stellar populations or local stellar solvers; chemistry and quantum dynamics run on bounded local domains.
- Observable universe ≠ simulation box. Define a **comoving periodic representative volume** and its finite-size/cosmic-variance effects, not an exact copy of the entire observable universe.
- The word **emergence** applies only when outcomes come from stated dynamical laws or stochastic subgrid prescriptions, not when structures are spawned to satisfy a story.
- Preserve cosmological/global energy subtleties: local covariant stress-energy conservation applies; an expanding FLRW spacetime does **not** generally have a globally conserved total energy of the naive Newtonian kind.
- All experimental theories are versioned, uncertainty tagged, and compatible only with declared solvers and observables.

## 1. System architecture: one runtime, three modes

**Known Universe (reference):** calibrated standard physics, proven solver schemes, physically reasonable subgrid closure, observation-driven tests.

**Experimental Universe (hypothesis lab):** fork identical compatible initial states, vary one or more explicitly selected models/parameters, run comparison sweeps, calculate predictions/likelihoods, and record failures. An interpretation such as many-worlds is not by itself a new distinguishable force law.

**Observer (read-only by default):** inspect synthetic spectra, skies, gravitational lensing, geometry, and histories; initiate a **new** bounded fine-scale simulation when warranted. A camera click must not collapse a physical wavefunction unless an explicitly chosen measurement dynamics requires it.

### Runtime schematic

```text
Versioned configuration + data provenance + reproducible seed
        |
Initial-condition generator -> cosmology + particle/field state
        |
   Adaptive orchestrator / event scheduler / resource governor
        |
  +-----+---------+----------+----------+----------+
  | gravity /    | gas/MHD   | radiation| chemistry| local
  | cosmology    | & plasma  | transport| & nuclei | quantum
  +-----+---------+----------+----------+----------+
        | conservative exchange, validity/error checks
        v
Star/black-hole/galaxy/planet/biology submodels (when applicable)
        |
Checkpoint / diagnostics / synthetic observations / validation
        |
Branch / refine region / compare hypothesis / resume
```

**State hierarchy:** `UniverseRun` → `CosmologicalVolume` → `Patch/Cell/Particle` → `Halo` → `Galaxy` → `StellarPopulation / Star` → `Disk / Planet` → `LocalChemicalRegion / Organism` → `BoundedQuantumExperiment`. A lower-level state is instantiated only as needed; parents receive statistically and physically meaningful feedback.

## 2. Master inventory: all required capability families

**Completion labels for every line:** `not_started`, `specified`, `implemented`, `numerically_verified`, `observation_validated`, `experimental`, `blocked`. Documentation alone never counts as implementation. Every system below also needs: units; inputs; outputs; equations and references; assumptions; applicability range; uncertainty; conservation/constraint expectations; activation; resolution requirements; fallback; test fixtures; performance envelope.

### A. Kernel, numerical architecture, and infrastructure — FIRST
- [ ] Canonical state schema for particles, meshes, fields, objects, material species, and provenance.
- [ ] Units, dimensional analysis, physical constants, reference frames, precision and significant figures.
- [ ] Physical cosmological clocks: proper time, conformal time, scale factor, redshift; conversions.
- [ ] Comoving/proper coordinates, metric-dependent distances, and robust coordinate transforms.
- [ ] Versioned starting epochs, cosmological parameters and measured initial conditions with uncertainties.
- [ ] Central registry: model scopes, compatible solvers, domain boundaries and epistemic status.
- [ ] Differential-equation solvers: ODE, PDE, Poisson, stiff reaction systems, conservative finite-volume / particle schemes.
- [ ] Adaptive timesteps, multirate coupling, stable source-term splitting, positivity and CFL enforcement.
- [ ] Adaptive mesh refinement / particle refinement / bounded local re-simulation with hysteresis.
- [ ] Energy-momentum, charge, baryon/lepton accounting where physically applicable; numerical residuals.
- [ ] Local relativistic causality and correct light-travel effects; no superluminal signaling via entanglement.
- [ ] Stochastic solvers, physically justified noise, reproducible random streams and seed isolation.
- [ ] Subgrid closure with calibrated transfer functions, limits, uncertainty and resolution dependence.
- [ ] Hierarchical module activation, deactivation, input validation and conservative cross-scale handoffs.
- [ ] Adaptive resource budgets for CPU, GPU, RAM, VRAM and persistent storage.
- [ ] Parallel domain decomposition, load balancing, communication and deterministic options.
- [ ] Checkpoints, schema migrations, integrity hashes, restart, replay and branching.
- [ ] Binary scientific data formats, compressed snapshots, time series and object catalogs.
- [ ] Crash recovery, solver divergence detection, graceful degradation and explicit unsupported-regime failures.
- [ ] Headless batch execution plus optional interactive viewer; no dependence on rendering to advance time.

### B. Fundamental fields, particles and quantum mechanics
- [ ] Hilbert-space states, operators, density matrices, unitary evolution and reduced-state tracing.
- [ ] Born-rule measurement statistics, POVMs/projective measurement and environment-induced decoherence.
- [ ] Entanglement preparation, multi-basis correlation measurements, CHSH/Bell tests and no-signaling checks.
- [ ] Fermions, Pauli exclusion, exchange antisymmetry; bosonic statistics and occupation numbers.
- [ ] Standard Model field/particle catalog, quantum numbers, gauge groups and free parameters.
- [ ] Electromagnetic/QED interactions and suitable effective cross sections.
- [ ] Strong/QCD and confinement approximations; weak processes, decay and flavor-changing interactions.
- [ ] Higgs sector, spontaneous symmetry breaking and particle-mass parameterization.
- [ ] Pair production, scattering, annihilation, antiparticles and thermal abundance equations.
- [ ] Neutrino mixing, masses and transport where relevant.
- [ ] Charge, angular-momentum, energy-momentum and permitted global-quantum-number constraints.
- [ ] Quantum tunneling, barrier transmission and transition rates.
- [ ] Finite-temperature field/effective-theory approximations, renormalization scales and validity limits.
- [ ] Dense-matter quantum statistics; degenerate Fermi gases and equations of state.
- [ ] Explicit computational boundary: small exact quantum systems only; approximations and measured data elsewhere.

### C. Gravity, spacetime and background cosmology
- [ ] FLRW metric, Friedmann equations, radiation/matter/curvature/Λ terms.
- [ ] Background thermal and expansion history, including equation-of-state transitions.
- [ ] Metric relativity: geodesics, redshift, time dilation, gravitational lensing and frame transformations.
- [ ] General-relativistic field equations / numerical relativity for targeted compact regions.
- [ ] Primordial scalar/tensor fluctuation parameterization and transfer functions.
- [ ] Linear perturbation dynamics and matter growth in expanding backgrounds.
- [ ] Periodic cosmological gravity: particle-mesh / tree / hybrid N-body solution.
- [ ] Nonlinear gravitational collapse and halo finding, with force-softening / resolution controls.
- [ ] Cosmological horizons, observer past light cones and retarded observable fields.
- [ ] Gravitational waves and compact-binary dynamics in supported regimes.
- [ ] CMB anisotropy and polarization predictions; recombination/reionization optical depth.
- [ ] Cosmic-variance, simulation box and boundary-condition diagnostics.
- [ ] Large-scale power spectra, correlation functions, halo mass functions and lensing observables.

### D. Primordial universe, asymmetry and early nuclear physics
- [ ] Clearly defined earliest simulated epoch, with special experimental pre-hot-Big-Bang module if requested.
- [ ] Inflationary primordial-spectrum models and competing early-history parameterizations.
- [ ] Reheating / thermalization / high-energy equation-of-state models.
- [ ] Electroweak and QCD-era transitions using physically justified effective descriptions.
- [ ] CP violation, baryogenesis hypotheses and observed baryon-asymmetry baseline.
- [ ] Matter–antimatter number-density evolution, production and annihilation.
- [ ] Neutrino decoupling and thermal particle histories.
- [ ] Big Bang nucleosynthesis reaction network with cross sections and uncertainty.
- [ ] H/He/D/Li abundance and radiation-temperature predictions.
- [ ] Recombination / photon decoupling; relic photons and neutrinos.
- [ ] Predefined bridges between early-background integration and later perturbation/N-body initial conditions.

### E. Atomic, nuclear, molecular and material physics
- [ ] Nuclide/isotope masses, spins, decays, reaction-rate and binding-energy datasets.
- [ ] Fusion, fission, alpha/beta/gamma decay and neutron-capture reaction networks.
- [ ] Nuclear astrophysical nucleosynthesis, including explosive pathways.
- [ ] Atomic electron structure, Pauli occupations and shell configurations.
- [ ] Atomic spectra, transition probabilities, line broadening and radiative processes.
- [ ] Ionization/recombination and collisional excitation, dependent on state variables.
- [ ] Molecules, bond formation, electronic structure approximations and force fields.
- [ ] Thermochemistry: partition functions, equilibrium, reaction rates, detailed balance.
- [ ] Molecular kinetics, collisions, thermal transport and diffusion.
- [ ] Solid/liquid/gas phase transitions, materials, dust and nonideal equations of state.
- [ ] State-dependent interpolation and validity checks on all cached tables (T, density, composition, radiation, etc.).
- [ ] Microscopic calculations limited to bounded zoom domains, not the full universe.

### F. Matter transport, radiation, plasma and continuum mechanics
- [ ] Euler/Navier–Stokes hydrodynamics, shocks and Riemann solvers.
- [ ] Self-gravitating fluid solvers and gas cooling/heating coupling.
- [ ] MHD induction, Lorentz forces, magnetic energy and divergence constraints.
- [ ] Plasma kinetic or fluid closure; ionization and two-temperature effects where necessary.
- [ ] Frequency-dependent photon radiation transport and moment / Monte Carlo schemes.
- [ ] Scattering, absorption, emission, opacities and non-LTE limits when relevant.
- [ ] Neutrino transport for appropriate nuclear/compact-object regimes.
- [ ] Thermodynamics, heat conduction, convection, diffusion and entropy production.
- [ ] Multiphase mixing, phase change, dust-gas coupling and grain charging.
- [ ] Turbulence and unresolved-scale closures with convergence diagnostics.
- [ ] Cosmic ray and energetic particle feedback (when enabled).
- [ ] Rare-event sampling: explosive, collisional and stochastic transitions.

### G. Dark sector and frontier-physics lab
- [ ] Cold collisionless dark matter reference model and calibrated halo predictions.
- [ ] Warm, axion-like and self-interacting dark matter alternatives.
- [ ] Alternative gravitational laws only as standalone consistent dynamics with observables.
- [ ] Λ reference and alternate dark-energy equations of state; quintessence options.
- [ ] Inflationary/bounce and other early-universe alternatives.
- [ ] Quantum interpretations: standard operational theory; Everett, Bohmian or collapse dynamics as compatible experimental variants.
- [ ] Quantum gravity candidates: loop, string-inspired, causal-set or phenomenological treatments, each scoped to defined predictions.
- [ ] Black-hole information, evaporation and quantum-interior hypotheses.
- [ ] Sterile neutrino and other beyond-Standard-Model candidates.
- [ ] Leptogenesis, electroweak baryogenesis and other specified matter-asymmetry scenarios.
- [ ] Hypothesis compatibility constraints, versioned equation sets and parameter domains.
- [ ] Explicit observational discrimination test, or flag `currently indistinguishable`.

### H. Dark matter halos and galaxies
- [ ] Initial density field from validated power spectra / phases.
- [ ] Particle-based collisionless dynamics, force accuracy and halo-resolution tests.
- [ ] Cosmic web, halo/subhalo catalogs and merger trees.
- [ ] Intergalactic gas inflow, accretion shocks, cooling and chemical networks.
- [ ] Gravitational instability, gas fragmentation and star-formation prescription.
- [ ] Stellar initial-mass function and population synthesis.
- [ ] Supernova winds, stellar radiation and subgrid feedback.
- [ ] Metal enrichment, mixing, gas recycling and dust.
- [ ] Galaxy dynamics, merger interactions, morphology and population statistics.
- [ ] Supermassive black-hole seeding/accretion and active galactic nucleus feedback.
- [ ] Magnetized interstellar medium and cosmic-ray feedback at supported resolution.
- [ ] High-redshift reionization sources and their radiative effects.
- [ ] Galaxy-luminosity, stellar-mass, metallicity and clustering synthetic observables.

### I. Stars, stellar remnants and black holes
- [ ] Protostar and circumstellar accretion regimes.
- [ ] Hydrostatic equilibrium, stellar structure and radiation/convective energy transport.
- [ ] Fusion networks, nuclear reaction rates and element yield tables.
- [ ] Opacity, equation-of-state, composition and mixing data.
- [ ] Stellar rotation, winds and mass loss.
- [ ] Binary and multiple-star dynamics, mass transfer and mergers.
- [ ] Stellar evolution tracks and phase transitions, with population-level approximations.
- [ ] White-dwarf, degenerate-matter and neutron-star models.
- [ ] Core-collapse / thermonuclear supernovae and feedback deposition.
- [ ] Compact-object mergers, gravitational waves and relativistic dynamics.
- [ ] Schwarzschild/Kerr black-hole spacetime, accretion and lensing.
- [ ] Jets, AGN and black-hole/gas feedback.
- [ ] Hawking radiation only where relevant; long timescales and speculative interior physics kept separate.
- [ ] Metals, dust, photons, neutrinos and mechanical energy returned to surrounding media.

### J. Planetary systems, geology, weather and habitability
- [ ] Protoplanetary disk gas/dust evolution.
- [ ] Grain sticking, fragmentation, streaming instability and planetesimal growth models.
- [ ] Accretion, core growth, migration and orbital resonances.
- [ ] N-body orbits, collisions and long-term stability.
- [ ] Moons, small bodies, impact distributions and debris evolution.
- [ ] Planet differentiation, interiors, mantle/core thermal evolution.
- [ ] Volcanism, tectonics, cryospheres and geological cycling.
- [ ] Magnetic dynamos and magnetosphere interaction with stellar wind.
- [ ] Atmosphere sources, escape and composition evolution.
- [ ] Radiative-convective climate, atmospheric circulation and greenhouse effects.
- [ ] Water inventory, ocean circulation, weathering, hydrology and erosion.
- [ ] Carbon/nitrogen/sulfur/redox chemistry and surface geochemistry.
- [ ] Stellar radiation, tides, impact events and orbital-climate forcing.
- [ ] Habitable-environment metrics and uncertainty; never equate `habitable` with `inhabited`.
- [ ] Downscaled local planet or reaction volume based on planetary boundary conditions.

### K. Prebiotic chemistry / origins of life (experimental, NOT established)
- [ ] Explicit chemical species graph with elemental/charge balancing.
- [ ] Energy gradients, geochemical feedstocks and environmental forcing.
- [ ] Non-equilibrium chemical reaction-diffusion dynamics.
- [ ] Organic molecule synthesis, degradation and environmental transport.
- [ ] Polymerization, hydrolysis, templating and concentration mechanisms.
- [ ] Autocatalysis, self-sustaining networks and kinetic stability.
- [ ] Compartments, membranes and protocell diffusion/transport.
- [ ] Replication hypotheses, fidelity, error thresholds and heredity.
- [ ] Protocell growth, division, competition and selective effects.
- [ ] Metabolic and genetic-code origin hypotheses as alternatives.
- [ ] Quantitative transition criteria from chemical replicators to evolving cellular agents.
- [ ] Multiple plausible outcomes, including `no emergence of life`.
- [ ] Robustness and sensitivity tests across many local trials and parameter values.

### L. Biology, adaptation and ecology (after a justified starting biology state)
- [ ] Biomolecules, cellular membranes, transport and energy metabolism.
- [ ] DNA/RNA/protein processes, genes, mutation and recombination.
- [ ] Regulation, signaling, replication and cellular division.
- [ ] Heredity-to-phenotype maps, developmental regulation and cell differentiation.
- [ ] Homeostasis, metabolism, biochemical constraints and resource budgets.
- [ ] Multicellular tissues and developmental mechanics where simulated.
- [ ] Populations, competition, selection, genetic drift and speciation.
- [ ] Microbial communities, symbiosis and host–pathogen interactions.
- [ ] Photosynthetic primary production and nutrient cycling.
- [ ] Food webs, ecosystem dynamics and environmental feedback.
- [ ] Migration, extinction, ecological succession and perturbation response.
- [ ] Sensors, neural systems, behavior, learning and agent-based cognition where supported.
- [ ] Biosphere effects on atmospheric composition and climate.
- [ ] Explicit distinction between simulated intelligence and any claim about subjective consciousness.

### M. Data, scientific validation, inference and uncertainty
- [ ] Versioned source-data packages with provenance, references and usage rights.
- [ ] Schema/units integrity checks and physical-range validation.
- [ ] Analytical unit tests, manufactured solutions and conservation/constraint tests.
- [ ] Convergence tests across resolutions, time steps and solver methods.
- [ ] Regressions against measured particle, nuclear and atomic properties.
- [ ] Early thermal history, BBN abundance and CMB reference comparisons.
- [ ] Cosmological expansion, power spectrum, halo and galaxy-population tests.
- [ ] Stellar tracks, nuclear yields, transients and synthetic photometric tests.
- [ ] Orbital, planetary energy-balance and laboratory chemistry benchmarks.
- [ ] Sensitivity / uncertainty propagation, model discrepancy and confidence intervals.
- [ ] Bayesian inference / parameter fitting with reserved out-of-sample data.
- [ ] Ensemble statistics, cosmic variance and stochastic repeatability.
- [ ] Hypothesis evidence ranking without confusing a good fit with unique confirmation.
- [ ] Complete scientific validation matrix: specified vs coded vs numerically tested vs empirically validated.
- [ ] A run-specific reproducibility manifest and machine-readable scientific report.

### N. Observation and visualization
- [ ] Observer frames, proper motion, past light cones and lookback time.
- [ ] Synthetic telescopes, spectra, filters, instrumental noise and resolution limits.
- [ ] CMB maps, galaxy catalogs, redshift-space distortions and projected light.
- [ ] Gravitational lensing, relativistic rendering and ray tracing.
- [ ] 3D cosmic web, galaxy, star system and planetary scenes.
- [ ] Object selection, measurable properties, spatial/time navigation.
- [ ] Region-of-interest refinement with boundary-condition provenance.
- [ ] Compare two runs using matched statistics and uncertainty bands.
- [ ] Timeline snapshots / replay / branching, never imply snapshots run backward physically.
- [ ] Explicit labels for measured, computed, interpolated and artistic visual elements.
- [ ] Headless operation so visual output cannot block scientific execution.

### O. Runtime operations, experiments and product reliability
- [ ] Reproducible config presets with strict input validation.
- [ ] Run IDs, model and source-data versioning, random seed record.
- [ ] Pause/resume, fork and checkpoint validation.
- [ ] CLI batch runner and optional UI for config, monitoring and navigation.
- [ ] Job queue, resource accounting, compute caps and timeouts.
- [ ] Logging, profiling, performance metrics and physics anomaly alarms.
- [ ] Fault isolation and recovery without silently corrupting scientific state.
- [ ] Automated tests and benchmark pipelines in continuous integration.
- [ ] Parameter sweep, sensitivity study and competing-hypothesis orchestration.
- [ ] Results DB, snapshots, object catalogs and exportable plots/reports.
- [ ] Schema/model migrations and compatibility policy.
- [ ] User-visible limits, known issues and unsupported dynamics.

## 3. Formal physics-module contract (REQUIRED for every active solver)

Each module must declare: `id`, `version`, `equations`, `reference`, `scientific_status`, `input_fields`, `output_fields`, `units`, `coordinate_frame`, `validity_domain`, `required_resolution`, `time_integrator`, `max_stable_timestep`, `conserved_quantities`, `constraints`, `numerical_error_estimator`, `model_error_estimator`, `dependencies`, `activation_predicate`, `deactivation_predicate`, `coupling_operator`, `fallback_policy`, `test_vectors`, and `performance_cost_estimate`.

**Proposed canonical interface (language-neutral):**

```text
prepare(state_view, parameters, datasets) -> initialized_model
propose_step(state_view, t, budget) -> {dt_max, refinement_requests, validity_flags}
advance(state_view, dt, boundary_conditions) -> {state_delta, boundary_fluxes, diagnostics}
validate(before, after, fluxes) -> {constraint_residuals, error_estimates, warnings}
observe(state_view, instrument) -> synthetic_measurement
checkpoint()/restore() -> versioned serializable state
```

**Contract rules:** A solver cannot silently exceed its validity domain. If it cannot advance, it must request refinement, switch to a supported fallback, reduce timestep or return a clear error. A coupling exchange must be dimensionally valid and conserve relevant *local* quantities after accounting for boundary fluxes, source terms, spacetime geometry and explicitly modeled sinks.

**Handoff examples:**
- Linear perturbation → N-body initial particles: preserve power spectrum and growth normalization.
- Baryonic gas cell → stellar population particle: transfer initial mass, metallicity and momentum; account for reservoirs and feedback.
- Stellar population → gas: deposit time-dependent radiation, metals, momentum and returned mass.
- Cosmological halo → detailed galaxy zoom: import history, tides, external radiation, inflows and boundary fluxes.
- Planet climate cell → chemistry chamber: import atmosphere/ocean composition, temperature, pressure, irradiation and mineral/energy flows.
- Quantum chemistry → coarse reaction table: retain domain of temperature/pressure/composition, interpolation and uncertainties.

## 4. Scientific data manifest — imported measurements are not 'emergent'

Reference sources to retrieve, pin to a release and validate before integration:

| Source | Intended data | Validation approach |
|---|---|---|
| CODATA / NIST | fundamental constants, units | dimensional and precision tests |
| Particle Data Group | particle masses, lifetimes, couplings and reviews | selected decay/threshold benchmarks |
| IAEA nuclear services and evaluated libraries | nuclides, masses, cross sections, decays | Q-values, reaction and isotope tests |
| NIST Atomic Spectra Database | atomic energy levels and transitions | air vs vacuum spectral reference tests |
| Atomic/molecular reference databases | opacities, molecular lines, cooling rates | temperature/composition-dependent tests |
| NASA LAMBDA / Planck | cosmological observations | CMB/expansion likelihood checks |
| DESI and other galaxy surveys | BAO, redshift distributions and clustering | held-out cosmology comparisons |
| IllustrisTNG public simulations | galaxy and halo reference catalogs | statistical population comparisons, not pixel matching |
| Stellar evolution/reference models | tracks, yields, opacities | HR diagrams, lifetimes and abundance evolution |
| Exoplanet/orbital/planetary observations | planetary and climate benchmarks | measured-orbit and energy-balance tests |
| Laboratory reaction databases | thermochemistry and kinetics | conservation, equilibrium and kinetics tests |

For every imported dataset store `source`, `version`, `download_date`, `license`, `checksum`, `units`, `coordinate convention`, `uncertainty`, `coverage`, `interpolation rule`, `out-of-range policy` and `attribution`.

**Important fidelity check:** hydrogen H-alpha is roughly 656.28 nm in standard air, with vacuum values around 656.46 nm depending on the exact transition convention; a reported 656.11 nm needs inspection of physical approximations and air/vacuum conventions before claiming close precision. The previous reported iron electron configuration and H→He energy outputs are useful regression tests but do not validate all of chemistry/stellar evolution.

## 5. Frontier theories: comparison protocol

Each speculative model must include: (1) exactly which established equation it modifies, (2) parameter priors, (3) consistency constraints, (4) predictions distinguishing it from baseline if any, (5) observation/test plan, (6) known exclusions and limitations, (7) compatibility rules with other hypotheses.

| Unknown | Reference baseline | Example experiments | Key constraint |
|---|---|---|---|
| Early-universe initial state | chosen hot-Big-Bang/inflation-compatible initial conditions | different inflation potentials, bounce scenarios | do not assert a first-moment solution |
| Matter–antimatter asymmetry | fitted baryon asymmetry | leptogenesis, electroweak alternatives | satisfy observed abundances |
| Dark matter | cold collisionless matter | warm, self-interacting, axion-like | lensing, structure and CMB |
| Dark energy | Λ | evolving w(z), quintessence | expansion/growth data |
| Quantum measurement | standard operational quantum mechanics | Everett, Bohmian, objective collapse | only objective collapse may introduce distinct measurable dynamics, depending on model |
| Entanglement | standard joint quantum state | specific nonstandard models | multi-basis Bell inequality, no signaling |
| Quantum gravity | GR below unknown quantum-gravity domain | loop/string/causal phenomenology | separate speculative Planck-scale regimes |
| Black-hole information | GR + semiclassical results where valid | bounce, remnants, information proposals | distinguish established exterior from unknown interiors |
| Neutrinos | measured mixing with constrained mass options | sterile species / other mass hypotheses | cosmic and laboratory bounds |
| Abiogenesis | initial lifeless chemistry | RNA-world, metabolism-first, hydrothermal models | no guaranteed creation of living systems |
| Consciousness | none inferred from physics solver | computational/neural models | never claim subjective experience verified |

**Entanglement test requirement:** correlated 50/50 results for a Bell pair are insufficient alone. Require measurements in multiple bases and a CHSH violation within statistical confidence, plus no-signaling sanity checks.

**Many-worlds requirement:** branch bookkeeping is an optional interpretive visualization of unitary/decohering evolution, not automatic physical duplication of universes with distinctive experimental outcomes.

## 6. Build order: the critical path that minimizes rework

**Rule:** DO NOT implement everything in parallel. Every phase produces a runnable artifact and advances only if its entry and exit criteria are met. Never build viewer-first spectacle and retrofit physical correctness.

### Phase 0 — Audit and freeze the foundation
1. Unzip existing project and inventory actual source, tests, dependencies and datasets; count independently.
2. Build a traceability table linking every existing file to an item in §2 and to a runnable test.
3. Record baseline test command/output, configuration format, executable targets and known failures.
4. Freeze current run format; plan backward-compatible migrations.

**Exit gate:** reproducible baseline build; no unverified claim that all documented rules are executable.

### Phase 1 — Minimal universe kernel
- Implement canonical state, units, clocks, deterministic seed, module registry, scheduler, numerical solvers, checkpoint I/O and validation hooks.
- Verify stable integration, lossless save/restore within declared precision, local conserved fluxes and deterministic small-run repeatability.

**Exit gate:** small headless physical system advances, restarts and yields equivalent results; unsafe solver states fail explicitly.

### Phase 2 — Defensible reference cosmology
- Implement Friedmann expansion with measured ΛCDM parameters, thermal history and primordial scalar perturbations.
- Generate a Gaussian random density field from a versioned input power spectrum, with finite-box/aliasing safeguards.
- Implement linear perturbation checks before nonlinear evolution; derive N-body initial conditions with appropriate approximation.

**Exit gate:** expansion and linear-growth benchmarks agree with reference calculations within predeclared tolerances, without fitting to test output.

### Phase 3 — Cosmic gravity and halos
- Add N-body gravity, expansion-aware integrator, halo detection, object catalogs, power-spectrum and mass-function diagnostics.
- Run small → medium → larger resolutions to establish a trustworthy convergence envelope.

**Exit gate:** halos emerge from initial perturbations without manual spawning; stability/convergence and comparison plots pass.

### Phase 4 — Baryonic physics and first stars
- Add gas hydrodynamics, thermochemistry, cooling, radiation approximations, self-gravity, star-formation subgrid, feedback, and enrichment.
- Use measured cooling/rate tables with local state and interpolation validity.

**Exit gate:** gas can collapse; stellar populations form via stated criteria; energy/mass/metal accounting passes; predicted populations are compared to observations.

### Phase 5 — Galaxies and compact objects
- Add galaxy and black-hole growth, mergers, accretion feedback, enriched ISM and population observables.
- Compare star-formation histories, halo occupation, luminosity/mass functions and clustering.

**Exit gate:** at least one representative cosmic box produces a self-consistent statistical population; physics and numerical uncertainties are reported.

### Phase 6 — Bounded stellar/planetary zoom models
- Create local star, disk and planet models receiving parent-scale boundary conditions.
- Build long-term orbits, impacts, stellar tracks, planetary atmosphere/climate, geochemistry and environment history.
- Return physically justified bulk outputs to parent states when appropriate.

**Exit gate:** selected systems evolve autonomously with validated orbits/energy budgets; global and zoom representations remain consistent within stated error.

### Phase 7 — Chemistry and open-ended evolution experiments
- Build thermodynamically constrained reaction networks, reaction–diffusion and compartment dynamics.
- Then, only if self-replication/heredity actually arise or are explicitly seeded, integrate an evolutionary ecology model.

**Exit gate:** chemical runs obey mass/charge accounting and demonstrate reproducible kinetic behavior; report if no life emerges. A seeded organism scenario must be labeled as seeded.

### Phase 8 — Frontier physics and inference
- Create isolated branches for dark matter, dark energy, quantum measurement, baryogenesis and selected GR alternatives.
- Run controlled matched-condition ensembles; compare predictions and out-of-sample observations.

**Exit gate:** every selectable hypothesis has correct status, compatibility constraints, meaningful experiment tests or an explicit `not empirically distinguishable` label.

### Phase 9 — Observer, long-run hardening and release
- Add scientific telescope/spectroscopy tools, light-cone views, snapshots, zoom requests and side-by-side run comparisons.
- Perform long-run load, restart, corruption and performance tests. Produce reproducible manifests and known-limit reports.

**Exit gate:** a headless run can be started, stopped, restored, monitored, queried and branched with reproducible physics outputs; UI does not change physical results.

## 7. Continuous quality gates (no phase may bypass these)

1. **Unit/typing gate:** every quantity, coordinate convention and dataset passes units/schema checks.
2. **Numerical gate:** stable timesteps; convergent quantities; independently checked analytical solutions.
3. **Physical constraint gate:** local conservation laws, proper boundary fluxes, relativistic and thermodynamic constraints.
4. **Cross-scale gate:** feedback and handoff closure across different resolutions/solvers.
5. **Empirical gate:** compare against observations and uncertainties; keep validation data separate from calibration data.
6. **Reproducibility gate:** seed, code commit, numerical settings, dependencies, models and source-data versions saved.
7. **Performance gate:** tracked elapsed time, peak RAM/VRAM, state size and cost per simulated time interval.
8. **Scientific honesty gate:** visual approximations, subgrid prescriptions, hypotheses and unknown regimes clearly labeled.

**Proposed baseline verification examples:** analytical Friedmann solutions; matter/radiation scaling; sinusoidal perturbation growth; Zel'dovich pancake (with resolution caveats); two-body orbit; gravitational force test; Sedov–Taylor blast wave; shock tube; hydrostatic polytrope; fusion Q-value; stable atomic transition comparisons; recombination benchmark; BBN abundance reference; Bell/CHSH experiment; low-dimensional quantum unitarity; stellar polytrope/track test; climate radiative balance; chemical detailed balance; checkpoint equivalence; N-body force convergence; halo mass function with confidence bands. Tests have individual tolerances and documented approximations.

## 8. Run configuration, experiment records and minimal interfaces

**Example conceptual configuration (not proof that the current repo supports it):**

```yaml
run:
  name: reference-small-box
  seed: 314159
  mode: reference                 # reference | hypothesis
  start: {kind: redshift, value: 99}
  stop: {kind: redshift, value: 0}
  box: {comoving_mpc_h: 20, boundaries: periodic}
  resources: {max_ram_gb: 16, max_vram_gb: 10, checkpoint_every_steps: 100}
cosmology:
  model: lcdm
  parameters_dataset: planck_baseline_versioned
physics:
  gravity: nbody_pm
  baryons: off                   # deliberately deferred for Phase 3
  dark_matter: cold_collisionless
  dark_energy: cosmological_constant
numerics:
  steps: adaptive
  resolution: {particles_per_axis: 64, mesh_per_axis: 128}
validation:
  enforce_units: true
  abort_on_invalid_state: true
  compare_to: [friedmann_growth, linear_power_spectrum]
outputs: {snapshots: true, diagnostics: true, provenance: true}
```

Each experiment record should include: `run_id`, `parent_run_id`, timestamp, code revision, hardware, seed, model IDs, parameter values and priors, domain/resolution, dataset hashes, relevant scientific assumptions, numerical tolerances, diagnostics, exception history, validation results and generated observables.

### Definition of DONE (module-level)
- [ ] Working implementation exists in runtime, not just docs.
- [ ] It executes a representative end-to-end scenario.
- [ ] Units, domains, dependencies, failure policies and tests are declared.
- [ ] At least one independent numerical check exists.
- [ ] Benchmark/reference data and citations are stored when available.
- [ ] Memory/runtime cost is measured.
- [ ] Documentation explains known physics vs approximations vs speculative choices.

### Definition of READY TO PRESS START (first functional release)
- [ ] Phase 0–3 gates pass: reproducible reference expanding N-body universe.
- [ ] A user can select initial conditions and launch a run without changing code.
- [ ] Density perturbations grow into halos through solved dynamics, not scripted events.
- [ ] The simulation checkpoint/resumes and produces validated scientific diagnostics.
- [ ] Unsupported early epochs/extra physics cannot be silently selected.

### Definition of AUTONOMOUS COSMIC EVOLUTION (larger release)
- [ ] Through Phase 5, gas/stars/galaxy feedback are integrated and validated for a declared resolution regime.
- [ ] Later planetary/biological capabilities are explicitly independent extensions, not falsely described as already emergent.
- [ ] Full run and branch experiments are reproducible and scientifically interpretable.

## 9. Repository organization and documentation mapping

Preserve and extend current documents rather than blindly creating duplicates. Suggested organization:

```text
big-bang-universe/
  README.md
  docs/
    MASTER_BLUEPRINT.md                # this file
    PHYSICS_SCOPE_AND_UNKNOWNS.md
    SCIENTIFIC_VALIDATION_MATRIX.md
    MODEL_CONTRACT.md
    INITIAL_CONDITIONS.md
    MULTISCALE_COUPLING.md
    FRONTIER_THEORY_REGISTRY.md
    DATA_PROVENANCE.md
    ROADMAP.md
    OBSERVATION_AND_ZOOM.md
  configs/reference/ configs/experiments/
  data/manifests/                 # download manifests, not uncontrolled raw binaries
  src/core/                       # clocks, units, state, schedule, coupling
  src/numerics/                   # integrators, mesh, gravity, statistics
  src/physics/                    # microphysics, gas, radiation, cosmology
  src/astrophysics/               # halos, stars, galaxies, black holes
  src/planetary/
  src/chemistry/
  src/biology/
  src/frontier/                   # explicit experiments only
  src/observers/
  tests/unit/ tests/integration/ tests/convergence/
  tests/observations/ tests/reproducibility/
  benchmarks/
  outputs/                        # generated; excluded from source control
```

**Previous proposed docs `20_initial_conditions.md` through `35_computational_architecture.md` remain valid topic suggestions.** Use their content as sections or references, not mandatory duplicate filenames if the existing repository already implements them.

## 10. Practical compute strategy for a local workstation

- Begin with 32³–128³ collisionless particles and a comparable mesh, then scale cautiously with profiling. These are starting points, not accuracy guarantees.
- Favor reliable CPU reference solvers before GPU kernels; compare GPU outputs numerically, not just visually.
- Use compressed/checksummed checkpoints and sparse time-series diagnostics instead of storing every field at every step.
- Use coarse full-volume simulations plus **independent localized zoom** experiments; do not promise global microscopic detail on a 12 GB VRAM GPU.
- Reserve expensive full GR, quantum, chemistry and biology solvers for physically justified small regions.
- Provide a graceful `computationally_unavailable` result instead of inventing a high-resolution state.
- Track resolution effects and random-ensemble variance; a beautiful picture is not a precision benchmark.

## 11. Scientific baseline, observation targets, and model honesty

A reference ΛCDM preset can begin from Planck-era values, for example approximately `H0 = 67.4 km/s/Mpc`, `Ωm = 0.315`, `n_s = 0.965`, `σ8 = 0.811` with accompanying uncertainties and other required parameters. Values are model/analysis-dependent, not intrinsic truths; version and cite the selected parameter set. Specify `Ωb`, `Ωcdm`, radiation, curvature, reionization optical depth, scalar amplitude and pivot, neutrino assumptions, and initial transfer functions before claiming a complete initialization.

**Do not confuse** a first observationally compatible ΛCDM run with a literal reconstruction of our exact cosmic history. Statistical similarity is the primary valid target. Dark-energy evolution, quantum gravity, black-hole interiors and consciousness remain unknown in different senses; avoid combining all guesses in one allegedly authoritative run.

## 12. First coding tasks, in exact order

1. **Audit** provided ZIP when available: directory tree, build scripts, runtime language, test output and actual executed modules.
2. **Create `SCIENTIFIC_VALIDATION_MATRIX`** using §2 and mark evidence; never mark presumed completion.
3. **Define the module API and state schema** before extending solvers.
4. **Pin scientific datasets/units/parameter preset** and verify checksums.
5. **Implement and test the reference FLRW background** with documented assumptions.
6. **Implement spectrum-to-initial-field generation** and validate statistics.
7. **Integrate small periodic cosmological gravity** with convergence and checkpoint tests.
8. **Produce first validated end-to-end run** with expansion, perturbations, emergent halos and scientific report.
9. **Then** integrate gas cooling and stellar population prescriptions.
10. **Only after that** add galaxy feedback, local zooms, chemistry/biology and alternate hypotheses.

## 13. Acceptance philosophy

The project should move forward by **vertical working slices**, each scientifically tested, rather than adding hundreds of impressive but disconnected stub files. The ultimate success criterion is a reproducible, physically justified run through the scopes actually supported, accompanied by diagnostics that make its assumptions and failures visible. There is no scientifically honest `100% of the real universe` completion flag.

---

### Version notes
- Consolidated the 15-domain inventory, previously proposed frontier/observer/validation modules, and the follow-up missing-system list.
- Added explicit numerical architecture, contracts, phase gates, physics caveats, data provenance, operational recovery and a first runnable configuration.
- Resolved a major ambiguity: **a live simulated universe is a layered computation, not a precomputed universe-wide exact wavefunction.**
- Preserved the intended outcomes: autonomous cosmic evolution, local scientific exploration, and credible investigation of unknown physics.
