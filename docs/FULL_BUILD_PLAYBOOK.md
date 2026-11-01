# Universe Engine — full implementation and research build playbook

**Status:** executable educational/reduced-order components plus a detailed research engineering specification. **Not an exact universe, a fully coupled hydrodynamic galaxy model, proven abiogenesis engine, or simulation of all known physical processes.**

## A. Start the project on Windows

1. Install Python 3.11 or 3.12, Git, and VS Code. WSL2 Ubuntu is recommended for research HPC codes. For Windows native, use PowerShell and Python's `py` launcher.
2. Extract the supplied ZIP into a NEW directory (do not overwrite the original `big-bang-universe.zip` without diffing it). Open a terminal inside `universe-starter`.
3. Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\universe.exe validate --config config/reference.yaml
.\.venv\Scripts\universe.exe run --config config/reference.yaml
.\.venv\Scripts\universe-evolve.exe --grid 16 --steps 140 --seed 4242 --output runs/integrated-toy
.\.venv\Scripts\universe-physics.exe --output runs/physical-demo
```

4. Linux/macOS/WSL equivalent:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pytest -q
universe validate --config config/reference.yaml
universe run --config config/reference.yaml
universe-evolve --grid 16 --steps 140 --seed 4242 --output runs/integrated-toy
universe-physics --output runs/physical-demo
```

5. Inspect `runs/physical-demo/physical_results.json`, `runs/physical-demo/physical_fields.npz`, and the `runs/integrated-toy` outputs. These commands do NOT execute an ab initio Big Bang-to-biology run; they execute distinct validated-in-the-small demonstrators.
6. Before changing any code, commit it: `git init; git add .; git commit -m "baseline research simulator"` (configure Git identity if needed).

## B. Architecture to build and enforce

Use six layers: `core` (units, states, time, integrators, RNG), `physics` (microphysical rates and equations of state), `cosmology` (FLRW + linear transfer + gravity), `baryons` (hydrodynamics, heating/cooling, feedback), `localized` (stellar, disk, chemical systems), and `research` (experiments, validity, provenance, observables). The current modules are named `physics.py`, `field.py`, `evolution.py`, and `physical_models.py`; migrate gradually, preserving legacy CLI and tests.

**Mandatory interface for EACH module:** versioned name and physical model; structured input/output state; explicit SI or named code units; coordinate frame; timestep/range of validity; Jacobian/rate representation where applicable; conserved quantities and explicit source/sink accounting; uncertainty and calibration provenance; numerical convergence suite; serialization format; test fixtures. Reject unsupported regimes with descriptive errors rather than returning plausible-looking numbers.

**State exchange contracts:** gravity -> gas acceleration potential and timestep; gas -> star-formation region and metal composition; stellar populations -> radiation, mechanical energy, metal yields and remnant masses; stars/metal-rich gas -> disk initial conditions; disks -> planetary boundary conditions; planets -> atmosphere/ocean/chemical reaction environments; chemistry -> reproducing entities **only if a chemical heredity/replication mechanism truly appears in the chemical model**. Do not call separately seeded evolutionary models emergent life.

**State metadata:** run UUID, git commit, config hash, code version, random seed per stream, physical units, cosmological scale factor, precision, mesh size, refinement criteria, solver tolerances, timestamps and provenance of every dataset. Use atomic checkpoint writes and restart tests.

## C. First reliable cosmology product

1. Use a flat FLRW reference cosmology, with explicit baryon/cold matter/radiation/neutrino contributions and cosmological constant; compute expansion using numerical quadrature rather than arbitrary elapsed-time labels.
2. Integrate a standard Boltzmann/linear perturbation solver such as CLASS (`https://github.com/lesgourg/class_public`) or CAMB (`https://camb.info/`) as an *external validated input generator*. Record exact solver version, cosmology and power-spectrum units; verify σ8 and growth at z=0.
3. Generate periodic Gaussian Fourier phases with Hermitian symmetry and the transferred P(k); use Zel'dovich or 2LPT initial displacements with displacement/velocity consistency and sufficient start redshift. Check ensemble spectrum, zero mean and phase reproducibility. Replace the current synthetic `gaussian_field` for scientific runs.
4. Run PM or PM+tree gravity in comoving coordinates; use a symplectic or suitably derived cosmological kick-drift-kick with correct H(a) factors. Current `evolution.py` is reduced-order and not suitable for calibrated mass functions.
5. Run isolated mode-growth, two-body, force-symmetry, momentum, high-z linear-growth, Layzer-Irvine diagnostic and multiple-resolution convergence experiments. Compute halo mass using physical box size, Ωm and particle count.
6. Identify halos with a friends-of-friends and spherical-overdensity finder; validate halo mass function and power spectrum against independent published benchmarks. Density peaks alone are NOT halos.
7. Save epoch-by-epoch simulation snapshots and measurable power spectra. Gate: calibrated halo statistics + stable restart before enabling baryonic physics.

## D. Gas, galaxy, and black-hole physics

1. Choose an established code for production work rather than hand-writing the entire production simulation: e.g. RAMSES (`https://www.ics.uzh.ch/~teyssier/ramses/RAMSES.html`), Enzo (`https://enzo-project.org/`), GIZMO (`http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html`), or AREPO (access/license dependent). Validate current availability, compiler and license before integrating. An in-house hydro solver should first pass Sod shock tube, contact advection, sound-wave dispersion, Sedov blast, Jeans collapse and Kelvin–Helmholtz convergence.
2. State: conserved gas density ρ, momentum ρv, total energy, chemical species mass, metal fraction, magnetic field if MHD. Implement a conservative Godunov finite-volume scheme (HLLC/HLLE) in 3D with CFL-controlled steps; add gravity and cosmological factors using operator splitting with error checks. The provided 1D Rusanov solver is ONLY a starter.
3. Add ionization/excitation, molecular H2/HD and metal-line cooling Λ(T,n,Z,UV), radiative heating Γ, UV background, dust, reionization, and high-density optical-depth regimes. Solve stiff networks implicitly where needed; check positivity and energy bookkeeping.
4. Gravitationally bound collapsing gas creates star particles through a documented subgrid rule (density/temperature/Jeans/virial criterion, efficiency per free-fall time, random seed). Do not spawn star particles just because time passed.
5. Star-particle packets age and return mechanically and chemically tracked yields; inject supernovae, radiation, mass loss and energy with resolution-aware coupling. Fit free parameters only against a declared training set. Reserve independent observations for evaluation.
6. Add SMBH seed eligibility, accretion law, mergers and AGN feedback; track accreted mass/energy and enforce feedback conservation.
7. Observe gas/star maps, temperature distribution, star-formation rates, metallicity, luminosity functions and galaxy scaling laws. Require convergence, no negative densities, mass and metal-budget closure.

## E. Stellar evolution and nucleosynthesis

1. For real tracks, integrate a validated framework such as MESA (`https://docs.mesastar.org/`), or precomputed public tracks with provenance, rather than using L≈M^3.5 as physical stellar evolution.
2. Inputs: initial mass, composition, rotation/binary parameters and accretion history; outputs: luminosity, radius, temperature, lifetimes, remnant type, time-dependent yields, spectra and mechanical/radiative feedback. Interpolate within track domain ONLY; flag extrapolation.
3. Benchmark solar standard model and HR sequences, metallicity-dependent lifetimes, mass–luminosity relation, abundances and remnant mass distributions. For supernova nucleosynthesis, couple separately validated explosion/yield models; ordinary tracks do not solve all explosions.
4. Provide integrated-stellar-population sampling with stochastic IMF realizations and a deterministic expectation mode, keeping population mass exact.

## F. Planet formation and evolution

1. Build disk profiles from gas angular momentum and dust/metal masses, without conjuring solids from arbitrary draws. Couple evolving Σgas(r), Σdust(r), T(r), turbulence, volatile condensation fronts and stellar radiation.
2. Include a conservative mass-flux advection/diffusion scheme, drag and grain growth/fragmentation. Validate radial mass fluxes and nonnegative concentrations.
3. Introduce planetesimals, embryos, and gravitational N-body interactions after physically documented thresholds. Use a symplectic integrator (e.g. REBOUND `https://rebound.readthedocs.io/`) and verified collision/accretion handling; conserve total mass and angular momentum, accounting for escaping and accreted material.
4. Include planetary migration, resonances, gravitational scattering, giant-impact outcomes, atmospheres and heat budgets. The supplied annular solid conversion implements none of the orbital N-body effects.
5. Compare mass/radius/period distributions and orbital resonance statistics with independently sourced exoplanet catalogs. Never infer life from equilibrium temperature alone.

## G. Chemistry, protocells, evolution and ecology

1. Specify species as chemical structures, charge, elemental stoichiometry, thermodynamic formation energies and reaction mechanisms. Each reaction must conserve atoms and electric charge; enforce detailed balance where near equilibrium.
2. Build spatial reaction–diffusion–advection with finite-volume transport and concentration positivity; couple geochemical energy sources, water/rock boundaries, photon flux and thermal conditions.
3. Read reference chemical rate data where available; otherwise label rates *hypothetical*. Use sensitivity studies with credible uncertainty ranges; do not overfit unknown prebiotic chemistry.
4. Add explicit polymer chemistry: monomer availability, chain length, templating, hydrolysis, catalytic networks and compartment transport. Model sequence-level heredity, errors and selection **only after** a mechanism of differential replication exists.
5. Protocells require mass, membrane growth/transport, replication inheritance and fission, with chemical substrates drawn from the environment. Demonstrate persistence under changing feed and without forced population replenishment.
6. An evolutionary ecosystem then uses genomes/traits with conservation of nutrient and energy budgets, reproduction, mutation, competition and death. Bound environments can be controlled; emergence of real-world abiogenesis cannot be guaranteed.
7. Benchmarks: mass/charge residuals, energy supply, chemical equilibrium checks, reproducible reaction trajectories, replication fidelity, generation time and extinction behavior. Keep *chemical replication* distinct from *life* and *consciousness*.

## H. Speculative physics laboratory

Keep normal calibrated ΛCDM/Standard Model computation as the reference. Implement alternative dark matter power spectra/interaction terms, dark energy w(a), primordial initial condition variations, baryogenesis models, neutrino variants, black-hole models and quantum measurement interpretations as independent registered experiments. An alternative requires a well-defined equation, parameter bounds, initialization and predicted observable; interpretations sharing the Born rule often have no distinct prediction. Bell tests must vary analyzer bases and test CHSH, not merely produce 50/50 correlated bits. Compare likelihoods to the same held-out observations, with uncertainty accounting. Never declare metaphysical questions solved from visual resemblance.

## I. Full continuous build order (required release gates)

- **R0 — Build/reproduce:** clean installation, CLI, all tests passing, fixed seeds, no unexplained warnings, README commands working on target OS.
- **R1 — Cosmology validation:** correct linear power spectra, expansion background, CMB/BBN references and calibrated physical units.
- **R2 — Gravity validation:** convergence, resolved halo masses, FoF/SO halo catalogs, large-scale power spectrum.
- **R3 — Gas physics:** conservative hydro benchmark suite and thermo-chemical cooling, stable gravity coupling.
- **R4 — Galaxies:** star-particle formation, yields and feedback, observed populations within quantified tolerances.
- **R5 — Stars:** calibrated track and nucleosynthesis tables, correct time evolution, checked mass return.
- **R6 — Planets:** disk material transport, grain/planetesimal/embryo growth, N-body stability and planet catalogs.
- **R7 — Chemistry:** physically explicit networks, elemental conservation, chemical replication experiments in bounded planetary environments.
- **R8 — Evolution:** resource-coupled, replicating agents only after chemical-to-biological handoff is demonstrated; otherwise report independently seeded evolutionary toy.
- **R9 — Frontier experiments:** competing theories plus uncertainty-aware, held-out observable comparisons.
- **R10 — Observer and scale zoom:** synthetic observations, units/cosmological light-cone correctness, local region restart with boundary conditions.
- **R11 — Operations:** checkpoint/restart bitwise or tolerance-documented equivalence, crash recovery, memory monitoring, long-term job resource limits, reproducible experiment tracking.

**Release rule:** code + tests + conservation results + published benchmark comparisons + documentation + measured performance before claiming a release complete. A passing unit test by itself is insufficient scientific validation.

## J. Performance and scale recommendations

Begin on CPU with n=16–64 PM grids; profile with `python -m cProfile`. Increase resolution only when force, timestep and mass convergence have been characterized. Use NumPy initially; for GPU explore CuPy/JAX/PyTorch for suitable kernels after CPU parity tests. Avoid blindly moving FFT and mesh memory to a 12 GB GPU. For scientific galaxy runs use published HPC frameworks, MPI and distributed checkpoints. Storage estimation must include particle state, mesh fields, metal species, histories, and restart snapshots. Use separate compute budgets per region and per scientific model, with a clear distinction between computed and interpolated outcomes.

## K. Troubleshooting

- **CFL violation:** decrease timestep, and inspect sound speed, cooling and steep gradients.
- **Negative density or thermal energy:** enforce conservative positivity and shorten steps; never clip a broken global solver silently.
- **Inconsistent mass budget:** inspect flux boundaries and conversion source/sink ledgers.
- **Non-repeatable results:** use named RNG streams and fixed update order; document floating-point non-determinism across GPU/parallel backends.
- **No halos:** check linear transfer normalization, particle displacement convention, gravitational kick/drift factors and resolution rather than inventing peaks.
- **Artificial life appears without chemistry:** reject as biological emergence; it is seeded evolution.
- **Strong numerical/observational disagreement:** run separate unit, convergence, source calibration and held-out validation checks before tuning physics.

## L. What is genuinely delivered here

- Flat ΛCDM background and illustrative linear perturbations.
- Nonlinear toy PM gravity plus local peak detection and sample catalogs.
- Conservative periodic 1D finite-volume gas solver with optional acceleration, energy loss and gas-to-star conversion.
- Simplified analytic main-sequence estimates.
- Mass-conserving annular dust-to-planetesimal conversion.
- Driven chemical reaction network with an autocatalytic-like step.
- Independently seeded selection/mutation ecology experiment.
- CLI, reproducibility controls, 24 automated tests, machine-readable outputs.

**Missing research breakthrough:** unknown early-universe initial physics, complete quantum gravity, accurate abiogenesis from physical chemistry, consciousness; these are not problems with deterministic known instructions. For the known astrophysical regimes, the playbook above supplies actual module and verification instructions, but a full research-grade implementation still takes sustained development and scientific validation.
