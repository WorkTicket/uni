# Scientific limits (current)

Verification tiers: T0 runs, T1 numerical, T2 integrated, T3 physically validated, T4 hypothesis.

| Subsystem | Status | Tier | Limit |
|---|---|---|---|
| FLRW background | implemented, cross-checked against independent code (age agrees to 1e-5) | T1 | flat LCDM only; radiation/neutrino treatment minimal |
| Linear density field | implemented | T0 | no transfer function |
| Particle-mesh gravity | prototype | T0/T1 | not converged science grade; no halo finder |
| **1D Euler solver (R3)** | **implemented + verified** | **T1** | 1D, ideal gas, dimensionless; cooling/chemistry not coupled |
| **Gravity-hydro coupling + 1D collisionless particles with CIC (R4)** | **implemented + verified (1D, periodic)** | **T1** | no 3D, no comoving/cosmological terms, no non-trivial hydrostatic test; accuracy verified in linear regime and pre-shell-crossing only; energy conserved to splitting error (drift reaches ~4e-3 in strong collapse) |
| Thermochemistry (R5) | not started | - | old `gas_step` cooling is ad hoc |
| Stars, galaxies, planets | phenomenological catalogs | T0 | not simulated physics |
| Abiogenesis | NOT SOLVED | - | prebiotic network is a toy mass-action model, not life |
| Observatory (R16) | 1D server + viewer working; 3D not started | T0/T1 | viewer never rendered in a browser here; 1D only |
No claim of observational validation (T3) is made for any subsystem.

## 3D viewer (observatory)

- The background expansion uses the same flat LCDM equations and the parameters of `config/reference.yaml`. Epoch timings are tabulated from them; the cinematic time axis is logarithmic and is not to scale.
- The cosmic web is a truncated Zel'dovich approximation of a Gaussian random field (BBKS-like spectrum). It is not an N-body or hydrodynamic result. Halo and galaxy formation times are heuristics from peak height.
- Early epochs (inflation, quark/plasma era, recombination) are schematic visualisations, not simulations; colours and brightness are exaggerated for legibility.
- Galaxies, stars, planetary systems, terrain, atmospheres and clouds are procedural and seed-generated, with plausible but not predictive physics (Kepler orbits, frost line, equilibrium temperature biomes). Nothing implies life or habitability.
- Rendering is approximate (no depth buffer, analytic impostors, single-scattering atmosphere).

## Planet environment and chemistry (Milestones A-B)

- Climate is a 1D latitudinal energy-balance model with a linear (Budyko) outgoing-radiation law, Earth-tuned albedo, diffusion and heat capacity. Valid only roughly for -50..+57 C global mean and 0.1-100x present CO2; outside this the model flags results rather than hiding them. Moist/runaway greenhouse is NOT modelled; planets above ~280 W/m2 absorbed are flagged invalid.
- The carbon cycle is the Walker-style weathering feedback with an assumed lumped response time; only its equilibrium is meaningful. Mobile vs stagnant lid is an input. Atmospheric escape, ocean carbon buffering, biology and impacts are absent.
- Carbonate pH uses fixed alkalinity and 298 K-dependence for K1/K2/Kw (recalled textbook values, not re-verified).
- Every empirical constant is listed with its provenance in `planet/constants.py`; 'recalled' means written from memory and not checked against the primary source here.
- The chemistry engine is exact for what it models (mass action, conservation, detailed balance) but ships NO measured reaction data. Reaction parameters used in tests are illustrative. Results say nothing about whether any real pathway to life occurred.
- No life, replication, compartments or evolution exist yet (Milestones C-H). The 3D viewer is not connected to this state.

## Prebiotic organisation (Milestone C)

Sourced: peptide-bond condensation free energy ~ +3.5 kcal/mol (American Peptide Society knowledge-base page; a secondary source, see `prebiotic/provenance.py`). Everything else below is ASSUMED and labelled.

- **Polymers** (`prebiotic/polymer.py`): monomer-addition chain growth vs hydrolysis with thermodynamically consistent reversibility; water activity enters the equilibrium constant. dS = 0 is assumed. Rate prefactors are placeholders; real uncatalysed peptide hydrolysis is extremely slow (years or longer; recalled, not verified), so the model does NOT claim a realistic prebiotic timescale. Wet/dry cycling is an imposed schedule; the retained oligomers in the test are a consequence of that schedule plus the assumed kinetics, not a discovery. Chain-chain ligation, activation chemistry, sequence and chirality are not modelled.
- **Autocatalytic sets** (`prebiotic/autocatalysis.py`): RAF algorithm and the random binary-polymer model. Which molecule catalyses which reaction is random with probability p; the appearance of RAFs is a property of that assumption, not evidence about real chemistry.
- **Protocells** (`prebiotic/compartment.py`): osmotic swelling, tension-enhanced amphiphile uptake, lysis at a strain limit, division at the geometric criterion A/A_sphere >= 2^(1/3). Permeability, uptake law, tension gain, lysis strain and critical vesicle concentration are placeholders (the primary source could not be fetched). Solute content is an abstract impermeant count; it does not yet come from the chemistry engine, and there is no replicator inside.
- The integrator's tolerance can leave tiny negative concentrations (~1e-13) in extremely stiff cases; conservation is checked in aggregate.
- Not done: coupling protocell contents to reaction networks, heritable replication (Milestone D), any claim that life emerged.

## Heritable replication (Milestone D)

- The copying chemistry is a SUPPLIED mechanism (`replicators/population.py`): template-directed copying with per-site substitution error mu, one monomer unit per site, no indels, uniform substitution. This does not show that replicators arise from chemistry.
- Replication/decay rates of the default mechanism derive from a duplex (strand-separation) model: only single-stranded molecules are copy-competent, duplexes are decay-protected. Per-base duplex enthalpy/entropy, decay constants and copy rate are PLACEHOLDERS of nearest-neighbour order, not fitted. Consequence tested: the minimum temperature for net growth rises with GC content (about 288, 307, 324 K for 0, 50, 100% GC at 100 nM strands).
- Verified against theory: copy-error statistics (Binomial(L, mu), P(exact)=(1-mu)^L), single-peak quasispecies equilibrium and loss of the master sequence beyond the error threshold, selection coefficient from log-ratio growth, neutral drift symmetry, exact integer monomer-unit accounting, genotype-level ancestry with Hamming-distance checks. Test landscapes that inject rates directly are labelled as such.
- Tau-leaping is approximate; exact Gillespie is used for small populations. Not yet done: coupling replicators to protocell compartments or to the reaction-network chemistry, recombination, indels, phenotype beyond duplex stability.
