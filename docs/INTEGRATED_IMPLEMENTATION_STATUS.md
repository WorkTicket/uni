# Integrated implementation status

- Nonlinear PM matter: working prototype, tested CIC conservation, FFT zero mode, periodicity and repeatability; not converged science grade.
- Structure: thresholded density maxima; no bound-halo identification or physical mass definitions.
- Galaxies: illustrative abundance and gas-to-star proxies dependent on density maxima; not hydrodynamic emergence.
- Star systems: sampled synthetic populations dependent on galaxy proxies; stars not evolving via fusion.
- Planets: sampled radii/orbits and equilibrium temperatures; no gravitational accretion or climate.
- Life and evolution: externally seeded resource-limited evolving scalar traits; no abiogenesis or biological genetics.
- Coupling: gravity -> density peaks -> galaxy proxy -> star sample -> planetary catalog. Evolution is a separate experiment **not** physically coupled to planets.
- Validation: 17 unit/integration tests, not observational validation or grid-convergence verification.

## Model equations and limitations

The comoving coordinates x are dimensionless in the periodic unit box. Set s=ln(a) and u=dx/ds. The solver approximates:

    dx/ds = u
    du/ds = -(2+dlnH/dlna)u - 3/2 Omega_m(a) grad Delta^{-1}(delta)

Density is computed using CIC, and Delta^{-1} is solved by FFT. Integration uses a simple two-kick update with drag approximated at step endpoints. Not symplectic for cosmological drag and not guaranteed accurate at large time steps. Starting density is a synthetic Gaussian field, not a cosmological transfer-function product. It uses one species only; baryonic hydrodynamics and radiation are absent. Galaxy mass units are nominal/proxy scaling not meaningful observational outputs. Stellar and planetary statistics are arbitrary illustrative assumptions. Temperate candidate means only equilibrium temperature screening, not a habitable planet. Evolution is a toy artificial population initialized with founders.

## Acceptance gates for genuine systems

1. Nonlinear halo formation: scale-appropriate power spectrum, convergence tests against analytical growth and controlled collapse, calibrated particle masses, force resolution, bound halo finder, mass functions vs benchmarks.
2. Galaxy formation: baryon fluids, cooling, star-formation and feedback plus convergence and comparison to galaxy scaling relations.
3. Stars: stellar structure and stellar evolution network or validated interpolation from established stellar tracks.
4. Planets: protoplanetary disk dynamics, collisions/accretion, migration, validated against N-body benchmarks.
5. Abiogenesis: explicit chemistry, thermodynamic driving, replication and heredity demonstrably emerging in simulation; no accepted model guarantees it.
6. Biology: cell chemistry/genome/genotype-phenotype mappings and ecology linked to environmental resources with empirically grounded benchmarks.
