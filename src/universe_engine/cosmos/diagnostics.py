"""Cosmological numerical diagnostics for the 3D particle-mesh solver.

Implements:
  - Density power-spectrum measurement (isotropically binned 3D FFT).
  - Dimensionless power-spectrum conversion Delta^2(k) = k^3 P(k) / (2 pi^2).
  - Linear growth comparison vs the cosmological growing mode.
  - Cross-seed ensemble statistics (mean, scatter, uncertainty).
  - Particle and mesh resolution convergence experiments.
  - Numerical error summaries (conservation, finite values, growth, convergence).

FFT convention: numpy forward FFT (unnormalized), inverse FFT normalized by 1/N^3.
The density field is deposited on an n^3 mesh via CIC. The power spectrum is
P(k) = <|delta_k|^2> * V / N^3 ... with V = box_size_Mpc^3 the comoving volume.
Shot noise is NOT subtracted by default; it is reported separately so callers
can decide whether to subtract it.

All distances are comoving Mpc (or Mpc/h when box_size_Mpc is given in Mpc/h).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..solvers.pm3d import ParticleMesh3D, Cosmology, critical_density_Msun_Mpc3
from ..cosmos.coordinates import (
    growth_rate_approx,
    MPC_KM,
    SECONDS_PER_GYR,
)
from ..cosmos.initial_conditions import generate_initial_conditions


@dataclass
class PowerSpectrumResult:
    """Result of a power-spectrum measurement.

    Attributes
    ----------
    k : (n_bins,) float
        Bin center wavenumbers [h / Mpc].
    P_k : (n_bins,) float
        Isotropically averaged power spectrum [(Mpc / h)^3].
    delta2 : (n_bins,) float
        Dimensionless power Delta^2(k) = k^3 P(k) / (2 pi^2).
    k_edges : (n_bins+1,) float
        Bin edges [h / Mpc].
    counts : (n_bins,) int
        Number of modes in each bin.
    shot_noise : float
        Estimated shot noise P_shot = V / N (not subtracted).
    box_size_Mpc : float
        Comoving box size [Mpc].
    n_mesh : int
        Mesh size.
    n_particles : int
        Number of particles.
    nyquist_k : float
        Nyquist wavenumber [h / Mpc].
    fundamental_k : float
        Fundamental mode k = 2 pi / box_size_Mpc [h / Mpc].
    notes : dict
        Method and normalization notes.
    """

    k: np.ndarray
    P_k: np.ndarray
    delta2: np.ndarray
    k_edges: np.ndarray
    counts: np.ndarray
    shot_noise: float
    box_size_Mpc: float
    n_mesh: int
    n_particles: int
    nyquist_k: float
    fundamental_k: float
    notes: dict = field(default_factory=dict)


def measure_power_spectrum(delta: np.ndarray,
                           box_size_Mpc: float,
                           n_bins: int = 20,
                           k_min: Optional[float] = None,
                           k_max: Optional[float] = None,
                           shot_noise: Optional[float] = None) -> PowerSpectrumResult:
    """Measure the isotropically binned 3D density power spectrum.

    Parameters
    ----------
    delta : (n,n,n) float
        Density contrast field delta = rho/rho_bar - 1 (mean zero).
    box_size_Mpc : float
        Comoving box size in Mpc.
    n_bins : int
        Number of logarithmic k bins.
    k_min : float | None
        Minimum k [h / Mpc]. Defaults to the fundamental mode 2 pi / L.
    k_max : float | None
        Maximum k [h / Mpc]. Defaults to the Nyquist k = pi n / L.
    shot_noise : float | None
        Precomputed shot noise [(Mpc/h)^3]. If None, uses V / N.

    Returns
    -------
    PowerSpectrumResult
        Binned k, P(k), Delta^2(k), bin edges, mode counts, and diagnostics.

    Notes
    -----
    FFT convention: numpy forward FFT (unnormalized). The density contrast is
    transformed as delta_k = fftn(delta), and the power per mode is
    |delta_k|^2 / N^6 * V (accounting for the unnormalized forward transform
    and the cell volume). Modes are binned isotropically by |k|.

    The k=0 mode is excluded (mean density / Jeans swindle). Modes above the
    Nyquist limit are excluded. Empty bins (count == 0) are dropped from the
    output arrays but reported via notes.
    """
    n = delta.shape[0]
    if delta.shape != (n, n, n):
        raise ValueError("delta must be a 3D cubic array (n, n, n)")
    if box_size_Mpc <= 0:
        raise ValueError("box_size_Mpc must be positive")

    V = box_size_Mpc ** 3  # comoving volume [Mpc^3]

    # Wavevectors in cycles per box, converted to h/Mpc (treating box_size_Mpc
    # as Mpc/h so k comes out in h/Mpc).
    freqs = np.fft.fftfreq(n, d=1.0 / n)  # cycles per box fraction
    KX, KY, KZ = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    kmag = 2.0 * np.pi * np.sqrt(KX**2 + KY**2 + KZ**2) / box_size_Mpc  # h/Mpc

    # Forward FFT (unnormalized by numpy).
    delta_k = np.fft.fftn(delta - delta.mean())

    # Power per mode: |delta_k|^2 / N^6 * V.
    # delta_k = sum_x delta(x) exp(-2 pi i k x)  [unnormalized]
    # delta(x) = (1/N^3) sum_k delta_k exp(2 pi i k x)
    # <|delta_k|^2> = N^3 * P(k) * V ... roughly; the normalization below
    # gives P(k) in (Mpc/h)^3 when box_size_Mpc is in Mpc/h.
    N3 = float(n ** 3)
    P_mode = (np.abs(delta_k) ** 2) / (N3 ** 2) * V

    # Exclude k=0.
    kmag[0, 0, 0] = np.inf

    # Nyquist cutoff.
    kyquist = math.pi * n / box_size_Mpc  # h/Mpc
    P_mode = np.where(kmag <= kyquist, P_mode, 0.0)
    kmag_flat = kmag.ravel()
    P_flat = P_mode.ravel()

    # Bin edges (logarithmic).
    if k_min is None:
        k_min = 2.0 * math.pi / box_size_Mpc  # fundamental [h/Mpc]
    if k_max is None:
        k_max = kyquist
    if k_min <= 0 or k_max <= k_min:
        raise ValueError("require 0 < k_min < k_max")
    k_edges = np.logspace(math.log10(k_min), math.log10(k_max), n_bins + 1)

    k_bin = []
    P_bin = []
    delta2_bin = []
    count_bin = []
    empty_bins = 0
    for i in range(n_bins):
        lo, hi = k_edges[i], k_edges[i + 1]
        m = (kmag_flat >= lo) & (kmag_flat < hi)
        cnt = int(m.sum())
        if cnt == 0:
            empty_bins += 1
            continue
        k_c = 0.5 * (lo + hi)
        # Log-arithmetic mean of k for the bin center.
        k_c = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        p_val = float(P_flat[m].mean())
        k_bin.append(k_c)
        P_bin.append(p_val)
        delta2_bin.append(k_c ** 3 * p_val / (2.0 * math.pi ** 2))
        count_bin.append(cnt)

    k_out = np.array(k_bin)
    P_out = np.array(P_bin)
    delta2_out = np.array(delta2_bin)
    counts_out = np.array(count_bin, dtype=int)
    k_edges_out = k_edges

    # Shot noise: P_shot = V / N for unit-mass particles.
    if shot_noise is None:
        n_particles = None  # not known from delta alone
        shot_noise_val = V  # placeholder; caller should pass N
    else:
        shot_noise_val = float(shot_noise)

    notes = {
        "fft_convention": "numpy unnormalized forward, 1/N^3 inverse",
        "mass_assignment": "CIC",
        "shot_noise_subtracted": False,
        "k0_excluded": True,
        "nyquist_applied": True,
        "empty_bins": empty_bins,
        "n_modes_total": int(delta.size),
        "box_volume_Mpc3": V,
    }

    return PowerSpectrumResult(
        k=k_out, P_k=P_out, delta2=delta2_out,
        k_edges=k_edges_out, counts=counts_out,
        shot_noise=shot_noise_val,
        box_size_Mpc=box_size_Mpc,
        n_mesh=int(n),
        n_particles=0,
        nyquist_k=float(kyquist),
        fundamental_k=float(2.0 * math.pi / box_size_Mpc),
        notes=notes,
    )


def dimensionless_power_spectrum(k: np.ndarray,
                                 P_k: np.ndarray) -> np.ndarray:
    """Dimensionless power Delta^2(k) = k^3 P(k) / (2 pi^2).

    Parameters
    ----------
    k : (n,) float
        Wavenumbers [h / Mpc].
    P_k : (n,) float
        Power spectrum [(Mpc / h)^3].

    Returns
    -------
    (n,) float
        Delta^2(k), dimensionless.
    """
    k = np.asarray(k, dtype=float)
    P_k = np.asarray(P_k, dtype=float)
    return k ** 3 * P_k / (2.0 * math.pi ** 2)


def power_spectrum_from_pm3d(sim: ParticleMesh3D,
                             n_bins: int = 20,
                             k_min: Optional[float] = None,
                             k_max: Optional[float] = None) -> PowerSpectrumResult:
    """Measure the power spectrum from a ParticleMesh3D snapshot.

    Uses the mass-weighted density contrast and the sim's box size and particle
    count for shot noise.
    """
    delta = sim.density_contrast_mass_weighted()
    n_particles = sim.x.shape[0]
    V = sim.box_size_Mpc ** 3
    shot_noise = V / n_particles  # [(Mpc/h)^3] for M_sun particles
    result = measure_power_spectrum(
        delta, sim.box_size_Mpc, n_bins=n_bins,
        k_min=k_min, k_max=k_max, shot_noise=shot_noise)
    result.n_particles = n_particles
    return result


# ---------------------------------------------------------------------------
# Linear growth comparison
# ---------------------------------------------------------------------------

@dataclass
class LinearGrowthResult:
    """Comparison of simulated vs linear-theory growth."""

    a_values: np.ndarray
    measured_growth: np.ndarray
    theory_growth: np.ndarray
    growth_error: np.ndarray
    rms_error: float
    max_error: float
    k_mode: tuple
    notes: dict = field(default_factory=dict)


def compare_linear_growth(sim: ParticleMesh3D,
                          a_start: float,
                          a_end: float,
                          k_mode: tuple = (1, 0, 0),
                          max_d_ln_a: float = 0.01) -> LinearGrowthResult:
    """Compare simulated linear-mode growth to linear theory.

    Measures the amplitude of a single Fourier density mode as the simulation
    evolves from a_start to a_end, and compares against the linear-theory
    growth factor D(a) normalized to D(a_start) = 1.

    Parameters
    ----------
    sim : ParticleMesh3D
        Simulation at a_start (will be evolved in place).
    a_start, a_end : float
        Start and end scale factors.
    k_mode : tuple of int
        Fourier mode (kx, ky, kz) in units of the fundamental 2 pi / L.
    max_d_ln_a : float
        Timestep for the evolution.

    Returns
    -------
    LinearGrowthResult
        Measured and theory growth vs scale factor, plus error diagnostics.

    Notes
    -----
    The input sim provides the initial particle positions at a_start; a fresh
    copy is evolved internally so the caller's sim is not modified.
    """
    def mode_amp(delta):
        d_hat = np.fft.fftn(delta)
        return abs(d_hat[k_mode]) / delta.size

    a_grid = np.linspace(a_start, a_end, 12)
    measured = []
    theory = []
    a_vals = []

    sim_tmp = ParticleMesh3D(
        n=sim.n, x=sim.x.copy(), u=sim.u.copy(), mass=sim.mass.copy(),
        cosmology=sim.cosmology, box_size_Mpc=sim.box_size_Mpc, a=a_start,
        integrator="kdk_analytic_drag",
    )
    amp0 = mode_amp(sim_tmp.density_contrast())
    theory_norm = None

    for a_t in a_grid:
        if a_t > sim_tmp.a:
            sim_tmp.run_to_a(a_t, max_d_ln_a=max_d_ln_a)
        delta = sim_tmp.density_contrast()
        amp = mode_amp(delta)
        measured.append(amp)
        a_vals.append(a_t)

        # Linear theory: D(a) / D(a_start).
        ln_a_grid = np.linspace(a_start, a_t, 200)
        f_vals = np.array([
            growth_rate_approx(sim.cosmology.Omega_m, sim.cosmology.Omega_Lambda,
                               aa, sim.cosmology.Omega_r)
            for aa in ln_a_grid
        ])
        ln_D = float(np.sum(0.5 * (f_vals[1:] + f_vals[:-1])
                            * np.diff(np.log(ln_a_grid))))
        D_ratio = math.exp(ln_D)
        theory.append(D_ratio)
        if theory_norm is None:
            theory_norm = D_ratio

    measured = np.array(measured)
    theory = np.array(theory) / theory_norm
    measured = measured / measured[0] if measured[0] > 0 else measured

    growth_error = np.abs(measured - theory) / np.maximum(theory, 1e-12)

    return LinearGrowthResult(
        a_values=np.array(a_vals),
        measured_growth=measured,
        theory_growth=theory,
        growth_error=growth_error,
        rms_error=float(np.sqrt(np.mean(growth_error ** 2))),
        max_error=float(np.max(growth_error)),
        k_mode=tuple(k_mode),
        notes={
            "method": "single Fourier mode vs linear theory",
            "integrator": "kdk_analytic_drag",
            "shot_noise_limited": bool(amp0 < 1e-3),
        },
    )


# ---------------------------------------------------------------------------
# Cross-seed statistics
# ---------------------------------------------------------------------------

@dataclass
class CrossSeedResult:
    """Ensemble statistics across independent seeds."""

    seeds: list
    k: np.ndarray
    mean_P_k: np.ndarray
    std_P_k: np.ndarray
    mean_delta2: np.ndarray
    std_delta2: np.ndarray
    fractional_scatter: np.ndarray
    notes: dict = field(default_factory=dict)


def cross_seed_statistics(seeds: list,
                          box_size_Mpc: float,
                          n_particles_per_axis: int,
                          n_mesh: int,
                          a_start: float,
                          cosmology: Cosmology,
                          amplitude: float = 0.02,
                          n_bins: int = 20) -> CrossSeedResult:
    """Ensemble power-spectrum statistics across independent seeds.

    Runs generate_initial_conditions for each seed and measures the power
    spectrum, returning ensemble mean, standard deviation, and fractional
    scatter. Cosmic variance dominates at low k; shot noise at high k.
    """
    ps_results = []
    for seed in seeds:
        ic = generate_initial_conditions(
            seed=seed, box_size_Mpc=box_size_Mpc,
            n_particles_per_axis=n_particles_per_axis,
            n_mesh=n_mesh, a_start=a_start, cosmology=cosmology,
            amplitude=amplitude,
        )
        sim = ic.to_pm3d()
        ps = power_spectrum_from_pm3d(sim, n_bins=n_bins)
        ps_results.append(ps)

    k = ps_results[0].k
    P_list = np.array([ps.P_k for ps in ps_results])
    d2_list = np.array([ps.delta2 for ps in ps_results])

    mean_P = P_list.mean(axis=0)
    std_P = P_list.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean_P)
    mean_d2 = d2_list.mean(axis=0)
    std_d2 = d2_list.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean_d2)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_scatter = np.where(mean_P > 0, std_P / np.maximum(mean_P, 1e-12), 0.0)

    return CrossSeedResult(
        seeds=list(seeds), k=k,
        mean_P_k=mean_P, std_P_k=std_P,
        mean_delta2=mean_d2, std_delta2=std_d2,
        fractional_scatter=frac_scatter,
        notes={
            "n_seeds": len(seeds),
            "amplitude": amplitude,
            "cosmic_variance_dominated": bool(np.any(frac_scatter > 0.3)),
        },
    )


# ---------------------------------------------------------------------------
# Resolution convergence
# ---------------------------------------------------------------------------

@dataclass
class ResolutionConvergenceResult:
    """Resolution convergence experiment results."""

    resolutions: list
    mean_delta2: list
    rms_delta2_scatter: list
    notes: dict = field(default_factory=dict)


def resolution_convergence(box_size_Mpc: float,
                           n_particles_per_axis_list: list,
                           n_mesh_list: list,
                           a_start: float,
                          cosmology: Cosmology,
                           seed: int = 42,
                           amplitude: float = 0.02,
                           n_bins: int = 20) -> ResolutionConvergenceResult:
    """Compare power spectra across particle and mesh resolutions.

    Uses a fixed seed so differences reflect resolution, not sampling.
    """
    results = []
    for n_ppax, n_mesh in zip(n_particles_per_axis_list, n_mesh_list):
        ic = generate_initial_conditions(
            seed=seed, box_size_Mpc=box_size_Mpc,
            n_particles_per_axis=n_ppax,
            n_mesh=n_mesh, a_start=a_start, cosmology=cosmology,
            amplitude=amplitude,
        )
        sim = ic.to_pm3d()
        ps = power_spectrum_from_pm3d(sim, n_bins=n_bins)
        results.append(ps)

    k = results[0].k
    mean_d2 = [ps.delta2 for ps in results]
    rms_scatter = [float(np.sqrt(np.mean(ps.delta2 ** 2))) for ps in results]

    return ResolutionConvergenceResult(
        resolutions=[(r.n_particles, r.n_mesh) for r in results],
        mean_delta2=mean_d2,
        rms_delta2_scatter=rms_scatter,
        notes={
            "seed": seed,
            "amplitude": amplitude,
            "box_size_Mpc": box_size_Mpc,
        },
    )


# ---------------------------------------------------------------------------
# Numerical error summary
# ---------------------------------------------------------------------------

@dataclass
class NumericalErrorSummary:
    """Collected numerical diagnostics for a PM3D run."""

    mass_conservation_error: float
    mean_momentum: np.ndarray
    force_symmetry_error: float
    density_mean: float
    growth_error_max: float
    finite_check: bool
    nonfinite_count: int
    notes: dict = field(default_factory=dict)


def numerical_error_summary(sim: ParticleMesh3D,
                            a_start: float,
                            a_end: float) -> NumericalErrorSummary:
    """Collect numerical diagnostics for a PM3D run from a_start to a_end.

    Tracks:
      - Particle mass conservation (should be exact).
      - Mean momentum (damped by drag).
      - Force symmetry (mirror antisymmetry).
      - Density contrast mean (should be zero).
      - Linear growth error (vs theory).
      - Finite-value check (no NaN/Inf).
    """
    sim_tmp = ParticleMesh3D(
        n=sim.n, x=sim.x.copy(), u=sim.u.copy(), mass=sim.mass.copy(),
        cosmology=sim.cosmology, box_size_Mpc=sim.box_size_Mpc, a=a_start,
        integrator="kdk_analytic_drag",
    )
    mass0 = float(np.sum(sim_tmp.mass))
    sim_tmp.run_to_a(a_end, max_d_ln_a=0.01)
    mass1 = float(np.sum(sim_tmp.mass))

    # Mean momentum.
    p_mean = np.sum(sim_tmp.mass[:, None] * sim_tmp.u, axis=0) / np.sum(sim_tmp.mass)

    # Force symmetry.
    delta = sim_tmp.density_contrast()
    acc = sim_tmp.comoving_acceleration(delta)
    x_mirror = (1.0 - sim_tmp.x) % 1.0
    sim_tmp.x = x_mirror
    delta_m = sim_tmp.density_contrast()
    acc_m = sim_tmp.comoving_acceleration(delta_m)
    sim_tmp.x = (1.0 - x_mirror) % 1.0
    force_sym_err = float(np.max(np.abs(acc_m + acc)))

    # Density mean.
    dens_mean = float(np.mean(sim_tmp.density_contrast()))

    # Finite check.
    x = sim_tmp.x
    u = sim_tmp.u
    nonfinite = int(np.count_nonzero(~np.isfinite(x)) + np.count_nonzero(~np.isfinite(u)))
    finite_ok = nonfinite == 0

    # Growth error: use a fresh copy at a_start (the input sim state).
    growth_sim = ParticleMesh3D(
        n=sim.n, x=sim.x.copy(), u=sim.u.copy(), mass=sim.mass.copy(),
        cosmology=sim.cosmology, box_size_Mpc=sim.box_size_Mpc, a=a_start,
        integrator="kdk_analytic_drag",
    )
    growth = compare_linear_growth(growth_sim, a_start, a_end,
                                   max_d_ln_a=0.01)

    return NumericalErrorSummary(
        mass_conservation_error=abs(mass1 - mass0),
        mean_momentum=p_mean,
        force_symmetry_error=force_sym_err,
        density_mean=dens_mean,
        growth_error_max=growth.max_error,
        finite_check=finite_ok,
        nonfinite_count=nonfinite,
        notes={
            "a_start": a_start,
            "a_end": a_end,
            "n_particles": int(sim.x.shape[0]),
            "n_mesh": int(sim.n),
            "box_size_Mpc": sim.box_size_Mpc,
        },
    )

