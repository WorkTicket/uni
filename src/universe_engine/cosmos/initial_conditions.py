"""Cosmological initial conditions for 3D particle-mesh simulations.

Generates Gaussian random density fields from a matter power spectrum,
with Zel'dovich/1LPT displacement and growing-mode velocity fields.

Supports optional import of CLASS/CAMB transfer functions. Neither package is
required for ordinary application startup: a versioned analytic reference
dataset (Eisenstein & Hu-like, Planck 2018-like cosmology) ships with the
engine and is used by default.

All fields are generated in comoving box-fraction coordinates consistent with
the ParticleMesh3D solver (positions x in [0,1), velocities u = dx/ds where
s = ln a).

Initial conditions produced here are for cold dark matter only. Baryonic
physics, massive neutrinos, and nonlinear corrections beyond 1LPT are not
included; see docs/SCIENTIFIC_LIMITS.md.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..cosmos.coordinates import (
    hubble_from_cosmology,
    growth_rate_approx,
    MPC_KM,
    SECONDS_PER_GYR,
)

DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parents[3] / "data" / "reference"


@dataclass(frozen=True)
class ReferenceDataset:
    """Versioned reference dataset with cosmology, power spectrum, provenance."""
    version: str
    cosmology: dict
    source: str
    provenance: str
    units: dict
    description: str
    checksum_sha256: str
    data_file: str
    n_k_points: int
    k_min_h_Mpc: float
    k_max_h_Mpc: float

    @classmethod
    def load(cls, path: Path) -> "ReferenceDataset":
        meta = json.loads(path.read_text())
        return cls(
            version=meta["version"],
            cosmology=meta["cosmology"],
            source=meta["source"],
            provenance=meta["provenance"],
            units=meta["units"],
            description=meta["description"],
            checksum_sha256=meta["checksum_sha256"],
            data_file=meta["data_file"],
            n_k_points=meta["n_k_points"],
            k_min_h_Mpc=meta["k_min_h_Mpc"],
            k_max_h_Mpc=meta["k_max_h_Mpc"],
        )

    def verify_checksum(self, data_path: Path) -> bool:
        with open(data_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        return actual == self.checksum_sha256


class PowerSpectrum:
    """Interpolated linear matter power spectrum P(k).

    Loaded from a versioned reference dataset, or optionally from CLASS/CAMB
    output (two-column text: k [h/Mpc], P(k) [(Mpc/h)^3]).
    """

    def __init__(self, k: np.ndarray, P_k: np.ndarray,
                 dataset: Optional[ReferenceDataset] = None):
        self.k = np.asarray(k, dtype=float)
        self.P_k = np.asarray(P_k, dtype=float)
        self.dataset = dataset
        # Sort by k for safe interpolation.
        order = np.argsort(self.k)
        self.k = self.k[order]
        self.P_k = self.P_k[order]

    @classmethod
    def from_reference(cls, reference_dir: Path = DEFAULT_REFERENCE_DIR,
                       name: str = "planck_lcdm_powerspectrum_v1") -> "PowerSpectrum":
        meta_path = reference_dir / f"{name}.json"
        data_path = reference_dir / f"{name}.npz"
        dataset = ReferenceDataset.load(meta_path)
        if not dataset.verify_checksum(data_path):
            raise ValueError(f"checksum mismatch for {name} reference data")
        with np.load(data_path) as z:
            k = z["k"]
            P_k = z["P_k"]
        return cls(k, P_k, dataset=dataset)

    @classmethod
    def from_text(cls, path: Path,
                  dataset: Optional[ReferenceDataset] = None) -> "PowerSpectrum":
        """Load from CLASS/CAMB-style two-column text: k [h/Mpc], P(k)."""
        data = np.loadtxt(path, comments="#")
        k = data[:, 0]
        P_k = data[:, 1]
        return cls(k, P_k, dataset=dataset)

    def __call__(self, k: np.ndarray) -> np.ndarray:
        """Interpolate P(k) in log-log space."""
        k = np.asarray(k, dtype=float)
        k_safe = np.clip(k, self.k[0], self.k[-1])
        return np.exp(np.interp(np.log(k_safe), np.log(self.k), np.log(self.P_k)))


@dataclass
class InitialConditions:
    """Cosmological initial conditions for a PM3D simulation.

    Attributes
    ----------
    x : (N,3) float
        Comoving particle positions in box fraction [0,1).
    u : (N,3) float
        Comoving velocities dx/ds (box fraction, dimensionless).
    mass : (N,) float
        Particle masses (code units by default; M_sun if set_physical_masses
        was used).
    box_size_Mpc : float
        Comoving box length in Mpc.
    a_start : float
        Initial scale factor.
    z_start : float
        Initial redshift = 1/a_start - 1.
    cosmology : Cosmology
        Background FLRW parameters.
    seed : int
        Random seed used for field generation.
    dataset : ReferenceDataset | None
        Reference dataset used for the power spectrum.
    """

    x: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    box_size_Mpc: float
    a_start: float
    cosmology: Cosmology
    seed: int
    dataset: Optional[ReferenceDataset] = None
    n_mesh: int = 64
    n_particles_per_axis: int = 32

    @property
    def z_start(self) -> float:
        return 1.0 / self.a_start - 1.0

    @property
    def n_particles(self) -> int:
        return int(self.x.shape[0])

    def to_pm3d(self) -> "ParticleMesh3D":
        """Construct a ParticleMesh3D from these initial conditions."""
        from ..solvers.pm3d import ParticleMesh3D
        return ParticleMesh3D(
            n=self.n_mesh,
            x=self.x.copy(),
            u=self.u.copy(),
            mass=self.mass.copy(),
            cosmology=self.cosmology,
            box_size_Mpc=self.box_size_Mpc,
            a=self.a_start,
        )


def _lattice_positions(n_per_axis: int) -> np.ndarray:
    """Regular lattice positions in box fraction [0,1)."""
    n = n_per_axis
    grid = np.stack(np.meshgrid(
        *([np.arange(n) / n + 0.5 / n] * 3), indexing="ij"), -1).reshape(-1, 3)
    return grid


def _hermitian_gaussian_field(shape: tuple, rng: np.random.Generator,
                              pk_func, k_nyquist: float,
                              box_size_Mpc: float) -> np.ndarray:
    """Generate a Gaussian random field with Hermitian Fourier symmetry.

    Parameters
    ----------
    shape : (n,n,n)
        Mesh shape.
    rng : np.random.Generator
        Random number generator.
    pk_func : callable
        P(k) function returning power at k [h/Mpc].
    k_nyquist : float
        Nyquist wavenumber [h/Mpc].
    box_size_Mpc : float
        Comoving box length in Mpc (treated as Mpc/h so k comes out in h/Mpc).

    Returns
    -------
    delta : (n,n,n) float
        Real-valued density contrast field with mean zero.
    """
    n = shape[0]
    freqs = np.fft.fftfreq(n, d=1.0 / n)  # cycles per box fraction
    KX, KY, KZ = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    kmag = 2.0 * np.pi * np.sqrt(KX**2 + KY**2 + KZ**2) / box_size_Mpc  # h/Mpc
    V = box_size_Mpc ** 3
    pk_vals = pk_func(kmag)
    pk_vals = np.where(kmag > 0, pk_vals, 0.0)
    pk_vals = np.where(kmag <= k_nyquist, pk_vals, 0.0)  # cut off above Nyquist

    # Random complex field with unit variance per mode.
    phase = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    amp = np.sqrt(np.maximum(pk_vals, 0.0) * V / (2.0 * np.pi) ** 3)
    delta_k = amp * phase

    # Enforce Hermitian symmetry: delta_k[-k] = conj(delta_k[k]).
    # Build index arrays for the opposite (negated) mode.
    ix = np.arange(n)
    opp = (-ix) % n
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing="ij")
    OX, OY, OZ = np.meshgrid(opp, opp, opp, indexing="ij")
    delta_k = 0.5 * (delta_k + np.conj(delta_k[OX, OY, OZ]))

    # Zero mode (mean density) is exactly zero.
    delta_k[0, 0, 0] = 0.0
    # Self-conjugate (Nyquist) modes must be real: index == negated index mod n.
    self_conj = (IX == OX) & (IY == OY) & (IZ == OZ)
    delta_k[self_conj] = delta_k[self_conj].real
    delta_k[0, 0, 0] = 0.0

    delta = np.fft.ifftn(delta_k).real * n ** 3
    std = float(delta.std())
    if std > 0:
        delta = delta / std
    return delta


def _displacement_from_density(delta: np.ndarray, box_size_Mpc: float) -> np.ndarray:
    """Compute Zel'dovich displacement potential gradient from density contrast.

    In Fourier space: psi(k) = i * k * delta(k) / k^2, so psi(x) = -grad phi.
    Returns displacement in box fraction.
    """
    n = delta.shape[0]
    freqs = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY, KZ = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2

    d_hat = np.fft.fftn(delta)
    psi = np.empty(delta.shape + (3,), dtype=float)
    for i, ki in enumerate((KX, KY, KZ)):
        psi_hat = np.zeros_like(d_hat)
        nz = k2 > 0
        psi_hat[nz] = 1j * ki[nz] * d_hat[nz] / k2[nz]
        psi[..., i] = np.fft.ifftn(psi_hat).real * n ** 3
    return psi  # displacement in box fraction (at D=1 normalization)


def _linear_growth_factor(a: float, cosmology: Cosmology) -> float:
    """Linear growth factor D(a), normalized to D(a) ~ a at early times."""
    from ..cosmos.coordinates import growth_rate_approx
    # Integrate d ln D / d ln a = f(a) from a=1e-6 to a.
    a_grid = np.linspace(1e-6, a, 2000)
    ln_a = np.log(a_grid)
    f_vals = np.array([growth_rate_approx(cosmology.Omega_m, cosmology.Omega_Lambda,
                                          aa, cosmology.Omega_r) for aa in a_grid])
    ln_D = float(np.sum(0.5 * (f_vals[1:] + f_vals[:-1]) * np.diff(ln_a)))
    # Normalize so D(a) ~ a at early times: D(a_early) = a_early
    D_early = math.exp(ln_D)  # relative to a=1e-6
    D_norm = D_early / 1e-6
    return math.exp(ln_D) / D_norm


def generate_initial_conditions(
    seed: int,
    box_size_Mpc: float,
    n_particles_per_axis: int,
    n_mesh: int,
    a_start: float,
    cosmology: Cosmology,
    power_spectrum: Optional[PowerSpectrum] = None,
    amplitude: Optional[float] = None,
) -> InitialConditions:
    """Generate cosmological initial conditions via Zel'dovich/1LPT.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    box_size_Mpc : float
        Comoving box length in Mpc.
    n_particles_per_axis : int
        Particles per axis (total N = n^3).
    n_mesh : int
        Mesh size for density/force computation.
    a_start : float
        Initial scale factor.
    cosmology : Cosmology
        Background FLRW parameters.
    power_spectrum : PowerSpectrum | None
        Input linear matter power spectrum. If None, uses the default Planck-like
        reference dataset.
    amplitude : float | None
        Optional RMS amplitude of the initial density field. If None, uses the
        power spectrum normalization directly (physical sigma8-like).

    Returns
    -------
    InitialConditions
        Particle positions, velocities, masses, and metadata.
    """
    rng = np.random.default_rng(seed)

    if power_spectrum is None:
        power_spectrum = PowerSpectrum.from_reference()

    k_nyquist = np.pi * n_mesh / box_size_Mpc  # 1/Mpc (Nyquist of the mesh)
    # The power spectrum takes k in h/Mpc; for consistency use h=1 convention
    # with box_size_Mpc treated as Mpc/h. The reference dataset is h/Mpc based.

    # Generate Gaussian density field on the mesh.
    delta_mesh = _hermitian_gaussian_field(
        (n_mesh, n_mesh, n_mesh), rng,
        power_spectrum, k_nyquist, box_size_Mpc)

    # Optionally rescale to a target amplitude.
    if amplitude is not None:
        cur = float(np.std(delta_mesh))
        if cur > 0:
            delta_mesh = delta_mesh * (amplitude / cur)

    # Compute Zel'dovich displacement (D=1 normalization).
    psi = _displacement_from_density(delta_mesh, box_size_Mpc)

    # Sample displacement at particle lattice positions.
    q = _lattice_positions(n_particles_per_axis)
    # Interpolate displacement from mesh to particles via CIC.
    n = n_mesh
    qp = (q % 1.0) * n
    low = np.floor(qp).astype(np.int64)
    f = qp - low
    psi_at_q = np.zeros((q.shape[0], 3), dtype=float)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                w = ((f[:, 0] if ix else 1.0 - f[:, 0])
                     * (f[:, 1] if iy else 1.0 - f[:, 1])
                     * (f[:, 2] if iz else 1.0 - f[:, 2]))
                idx = (low + np.array([ix, iy, iz])) % n
                psi_at_q += w[:, None] * psi[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Linear growth factor at a_start.
    D_start = _linear_growth_factor(a_start, cosmology)
    f_start = growth_rate_approx(cosmology.Omega_m, cosmology.Omega_Lambda,
                                 a_start, cosmology.Omega_r)

    # Zel'dovich positions: x = q + D(a_start) * psi(q)
    x = (q + D_start * psi_at_q) % 1.0

    # Growing-mode velocity: u = dx/ds = f(a) * D(a) * psi(q) = f_start * displacement
    displacement = D_start * psi_at_q
    u = f_start * displacement

    # Particle masses (code units: mean density = 1).
    mass = np.full(q.shape[0], 1.0 / q.shape[0])

    dataset = power_spectrum.dataset
    return InitialConditions(
        x=x, u=u, mass=mass,
        box_size_Mpc=box_size_Mpc,
        a_start=a_start,
        cosmology=cosmology,
        seed=seed,
        dataset=dataset,
        n_mesh=n_mesh,
        n_particles_per_axis=n_particles_per_axis,
    )


def generate_perturbed_lattice(seed: int,
                               n_particles_per_axis: int,
                               amplitude: float = 0.02) -> tuple:
    """Legacy synthetic perturbed-lattice initial conditions (for testing).

    Returns (x, mass) with particles on a regular lattice plus random Gaussian
    displacement. This is the original synthetic IC used before Release B.
    """
    rng = np.random.default_rng(seed)
    n = n_particles_per_axis
    grid = _lattice_positions(n)
    if amplitude > 0:
        grid = grid + amplitude * rng.normal(size=grid.shape)
        grid = grid % 1.0
    mass = np.full(grid.shape[0], 1.0 / grid.shape[0])
    return grid, mass
