"""Quantum state mathematics (Research Program B1).

Implements QuantumState, DensityMatrix, Hamiltonian, QuantumOperator,
QuantumMeasurement, EntangledSystem, and QuantumEnvironment using
complex-valued numerical arrays.

Supports pure and mixed states, tensor products, partial traces, and
expectation values. Tests normalization, Hermiticity, positivity, and trace.

This is a minimal, verified subsystem for small quantum experiments (two-qubit
Bell tests, dephasing, amplitude damping, Lindblad evolution). It is NOT a
global wavefunction for every particle in the universe.

Provenance: standard quantum mechanics (unitary evolution, Born rule,
partial trace). No speculative quantum cognition is claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

TOL = 1e-10


def _is_hermitian(A: np.ndarray, tol: float = TOL) -> bool:
    return A.shape[0] == A.shape[1] and np.allclose(A, A.conj().T, atol=tol, rtol=tol)


def _is_positive(A: np.ndarray, tol: float = TOL) -> bool:
    if not _is_hermitian(A, tol):
        return False
    eig = np.linalg.eigvalsh(A)
    return np.all(eig >= -tol)


def _is_trace_one(A: np.ndarray, tol: float = TOL) -> bool:
    return bool(np.isclose(np.trace(A), 1.0, atol=tol, rtol=tol))


# Pauli matrices (single qubit).
def pauli_x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# QuantumState: pure state vector
# ---------------------------------------------------------------------------

@dataclass
class QuantumState:
    """A pure quantum state vector in a finite-dimensional Hilbert space.

    Attributes
    ----------
    vector : complex array, shape (d,)
        Normalized state vector |psi>.
    dim : int
        Hilbert-space dimension.
    label : str
        Human-readable label for provenance.
    """

    vector: np.ndarray
    dim: int
    label: str = "pure"

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=complex).reshape(-1)
        if self.vector.shape[0] != self.dim:
            raise ValueError(f"vector length {self.vector.shape[0]} != dim {self.dim}")
        if not np.all(np.isfinite(self.vector)):
            raise ValueError("state vector must be finite")
        norm = float(np.linalg.norm(self.vector))
        if not np.isclose(norm, 1.0, atol=TOL, rtol=TOL):
            self.vector = self.vector / norm

    @classmethod
    def basis(cls, dim: int, index: int, label: str = "") -> "QuantumState":
        """Computational basis state |index>."""
        v = np.zeros(dim, dtype=complex)
        v[index] = 1.0 + 0j
        return cls(v, dim, label or f"|{index}>")

    @classmethod
    def plus(cls, dim: int = 2, label: str = "|+>") -> "QuantumState":
        return cls(np.ones(dim, dtype=complex) / np.sqrt(dim), dim, label)

    @classmethod
    def minus(cls, dim: int = 2, label: str = "|->") -> "QuantumState":
        v = np.ones(dim, dtype=complex) / np.sqrt(dim)
        v[1] *= -1.0
        return cls(v, dim, label)

    def dagger(self) -> np.ndarray:
        return self.vector.conj().T

    def density_matrix(self) -> "DensityMatrix":
        """Convert pure state to density matrix |psi><psi|."""
        rho = np.outer(self.vector, self.vector.conj())
        return DensityMatrix(rho, self.dim, f"{self.label} (pure)")

    def overlap(self, other: "QuantumState") -> complex:
        return complex(np.vdot(self.vector, other.vector))

    def probabilities(self) -> np.ndarray:
        """Born-rule measurement probabilities in the computational basis."""
        return np.abs(self.vector) ** 2

    def to_dict(self) -> dict:
        return {
            "schema": "quantum-state/v1",
            "type": "pure",
            "dim": self.dim,
            "vector": self.vector.tolist(),
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# DensityMatrix: mixed state
# ---------------------------------------------------------------------------

@dataclass
class DensityMatrix:
    """A density matrix (pure or mixed quantum state).

    Attributes
    ----------
    matrix : complex array, shape (d, d)
        Hermitian, positive, trace-one operator rho.
    dim : int
        Hilbert-space dimension.
    label : str
        Human-readable label.
    """

    matrix: np.ndarray
    dim: int
    label: str = "mixed"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=complex)
        if self.matrix.shape != (self.dim, self.dim):
            raise ValueError(f"matrix shape {self.matrix.shape} != ({self.dim}, {self.dim})")
        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("density matrix must be finite")
        if not _is_hermitian(self.matrix):
            raise ValueError("density matrix must be Hermitian")
        if not _is_positive(self.matrix):
            raise ValueError("density matrix must be positive semidefinite")
        if not _is_trace_one(self.matrix):
            # Renormalize trace rather than silently accepting invalid state.
            self.matrix = self.matrix / np.trace(self.matrix)

    @classmethod
    def from_state(cls, state: QuantumState) -> "DensityMatrix":
        return state.density_matrix()

    @classmethod
    def maximally_mixed(cls, dim: int, label: str = "I/d") -> "DensityMatrix":
        return cls(np.eye(dim, dtype=complex) / dim, dim, label)

    def is_pure(self, tol: float = TOL) -> bool:
        return bool(np.isclose(np.trace(self.matrix @ self.matrix), 1.0, atol=tol, rtol=tol))

    def purity(self) -> float:
        return float(np.real(np.trace(self.matrix @ self.matrix)))

    def von_neumann_entropy(self, base: float = 2.0) -> float:
        """Von Neumann entropy S = -Tr(rho log rho)."""
        eig = np.linalg.eigvalsh(self.matrix)
        eig = eig[eig > 0]
        s = -float(np.sum(eig * np.log(eig)))
        return s / np.log(base)

    def partial_trace(self, keep_dims: Tuple[int, ...],
                      keep_subsystems: Tuple[int, ...]) -> "DensityMatrix":
        """Partial trace over all subsystems not in keep_subsystems.

        The full Hilbert space is a tensor product of subsystems with
        dimensions given by keep_dims. keep_subsystems indexes which
        subsystems to retain.

        Method: reshape rho into a rank-2*n_sub tensor with axes ordered
        [ket_0, ket_1, ..., ket_{n-1}, bra_0, bra_1, ..., bra_{n-1}].
        Transpose to interleave [ket_0, bra_0, ket_1, bra_1, ...], then
        use einsum to contract traced-out subsystems (ket_i == bra_i).
        """
        n_sub = len(keep_dims)
        if len(keep_subsystems) == 0 or len(keep_subsystems) > n_sub:
            raise ValueError("invalid keep_subsystems")
        # Reshape rho into [ket_0,...,ket_{n-1}, bra_0,...,bra_{n-1}].
        rho = self.matrix.reshape(tuple(keep_dims) + tuple(keep_dims))
        # Transpose to [ket_0, bra_0, ket_1, bra_1, ...].
        order = []
        for i in range(n_sub):
            order.append(i)          # ket_i
            order.append(n_sub + i)  # bra_i
        rho = np.transpose(rho, axes=order)

        # Build einsum subscripts.
        # Each kept subsystem contributes two output axes (ket, bra) with distinct chars.
        # Each traced-out subsystem has ket==bra (same char -> contraction).
        in_chars = []
        out_chars = []
        char_idx = 0
        for i in range(n_sub):
            ket_c = chr(97 + char_idx); char_idx += 1
            bra_c = chr(97 + char_idx); char_idx += 1
            in_chars.append(ket_c)
            in_chars.append(bra_c)
            if i in keep_subsystems:
                out_chars.append(ket_c)
                out_chars.append(bra_c)
            else:
                # Trace out: bra char == ket char (contraction).
                in_chars[-1] = in_chars[-2]
        subscript = "".join(in_chars) + "->" + "".join(out_chars)
        reduced = np.einsum(subscript, rho)
        kept_dim = int(np.prod([keep_dims[i] for i in keep_subsystems]))
        return DensityMatrix(reduced.reshape(kept_dim, kept_dim),
                             kept_dim, f"Tr_out({self.label})")

    def expectation(self, operator: np.ndarray) -> complex:
        """Expectation value <O> = Tr(rho O)."""
        op = np.asarray(operator, dtype=complex)
        return complex(np.trace(self.matrix @ op))

    def to_dict(self) -> dict:
        return {
            "schema": "density-matrix/v1",
            "type": "mixed",
            "dim": self.dim,
            "matrix": self.matrix.tolist(),
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# QuantumOperator
# ---------------------------------------------------------------------------

@dataclass
class QuantumOperator:
    """A linear operator on a Hilbert space.

    Attributes
    ----------
    matrix : complex array, shape (d, d)
    dim : int
    is_hermitian : bool
        Declared; verified on construction if check=True.
    label : str
    """

    matrix: np.ndarray
    dim: int
    is_hermitian: bool = False
    label: str = "operator"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=complex)
        if self.matrix.shape != (self.dim, self.dim):
            raise ValueError(f"operator shape {self.matrix.shape} != ({self.dim}, {self.dim})")
        if self.is_hermitian and not _is_hermitian(self.matrix):
            raise ValueError("operator declared Hermitian but matrix is not")

    def dagger(self) -> "QuantumOperator":
        return QuantumOperator(self.matrix.conj().T, self.dim,
                               is_hermitian=self.is_hermitian, label=self.label + "†")

    def apply(self, state: QuantumState) -> QuantumState:
        """Apply operator to a state vector (not necessarily normalized)."""
        v = self.matrix @ state.vector
        return QuantumState(v, self.dim, self.label + " applied")

    def expm(self, t: float) -> "QuantumOperator":
        """Matrix exponential exp(-i H t) for unitary evolution."""
        from scipy.linalg import expm
        U = expm(-1j * self.matrix * t)
        return QuantumOperator(U, self.dim, is_hermitian=False, label=self.label + f"U(t={t})")

    def to_dict(self) -> dict:
        return {
            "schema": "quantum-operator/v1",
            "dim": self.dim,
            "matrix": self.matrix.tolist(),
            "is_hermitian": self.is_hermitian,
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

@dataclass
class Hamiltonian(QuantumOperator):
    """Hermitian operator generating time evolution via Schrödinger equation.

    i d|psi>/dt = H |psi>  (hbar = 1)
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _is_hermitian(self.matrix):
            raise ValueError("Hamiltonian must be Hermitian")
        self.is_hermitian = True

    def time_evolution(self, t: float) -> QuantumOperator:
        """Unitary time-evolution operator U(t) = exp(-i H t)."""
        return self.expm(t)

    def evolve_state(self, state: QuantumState, t: float) -> QuantumState:
        """Evolve a pure state by time t under this Hamiltonian."""
        U = self.time_evolution(t)
        v = U.matrix @ state.vector
        return QuantumState(v, self.dim, f"{state.label} evolved t={t}")


# ---------------------------------------------------------------------------
# QuantumMeasurement
# ---------------------------------------------------------------------------

@dataclass
class QuantumMeasurement:
    """A projective measurement in a given basis.

    Attributes
    ----------
    projectors : list of (d, d) complex arrays
        Orthogonal projectors P_i with sum_i P_i = I.
    outcomes : list of labels
    dim : int
    """

    projectors: List[np.ndarray]
    outcomes: List[str]
    dim: int

    def __post_init__(self) -> None:
        if len(self.projectors) != len(self.outcomes):
            raise ValueError("n projectors must equal n outcomes")
        for P in self.projectors:
            P = np.asarray(P, dtype=complex)
            if P.shape != (self.dim, self.dim):
                raise ValueError("projector shape mismatch")
        completeness = sum(P.conj().T @ P for P in self.projectors)
        if not np.allclose(completeness, np.eye(self.dim), atol=TOL, rtol=TOL):
            raise ValueError("projectors must satisfy sum P_i = I")

    def probabilities(self, state: DensityMatrix) -> np.ndarray:
        """Born-rule probabilities p_i = Tr(rho P_i)."""
        return np.array([float(np.real(state.expectation(P))) for P in self.projectors])

    def measure(self, state: DensityMatrix, rng: Optional[np.random.Generator] = None) -> Tuple[str, DensityMatrix]:
        """Sample one measurement outcome and return (outcome, post-measurement state).

        If rng is None, returns probabilities without sampling.
        """
        probs = self.probabilities(state)
        if rng is None:
            return self.outcomes[int(np.argmax(probs))], state
        rng = rng or np.random.default_rng()
        outcome_idx = int(rng.choice(len(self.outcomes), p=probs))
        P = self.projectors[outcome_idx]
        # Lüders rule: rho' = P rho P / Tr(rho P)
        rho_post = P @ state.matrix @ P
        p = float(np.real(np.trace(rho_post)))
        if p > 0:
            rho_post = rho_post / p
        else:
            rho_post = state.matrix.copy()
        return self.outcomes[outcome_idx], DensityMatrix(rho_post, self.dim,
                                                          f"post-{self.outcomes[outcome_idx]}")

    def to_dict(self) -> dict:
        return {
            "schema": "quantum-measurement/v1",
            "dim": self.dim,
            "outcomes": self.outcomes,
            "projectors": [P.tolist() for P in self.projectors],
        }


# ---------------------------------------------------------------------------
# EntangledSystem: bipartite entangled state
# ---------------------------------------------------------------------------

@dataclass
class EntangledSystem:
    """A bipartite quantum system AB with dimensions dA, dB.

    Provides Bell states, entanglement entropy, and correlation measurements.
    """

    dim_a: int
    dim_b: int
    state: DensityMatrix
    label: str = "bipartite"

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=complex)
        if self.state.shape != (self.dim_a * self.dim_b, self.dim_a * self.dim_b):
            raise ValueError("state dimension mismatch")
        self.density = DensityMatrix(self.state, self.dim_a * self.dim_b, self.label)

    @classmethod
    def bell_state(cls, kind: str = "phi_plus") -> "EntangledSystem":
        """One of the four Bell states on two qubits."""
        d = 4
        rho = np.zeros((d, d), dtype=complex)
        if kind == "phi_plus":
            # (|00> + |11>) / sqrt(2)
            v = np.zeros(d, dtype=complex)
            v[0] = 1 / np.sqrt(2)
            v[3] = 1 / np.sqrt(2)
        elif kind == "phi_minus":
            v = np.zeros(d, dtype=complex)
            v[0] = 1 / np.sqrt(2)
            v[3] = -1 / np.sqrt(2)
        elif kind == "psi_plus":
            v = np.zeros(d, dtype=complex)
            v[1] = 1 / np.sqrt(2)
            v[2] = 1 / np.sqrt(2)
        elif kind == "psi_minus":
            v = np.zeros(d, dtype=complex)
            v[1] = 1 / np.sqrt(2)
            v[2] = -1 / np.sqrt(2)
        else:
            raise ValueError(f"unknown Bell state {kind!r}")
        rho = np.outer(v, v.conj())
        return cls(2, 2, rho, f"Bell {kind}")

    def reduced_density_a(self) -> DensityMatrix:
        """Partial trace over subsystem B."""
        return self.density.partial_trace((self.dim_a, self.dim_b), (0,))

    def reduced_density_b(self) -> DensityMatrix:
        """Partial trace over subsystem A."""
        return self.density.partial_trace((self.dim_a, self.dim_b), (1,))

    def entanglement_entropy(self, base: float = 2.0) -> float:
        """Von Neumann entropy of either reduced state (S_A = S_B)."""
        rho_a = self.reduced_density_a()
        return rho_a.von_neumann_entropy(base=base)

    def concurrence(self) -> float:
        """Wootters concurrence for a two-qubit state."""
        if self.dim_a != 2 or self.dim_b != 2:
            raise ValueError("concurrence defined for two qubits only")
        rho = self.density.matrix
        # spin-flip: rho_tilde = (sigma_y x sigma_y) rho* (sigma_y x sigma_y)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sy_sy = np.kron(sy, sy)
        rho_tilde = sy_sy @ rho.conj() @ sy_sy
        R = rho @ rho_tilde
        eig = np.linalg.eigvals(R)
        # R has real non-negative eigenvalues. Take real parts, sort descending,
        # clamp negatives to zero (numerical noise), then square-root.
        vals = np.sort(np.real(eig))[::-1]
        vals = np.maximum(vals, 0.0)
        sqrt_eig = np.sqrt(vals)
        c = float(sqrt_eig[0] - sqrt_eig[1] - sqrt_eig[2] - sqrt_eig[3])
        return max(0.0, c)

    def to_dict(self) -> dict:
        return {
            "schema": "entangled-system/v1",
            "dim_a": self.dim_a,
            "dim_b": self.dim_b,
            "label": self.label,
            "entanglement_entropy": self.entanglement_entropy(),
            "concurrence": self.concurrence() if self.dim_a == 2 and self.dim_b == 2 else None,
        }


# ---------------------------------------------------------------------------
# QuantumEnvironment: open-system dynamics
# ---------------------------------------------------------------------------

@dataclass
class QuantumEnvironment:
    """Open quantum system with Hamiltonian H and Lindblad operators L_k.

    Evolution: d rho/dt = -i[H, rho] + sum_k (L_k rho L_k† - 1/2 {L_k† L_k, rho})
    """

    hamiltonian: Hamiltonian
    lindblad_ops: List[np.ndarray] = field(default_factory=list)

    def add_lindblad(self, L: np.ndarray) -> None:
        L = np.asarray(L, dtype=complex)
        if L.shape != (self.hamiltonian.dim, self.hamiltonian.dim):
            raise ValueError("Lindblad operator dimension mismatch")
        self.lindblad_ops.append(L)

    def liouvillian(self, rho: np.ndarray) -> np.ndarray:
        """Compute d rho/dt = L(rho)."""
        H = self.hamiltonian.matrix
        drho = -1j * (H @ rho - rho @ H)
        for L in self.lindblad_ops:
            Ld = L.conj().T
            LdL = Ld @ L
            drho += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
        return drho

    def evolve(self, rho0: np.ndarray, t: float,
               n_steps: int = 100) -> np.ndarray:
        """Evolve density matrix from t=0 to t using a simple integrator.

        For stiff/dissipative dynamics use scipy.integrate.solve_ivp with BDF.
        """
        from scipy.integrate import solve_ivp

        def rhs(t_, y_flat):
            y = y_flat.reshape((self.hamiltonian.dim, self.hamiltonian.dim))
            return self.liouvillian(y).reshape(-1)

        sol = solve_ivp(rhs, (0.0, t), rho0.reshape(-1),
                        method="BDF", dense_output=False,
                        rtol=1e-8, atol=1e-10,
                        max_step=t / max(n_steps, 1))
        if not sol.success:
            raise RuntimeError(f"open-system integration failed: {sol.message}")
        rho = sol.y[:, -1].reshape((self.hamiltonian.dim, self.hamiltonian.dim))
        # Re-hermitianize and renormalize.
        rho = (rho + rho.conj().T) / 2.0
        rho = rho / np.trace(rho)
        return rho

    def to_dict(self) -> dict:
        return {
            "schema": "quantum-environment/v1",
            "dim": self.hamiltonian.dim,
            "n_lindblad": len(self.lindblad_ops),
            "hamiltonian": self.hamiltonian.to_dict(),
        }
