"""Tests for the quantum reality subsystem (Research Program B).

Tests B1 (state mathematics), B2 (dynamics), B3 (CHSH/Bell).
"""

import math
import numpy as np
import pytest

from universe_engine.research.quantum import (
    QuantumState, DensityMatrix, QuantumOperator, Hamiltonian,
    QuantumMeasurement, EntangledSystem, QuantumEnvironment,
    CHSHExperiment, verify_chsh_limits,
    chsh_s, classical_chsh_bound, quantum_chsh_max,
    optimal_chsh_settings, measurement_in_basis,
    pauli_z,
)


# ---------------------------------------------------------------------------
# B1: Quantum state mathematics
# ---------------------------------------------------------------------------

def test_quantum_state_normalization():
    """A pure state vector must be normalized."""
    psi = QuantumState(np.array([1.0, 1.0], dtype=complex) / math.sqrt(2), 2)
    assert np.isclose(np.linalg.norm(psi.vector), 1.0)
    # Unnormalized input is renormalized.
    psi2 = QuantumState(np.array([3.0, 4.0], dtype=complex), 2)
    assert np.isclose(np.linalg.norm(psi2.vector), 1.0)


def test_density_matrix_properties():
    """Density matrix must be Hermitian, positive, trace one."""
    psi = QuantumState.basis(2, 0)
    rho = psi.density_matrix()
    assert rho.is_pure()
    assert np.isclose(rho.purity(), 1.0)
    assert np.isclose(np.trace(rho.matrix), 1.0)
    assert rho.von_neumann_entropy() < 1e-10


def test_maximally_mixed():
    """Maximally mixed state: I/d, purity = 1/d, entropy = log2(d)."""
    rho = DensityMatrix.maximally_mixed(2)
    assert np.isclose(rho.purity(), 0.5)
    assert np.isclose(rho.von_neumann_entropy(base=2.0), 1.0)


def test_tensor_product_and_partial_trace():
    """Tensor product state then partial trace recovers the subsystem."""
    psi_a = QuantumState.basis(2, 0)
    psi_b = QuantumState.basis(2, 1)
    # Product state |0>|1>
    v = np.kron(psi_a.vector, psi_b.vector)
    psi_ab = QuantumState(v, 4)
    rho_ab = psi_ab.density_matrix()
    # Trace out B -> should get |0><0|
    rho_a = rho_ab.partial_trace((2, 2), (0,))
    assert rho_a.is_pure()
    assert np.allclose(rho_a.matrix, np.array([[1, 0], [0, 0]], dtype=complex))


def test_hermiticity_positivity_trace():
    """Basic properties: random density matrix is Hermitian, positive, trace one."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    rho_mat = A @ A.conj().T
    rho_mat = rho_mat / np.trace(rho_mat)
    rho = DensityMatrix(rho_mat, 3)
    assert np.allclose(rho.matrix, rho.matrix.conj().T)
    assert np.all(np.linalg.eigvalsh(rho.matrix) >= -1e-10)
    assert np.isclose(np.trace(rho.matrix), 1.0)


# ---------------------------------------------------------------------------
# B2: Quantum dynamics
# ---------------------------------------------------------------------------

def test_unitary_evolution_preserves_norm():
    """Schrödinger evolution under a Hermitian H preserves state norm."""
    H = Hamiltonian(np.array([[1, 0], [0, -1]], dtype=complex), 2, label="sigma_z")
    psi = QuantumState.plus(2, "|+>")
    psi_t = H.evolve_state(psi, 0.5)
    assert np.isclose(np.linalg.norm(psi_t.vector), 1.0)


def test_schroedinger_phase():
    """Energy eigenstate acquires only a phase under evolution."""
    H = Hamiltonian(np.diag([1.0, -1.0]), 2, label="sigma_z")
    psi = QuantumState.basis(2, 0)  # |0>, energy +1
    psi_t = H.evolve_state(psi, math.pi)
    expected = np.exp(-1j * math.pi)
    assert np.isclose(psi_t.vector[0], expected, atol=1e-10)
    assert np.isclose(psi_t.vector[1], 0.0, atol=1e-10)


def test_lindblad_dephasing():
    """Dephasing damps off-diagonal elements, populations unchanged."""
    from universe_engine.research.quantum.states import pauli_z
    H = Hamiltonian(np.zeros((2, 2), dtype=complex), 2)
    env = QuantumEnvironment(H)
    L = np.sqrt(0.5) * pauli_z()
    env.add_lindblad(L)
    rho0 = QuantumState.plus(2).density_matrix().matrix.copy()
    rho_t = env.evolve(rho0, 1.0, n_steps=50)
    # Off-diagonal should be damped relative to initial.
    assert abs(rho_t[0, 1]) < abs(rho0[0, 1])
    # Populations unchanged under pure dephasing.
    assert np.isclose(rho_t[0, 0].real, 0.5)
    assert np.isclose(rho_t[1, 1].real, 0.5)


# ---------------------------------------------------------------------------
# B3: Entanglement and CHSH
# ---------------------------------------------------------------------------

def test_bell_state_entanglement():
    """A Bell state has entanglement entropy 1 (ebit) and concurrence 1."""
    bell = EntangledSystem.bell_state("phi_plus")
    assert np.isclose(bell.entanglement_entropy(base=2.0), 1.0, atol=1e-8)
    assert np.isclose(bell.concurrence(), 1.0, atol=1e-8)


def test_classical_chsh_bound():
    """Classical local hidden variable bound is 2."""
    assert classical_chsh_bound() == 2.0


def test_quantum_chsh_max():
    """Tsirelson bound is 2*sqrt(2)."""
    assert np.isclose(quantum_chsh_max(), 2.0 * math.sqrt(2))


def test_chsh_optimal_settings_produce_violation():
    """Optimal settings on phi+ should give S close to 2*sqrt(2)."""
    exp = CHSHExperiment(bell_state="phi_plus", n_shots=50000, seed=11)
    result = exp.run(settings=optimal_chsh_settings(), n_shots=50000)
    assert result.violates_classical
    assert abs(result.s_value - quantum_chsh_max()) < 0.1
    assert result.no_signalling_alice
    assert result.no_signalling_bob


def test_classical_chsh_no_violation():
    """Classical local model must not violate the CHSH bound."""
    exp = CHSHExperiment(bell_state="phi_plus", n_shots=30000, seed=11)
    result = exp.run_classical_baseline(n_shots=30000)
    assert not result.violates_classical
    assert abs(result.s_value) <= classical_chsh_bound() + 0.1


def test_verify_chsh_limits():
    """The verification helper should report both quantum and classical pass."""
    out = verify_chsh_limits(n_shots=30000, seed=13)
    assert out["quantum_passes"]
    assert out["classical_passes"]
    assert out["all_pass"]


def test_different_bell_states_all_entangled():
    """All four Bell states are maximally entangled."""
    for kind in ("phi_plus", "phi_minus", "psi_plus", "psi_minus"):
        bell = EntangledSystem.bell_state(kind)
        assert np.isclose(bell.entanglement_entropy(base=2.0), 1.0, atol=1e-8)
        assert np.isclose(bell.concurrence(), 1.0, atol=1e-8)


def test_measurement_in_basis_projectors_complete():
    """Projectors for a basis measurement sum to identity."""
    M = measurement_in_basis(0.3)
    completeness = sum(P.conj().T @ P for P in M.projectors)
    assert np.allclose(completeness, np.eye(2), atol=1e-10)
