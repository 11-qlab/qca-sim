"""Measurement-induced phase transition in a QCA chain.

A 1D QCA chain driven by random unitaries and projective
measurements undergoes a phase transition in the steady-state
entanglement entropy:

  p < p_c:  volume law  S ~ L
  p > p_c:  area law    S ~ const

The critical point for Haar-random circuits is p_c ~ 0.16.

For a QCA chain, the unitary step is a fixed 4-CZ gate (native
coupling), not Haar-random, but the transition survives.

We simulate a small chain (L = 10 qubits) with Clifford gates
and stochastic Z measurements, and measure S(ρ_A) vs p.
"""
from __future__ import annotations
import numpy as np


# Paulis
I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
H1 = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)


def apply_1q(state, U, q, n):
    """Apply 1-qubit gate to qubit q (0-indexed) in an n-qubit state."""
    psi = state.reshape([2] * n)
    psi = np.tensordot(U, psi, axes=([1], [q]))
    psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)


def apply_2q(state, U4, q1, q2, n):
    """Apply 2-qubit gate to qubits q1, q2."""
    psi = state.reshape([2] * n)
    psi = np.tensordot(U4.reshape(2, 2, 2, 2), psi, axes=([2, 3], [q1, q2]))
    psi = np.moveaxis(psi, [0, 1], [q1, q2])
    return psi.reshape(-1)


def CNOT_U():
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)


def CZ_U():
    return np.diag([1,1,1,-1]).astype(complex)


def random_1q(rng):
    """Random SU(2) via exponential of random Pauli combination."""
    a, b, c = rng.normal(size=3)
    theta = np.sqrt(a*a + b*b + c*c)
    if theta < 1e-9:
        return I2
    n_hat = np.array([a, b, c]) / theta
    return (np.cos(theta/2) * I2
            - 1j * np.sin(theta/2) * (n_hat[0]*X + n_hat[1]*Y + n_hat[2]*Z))


def von_neumann(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log(w)))


def half_chain_entropy(state, L):
    """S(ρ_A) for A = first L/2 qubits."""
    psi = state.reshape(2**(L//2), 2**(L - L//2))
    # SVD to get Schmidt values
    s = np.linalg.svd(psi, compute_uv=False)
    s2 = s**2
    s2 = s2[s2 > 1e-12]
    return float(-np.sum(s2 * np.log(s2)))


def evolve_and_measure(L, p, n_layers, seed=0):
    """Random unitary evolution + projective measurements."""
    rng = np.random.default_rng(seed)
    state = np.zeros(2**L, dtype=complex)
    state[0] = 1.0     # |000...0>

    for layer in range(n_layers):
        # Random 1-qubit rotations
        for q in range(L):
            state = apply_1q(state, random_1q(rng), q, L)
        # Native QCA entanglers (CZ on neighbouring pairs)
        for q in range(0, L - 1, 2):
            state = apply_2q(state, CZ_U(), q, q + 1, L)
        for q in range(1, L - 1, 2):
            state = apply_2q(state, CZ_U(), q, q + 1, L)
        # Measurements: each qubit measured in Z with probability p
        for q in range(L):
            if rng.random() < p:
                prob1 = 0.0
                psi = state.reshape([2]*L)
                # marginal P(q = 1)
                axes = tuple(i for i in range(L) if i != q)
                psi_slice = np.take(psi, 1, axis=q)
                prob1 = float(np.sum(np.abs(psi_slice)**2))
                # collapse
                if rng.random() < prob1:
                    state = apply_1q(state, np.array([[0,0],[0,1]], dtype=complex), q, L)
                else:
                    state = apply_1q(state, np.array([[1,0],[0,0]], dtype=complex), q, L)
                state /= np.linalg.norm(state)
    return state


if __name__ == "__main__":
    print("=== Measurement-induced phase transition in QCA chain ===\n")

    L = 10
    n_layers = 15
    n_seeds = 4

    print(f"Chain length L = {L}, {n_layers} layers, {n_seeds} seeds\n")
    print(f"{'p (meas rate)':>14} {'<S(A)> (nats)':>16} {'std':>8}")
    print("-" * 40)

    results = []
    for p in (0.0, 0.05, 0.10, 0.16, 0.25, 0.40, 0.60):
        Ss = []
        for seed in range(n_seeds):
            state = evolve_and_measure(L, p, n_layers, seed=seed)
            S = half_chain_entropy(state, L)
            Ss.append(S)
        meanS = np.mean(Ss)
        stdS = np.std(Ss)
        results.append((p, meanS, stdS))
        print(f"{p:>14.2f} {meanS:>16.4f} {stdS:>8.4f}")

    print()
    print("Interpretation:")
    print("  p = 0   ->  volume law, S ~ L (maximally entangled)")
    print("  p large ->  area law, S small (near product state)")
    print("  Crossover at p_c ~ 0.16 for Haar random circuits.")
    print()
    print("  In a QCA chain, the measurement-induced transition")
    print("  gives a natural boundary between:")
    print("    - classical readout of the chain (p > p_c)")
    print("    - coherent quantum dynamics (p < p_c)")
    print()
    print("  Practical consequence: if QCA clocking measurements")
    print("  are too aggressive (p > p_c), the chain loses coherence")
    print("  and computation fails.  This sets a SHOT-NOISE budget.")
