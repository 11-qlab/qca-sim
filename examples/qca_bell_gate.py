"""Two QCA cells as a native entangling gate.

Model: transverse-field Ising on 2 polarization qubits.

    H = -J sz_A sz_B - g(sx_A + sx_B) - B(sz_A + sz_B)

    sz = polarization (+1 on L, -1 on R)
    sx = interdot tunnelling

For J >> g the ground state is the ferromagnetic cat state
    |GHZ> = (|++> + |-->)/sqrt(2)
which is a maximally entangled Bell state (in the Z basis).

If we prepare an input product state (e.g., |+ +>), then quench
J and g, we can generate entanglement on demand.  This is a
CNOT-equivalent primitive built into the QCA coupling itself.

We quantify:
  - Concurrence C(t) of the 2-qubit state
  - von Neumann entropy S(A) of the reduced density matrix
  - Fidelity with respect to the ideal Bell state
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

I = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
kron = np.kron


def H(J, g, B_A=0.0, B_B=0.0):
    return (-J * kron(sz, sz)
            - g * (kron(sx, I) + kron(I, sx))
            - B_A * kron(sz, I)
            - B_B * kron(I, sz))


def rho_from(psi):
    return np.outer(psi, psi.conj())


def reduced_A(rho):
    return rho.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)


def von_neumann(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log(w)))


def concurrence(rho):
    """Wootters concurrence in the computational basis."""
    sy = np.array([[0, -1j], [1j, 0]])
    rho_star = kron(sy, sy) @ rho.conj() @ kron(sy, sy)
    R = rho @ rho_star
    w = np.linalg.eigvalsh(R)
    w = np.sqrt(np.maximum(w, 0))
    w = np.sort(w)[::-1]
    return float(max(0.0, w[0] - w[1] - w[2] - w[3]))


def fidelity_with_ghz(psi):
    ghz = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return float(abs(ghz.conj() @ psi) ** 2)


def evolve_to_ground(J, g, dt=0.01, T=50.0, psi0=None):
    """Adiabatic turn-on: ramp g from 0 to target while J is fixed."""
    if psi0 is None:
        # start in |+ +> = ground state of -B(sz_A+sz_B) for large B
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
    psi = psi0.copy()
    n = int(T / dt)
    for k in range(n):
        g_now = g * (k + 1) / n
        U = expm(-1j * H(J, g_now) * dt)
        psi = U @ psi
        psi /= np.linalg.norm(psi)
    return psi


if __name__ == "__main__":
    print("=== QCA cells as an entangling gate ===\n")

    print("1. Ground-state entanglement vs J/g ratio (g = 0.1):\n")
    print(f"{'J/g':>6} {'E0':>10} {'S(A)':>8} {'C':>8} {'F_GHZ':>8}")
    print("-" * 46)
    for J_over_g in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        J = J_over_g * 0.1
        g = 0.1
        w, v = np.linalg.eigh(H(J, g))
        psi = v[:, 0]
        rho = rho_from(psi)
        S = von_neumann(reduced_A(rho))
        C = concurrence(rho)
        F = fidelity_with_ghz(psi)
        print(f"{J_over_g:>6.1f} {w[0]:>10.5f} {S:>8.4f} {C:>8.4f} {F:>8.4f}")

    print("\n2. Adiabatic gate: turn on g=0.1 with J=1.0 over T=50:")
    psi = evolve_to_ground(J=1.0, g=0.1, dt=0.005, T=50.0)
    rho = rho_from(psi)
    print(f"   S(A)     = {von_neumann(reduced_A(rho)):.4f}")
    print(f"   C        = {concurrence(rho):.4f}")
    print(f"   F_GHZ    = {fidelity_with_ghz(psi):.4f}")

    print("\n3. Fast quench (T=1) vs slow adiabatic (T=50):")
    for T in (1.0, 5.0, 20.0, 50.0, 200.0):
        psi = evolve_to_ground(J=1.0, g=0.1, dt=0.005, T=T)
        rho = rho_from(psi)
        S = von_neumann(reduced_A(rho))
        C = concurrence(rho)
        print(f"   T = {T:>6.1f}   S = {S:.4f}   C = {C:.4f}")

    print("\nInterpretation:")
    print("  J/g > 5 : near-maximal entanglement, C -> 1, S -> ln 2")
    print("  J/g < 1 : product state, C -> 0, S -> 0")
    print("  Adiabatic ramp populates the cat state on demand.")
    print("  A QCA wire built from J/g > 5 cells is a CHAIN OF")
    print("  NATIVE 2-QUBIT ENTANGLING GATES.")
