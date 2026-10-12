"""Entanglement between QCA cells.

Question: is a QCA wire a *classical* channel (bit in, bit out)
or a *quantum* channel (entangled throughout)?  If S > 0 between
neighboring cells, the wire is quantum, and QCA becomes a native
2-qubit gate platform.

Model: 2 coupled cells, each a 2-state polarization qubit
    |+> = P = +1     |-> = P = -1
with coupling J (ferromagnetic).  Compute reduced density
matrix of cell A and its von Neumann entropy.
"""
from __future__ import annotations
import numpy as np


def H_two_cell(J=0.1, B=0.0):
    """2x2 per cell.  Basis: ++, +-, -+, --."""
    H = np.zeros((4, 4))
    # diagonal: -J * P_A * P_B  (ferromagnetic if J>0)
    P = [+1, +1, -1, -1]     # A values
    Q = [+1, -1, +1, -1]     # B values
    for i in range(4):
        H[i, i] = -J * P[i] * Q[i] - B * (P[i] + Q[i])
    # transverse field mixes |+> and |-> within each cell
    g = 0.05
    for i in range(4):
        for j in range(4):
            pi, qi = P[i], Q[i]
            pj, qj = P[j], Q[j]
            # flip A only
            if pi != pj and qi == qj:
                H[i, j] += -g
            # flip B only
            if qi != qj and pi == pj:
                H[i, j] += -g
    return H


def reduced_rho_A(psi):
    """Trace out cell B from a 4-dim wavefunction."""
    psi = psi.reshape(2, 2)      # (A, B)
    return psi @ psi.conj().T


def von_neumann(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log(w)))


if __name__ == "__main__":
    print("=== Entanglement in a 2-cell QCA system ===\n")
    print(f"{'J':>8} {'E0':>12} {'S(A)':>10} {'P_A':>8} {'P_B':>8}")
    print("-" * 50)
    for J in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        H = H_two_cell(J=J)
        w, v = np.linalg.eigh(H)
        psi = v[:, 0]
        rhoA = reduced_rho_A(psi)
        S = von_neumann(rhoA)
        # Polarization of A: <P_A> = |a_+|^2 - |a_-|^2 in reduced basis
        pA = float(np.real(rhoA[0, 0] - rhoA[1, 1]))
        # For B, use symmetry: same by construction here
        pB = pA
        print(f"{J:>8.3f} {w[0]:>12.5f} {S:>10.5f} {pA:>+8.4f} {pB:>+8.4f}")

    print()
    print("Interpretation:")
    print("  S = 0  : cells are classically correlated (product state)")
    print("  S > 0  : cells are quantum-entangled; the wire is a")
    print("           coherent quantum channel, not a classical one.")
    print("  Max S = ln(2) ~ 0.693 for a maximally entangled pair.")
