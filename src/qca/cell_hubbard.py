"""QCA cell exact diagonalization (Lent/Tougaw model).

Geometry: 4 quantum dots at the corners of a square,
2 excess electrons, inter-dot tunnelling t, on-site U and
inter-site Coulomb V.  A "driver" bias simulates the
neighbouring driver cell.

Site numbering:
    0 --- 1
    |     |
    3 --- 2       ->  0=TL, 1=TR, 2=BR, 3=BL

Polarization:   P = (n0 + n2 - n1 - n3) / 2    in [-1, +1]

Sign convention: a POSITIVE driver bias lowers the energy of
the P = +1 state (electrons on 0 and 2).  This matches the
standard Lent/Tougaw convention.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

ADJ  = [(0, 1), (1, 2), (2, 3), (3, 0)]   # tunnelling edges
DIAG = [(0, 2), (1, 3)]                   # diagonal Coulomb pairs


def basis():
    """All 2-electron occupations of 4 dots (spinless)."""
    return list(combinations(range(4), 2))


def hamiltonian(t: float = 0.5,
                V_diag: float = 1.0,
                V_adj: float = 0.2,
                driver: float = 0.0,
                onsite: float = 0.0):
    """Return (H, basis_list) for the 2-electron QCA cell."""
    B = basis()
    idx = {s: i for i, s in enumerate(B)}
    n = len(B)
    H = np.zeros((n, n), dtype=float)

    coul = {}
    for p in DIAG:
        coul[tuple(sorted(p))] = V_diag
    for p in ADJ:
        coul[tuple(sorted(p))] = V_adj

    # diagonal: Coulomb + driver bias
    # NOTE the leading minus sign: +driver favours P = +1 (electrons on 0,2)
    for i, (a, b) in enumerate(B):
        e = 2.0 * onsite + coul.get((a, b), 0.0)
        e -= driver * ((a in (0, 2)) + (b in (0, 2))
                       - (a in (1, 3)) - (b in (1, 3))) / 2.0
        H[i, i] = e

    # off-diagonal: electron hopping along adjacent edges
    edges = []
    for (p, q) in ADJ:
        edges.append((p, q))
        edges.append((q, p))

    for i, (a, b) in enumerate(B):
        for (x, y) in edges:
            if x in (a, b) and y not in (a, b):
                new = tuple(sorted([s for s in (a, b) if s != x] + [y]))
                j = idx[new]
                H[i, j] += -t

    return H, B


def ground_state(H):
    w, v = np.linalg.eigh(H)
    return float(w[0]), v[:, 0]


def polarization(psi, B):
    n = np.zeros(4)
    for c, (a, b) in zip(np.abs(psi) ** 2, B):
        n[a] += c
        n[b] += c
    return float((n[0] + n[2] - n[1] - n[3]) / 2.0)
