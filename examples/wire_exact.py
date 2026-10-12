"""Exact many-body QCA wire with proper electrostatics.

Physical model (this is how real QCA actually couples):

  - 3 cells, each with 4 dots at the corners of a rectangle.
  - 2 electrons per cell (enforced by construction: C(4,2) = 6
    configurations per cell -> 6^3 = 216 total basis states).
  - Intra-cell nearest-neighbour tunnelling t.
  - Coulomb repulsion between ALL pairs of occupied dots,
    computed from real 2D geometry: V_ij = 1 / (r_ij + a).
  - NO inter-cell tunnelling (cells are isolated by barriers).
  - Driver bias on cell 0, pinning its polarization.

Geometry per cell (0=TL, 1=TR, 2=BR, 3=BL):

    (0,1) ---- (0.9,1)
      |           |
    (0,0) ---- (0.9,0)

Next cell starts at x = 1.0, so there is a 0.1 gap between cells.
Cell 0's BR dot is 0.1 away from cell 1's BL dot -- this is what
drives ferromagnetic alignment: if cells have the SAME polarization,
the two closest electrons across the boundary are ~1.0 apart
(vertically displaced).  If they had OPPOSITE polarization, they
would be 0.1 apart -- huge Coulomb penalty.
"""
from itertools import combinations
import numpy as np

N_CELLS = 3
DX, DY, CELL_W = 0.9, 1.0, 1.0
SOFT = 0.4  # Coulomb softening


def dot_xy(site):
    cell, dot = divmod(site, 4)
    x0 = cell * CELL_W
    return {
        0: (x0, DY),
        1: (x0 + DX, DY),
        2: (x0 + DX, 0.0),
        3: (x0, 0.0),
    }[dot]


def coulomb(a, b):
    xa, ya = dot_xy(a)
    xb, yb = dot_xy(b)
    return 1.0 / (np.hypot(xa - xb, ya - yb) + SOFT)


V_CACHE = {(i, j): coulomb(i, j) for i, j in combinations(range(12), 2)}


def build_basis():
    """Every state has exactly 2 electrons in each cell."""
    per_cell = list(combinations(range(4), 2))
    states = []
    for s0 in per_cell:
        for s1 in per_cell:
            for s2 in per_cell:
                occ = tuple(sorted(
                    [s0[0], s0[1],
                     4 + s1[0], 4 + s1[1],
                     8 + s2[0], 8 + s2[1]]))
                states.append(occ)
    return states


EDGES_INTRA = []
for c in range(N_CELLS):
    base = 4 * c
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        EDGES_INTRA.append((base + a, base + b))


def hamiltonian(t_intra=0.05, driver=3.0):
    B = build_basis()
    idx = {s: i for i, s in enumerate(B)}
    n = len(B)
    H = np.zeros((n, n), dtype=float)

    for i, occ in enumerate(B):
        occ_set = set(occ)

        # --- diagonal: Coulomb + driver ---
        e = 0.0
        for a, b in combinations(sorted(occ_set), 2):
            e += V_CACHE[(a, b)]

        P0 = ((0 in occ_set) + (2 in occ_set)
              - (1 in occ_set) - (3 in occ_set)) / 2.0
        e -= driver * P0
        H[i, i] = e

        # --- off-diagonal: intra-cell tunnelling only ---
        for x, y in EDGES_INTRA:
            if x in occ_set and y not in occ_set:
                new = tuple(sorted([s for s in occ if s != x] + [y]))
                j = idx[new]
                if j > i:
                    H[i, j] += -t_intra
                    H[j, i] += -t_intra

    return H, B


def cell_polarization(psi, B, cell):
    P = 0.0
    probs = np.abs(psi) ** 2
    for i, occ in enumerate(B):
        occ_set = set(occ)
        P += probs[i] * (
            (cell * 4 + 0 in occ_set) + (cell * 4 + 2 in occ_set)
            - (cell * 4 + 1 in occ_set) - (cell * 4 + 3 in occ_set)
        ) / 2.0
    return P


print("=== Exact 3-cell QCA wire (electrostatics, 216 states) ===\n")
print(f"{'t_intra':>8} {'driver':>7} | {'P0':>8} {'P1':>8} {'P2':>8}")
print("-" * 44)

for t in (0.10, 0.05, 0.02, 0.01):
    for driver in (+3.0, -3.0):
        H, B = hamiltonian(t_intra=t, driver=driver)
        w, v = np.linalg.eigh(H)
        psi = v[:, 0]
        P = [cell_polarization(psi, B, c) for c in range(N_CELLS)]
        print(f"{t:>8.3f} {driver:>+7.1f} | "
              f"{P[0]:>+8.4f} {P[1]:>+8.4f} {P[2]:>+8.4f}")
    print()
