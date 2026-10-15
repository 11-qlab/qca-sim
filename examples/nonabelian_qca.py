"""Kitaev honeycomb on a QCA lattice -- corrected geometry.

The previous version used the wrong bond vectors.  Corrected here:
lattice vectors a1 = (3/2, sqrt(3)/2), a2 = (3/2, -sqrt(3)/2); A at
origin, B at (1, 0).  Bond types are labelled by direction:

    x-bond (angle 0°):   A(i,j) -- B(i,j)
    y-bond (angle 120°): A(i,j) -- B(i,j-1)
    z-bond (angle 240°): A(i,j) -- B(i-1,j)

The three bonds from every A-site are 120° apart -- real honeycomb.
The plaquette operator uses alternating Paulis (Y,Z,X,Y,Z,X going
around), not all-Z.  Self-test: [H, W_p] = 0 for every plaquette.
"""
from __future__ import annotations
import numpy as np
import time
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import eigsh

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P = {"I": I2, "X": X, "Y": Y, "Z": Z}


def site_idx(L, sub, i, j):
    return 2 * ((i % L) * L + (j % L)) + sub


def bonds(L):
    xb, yb, zb = [], [], []
    for i in range(L):
        for j in range(L):
            A = site_idx(L, 0, i, j)
            xb.append((A, site_idx(L, 1, i, j)))          # A(i,j) -- B(i,j)
            yb.append((A, site_idx(L, 1, i, j - 1)))      # A(i,j) -- B(i,j-1)
            zb.append((A, site_idx(L, 1, i - 1, j)))      # A(i,j) -- B(i-1,j)
    return xb, yb, zb


def kron_tensor(ops):
    while len(ops) > 1:
        new = [kron(ops[i], ops[i + 1], format="csr")
               for i in range(0, len(ops) - 1, 2)]
        if len(ops) % 2 == 1:
            new.append(ops[-1])
        ops = new
    return ops[0]


def two_site(n, q1, q2, p1, p2):
    ops = [csr_matrix(P[p1]) if q == q1
           else csr_matrix(P[p2]) if q == q2
           else csr_matrix(I2)
           for q in range(n)]
    return kron_tensor(ops)


def kitaev_H(L, Jx=1.0, Jy=1.0, Jz=1.0):
    n = 2 * L * L
    H = csr_matrix((2 ** n, 2 ** n), dtype=complex)
    xb, yb, zb = bonds(L)
    for (a, b) in xb:
        H = H - Jx * two_site(n, a, b, "X", "X")
    for (a, b) in yb:
        H = H - Jy * two_site(n, a, b, "Y", "Y")
    for (a, b) in zb:
        H = H - Jz * two_site(n, a, b, "Z", "Z")
    return H, n


def plaquette_ops(L):
    """Hexagon (i,j) has sites in cyclic order:
       A(i,j), B(i,j), A(i,j+1), B(i-1,j+1), A(i-1,j+1), B(i-1,j)
       with alternating Paulis (Y,Z,X,Y,Z,X) at those sites.
    """
    n = 2 * L * L
    ops = []
    for i in range(L):
        for j in range(L):
            paulis = [
                (site_idx(L, 0, i, j),       "Y"),
                (site_idx(L, 1, i, j),       "Z"),
                (site_idx(L, 0, i, j + 1),   "X"),
                (site_idx(L, 1, i - 1, j + 1), "Y"),
                (site_idx(L, 0, i - 1, j + 1), "Z"),
                (site_idx(L, 1, i - 1, j),   "X"),
            ]
            pdict = dict(paulis)
            ops_list = [csr_matrix(P[pdict.get(q, "I")]) for q in range(n)]
            ops.append(((i, j), kron_tensor(ops_list)))
    return ops


def check_commutators(H, w_ops, tol=1e-8):
    """Verify [H, W_p] = 0 for every plaquette."""
    worst = 0.0
    for (ij, W) in w_ops:
        comm = H @ W - W @ H
        err = abs(comm).max()
        worst = max(worst, err)
    return worst


def lowest(H, n, k=6):
    dim = 2 ** n
    if dim <= 4096:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = eigsh(H, k=k, which="SA", tol=1e-9)
    order = np.argsort(w)
    return w[order], v[:, order]


if __name__ == "__main__":
    print("=== Kitaev honeycomb (corrected geometry) ===\n")

    for L in (2, 3):
        n = 2 * L * L
        print(f"--- L = {L}  ({n} sites, dim = 2^{n}) ---")

        t0 = time.time()
        H, _ = kitaev_H(L, 1.0, 1.0, 1.0)
        print(f"  H built in {time.time()-t0:.1f}s")

        # Self-test: [H, W_p] = 0
        w_ops = plaquette_ops(L)
        t0 = time.time()
        err = check_commutators(H, w_ops)
        print(f"  max |[H, W_p]| = {err:.2e}  ({time.time()-t0:.1f}s)"
              f"  [{'PASS' if err < 1e-6 else 'FAIL'}]")

        t0 = time.time()
        w, v = lowest(H, n, k=8)
        print(f"  ED in {time.time()-t0:.1f}s")
        E0 = w[0]
        deg = int(np.sum(np.abs(w - E0) < 1e-6))
        gap = w[deg] - E0 if deg < len(w) else float("nan")
        print(f"  E0 = {E0:+.6f}")
        print(f"  Lowest 8: " + " ".join(f"{e:+.4f}" for e in w))
        print(f"  Degeneracy = {deg}   (expected: 4)")
        print(f"  Gap = {gap:.4f}")

        # Plaquette flux on ground state
        psi = v[:, 0]
        fluxes = []
        for (ij, W) in w_ops:
            fluxes.append(float(np.real(psi.conj() @ (W @ psi))))
        print(f"  <W_p> = " + " ".join(f"{x:+.3f}" for x in fluxes))
        print()

    print("Interpretation:")
    print("  Isotropic (Jx=Jy=Jz=1) B-phase of the Kitaev honeycomb")
    print("  has 4-fold ground-state degeneracy on the torus, and")
    print("  every plaquette flux <W_p> = +1 (vortex-free).")
    print("  If both checks pass, the QCA lattice hosts the")
    print("  non-Abelian Ising anyons under a magnetic field.")
