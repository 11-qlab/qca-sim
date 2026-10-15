"""Kitaev honeycomb via exact Majorana fermionization.

Kitaev 2006 solution: substitute sigma^a = i b^a c.  The operators
u_ij = i b_i^a b_j^a commute with H and each other, with eigenvalues
+/-1.  Fixing u_ij = +1 everywhere (flux-free sector, Lieb's theorem)
reduces H to a quadratic form in the c fermions:

    H = (i/4) sum_{ij} A_ij c_i c_j

with A_ij = 2 J_a for an a-bond (i,j), A antisymmetric.

For real antisymmetric A, eigenvalues come in pairs +/- i*lambda.
Ground-state energy: E0 = -(1/2) sum_{lambda > 0} lambda.

This scales as (2 L^2)^3 for dense ED of A, so L=10 or more is
trivial.  It's the standard method for the Kitaev honeycomb.

The 4-fold topological degeneracy is checked by computing E0 in
the 4 flux sectors (0, 0), (1, 0), (0, 1), (1, 1).  In the B-phase
all four give the same E0 to numerical precision.
"""
from __future__ import annotations
import numpy as np


def site_idx(L, sub, i, j):
    return 2 * ((i % L) * L + (j % L)) + sub


def bonds(L):
    xb, yb, zb = [], [], []
    for i in range(L):
        for j in range(L):
            A = site_idx(L, 0, i, j)
            xb.append((A, site_idx(L, 1, i, j)))          # 0 deg
            yb.append((A, site_idx(L, 1, i, j - 1)))      # 120 deg
            zb.append((A, site_idx(L, 1, i - 1, j)))      # 240 deg
    return xb, yb, zb


def majorana_matrix(L, Jx=1.0, Jy=1.0, Jz=1.0, sector=(0, 0)):
    """Build the Majorana coupling matrix A.

    sector = (sx, sy) in {0,1}^2 controls the twist along the two
    non-contractible cycles.  (0,0) = flux-free.
    """
    n = 2 * L * L
    A = np.zeros((n, n))

    xb, yb, zb = bonds(L)

    # Sign flip pattern for a twisted boundary along x
    def x_twist(i, j):
        return -1 if (sector[0] and i == L - 1) else 1

    def y_twist(i, j):
        return -1 if (sector[1] and j == L - 1) else 1

    for (i, j) in xb:
        # x-bond is within a unit cell (i, j) -- always identity sign
        A[i, j] = 2 * Jx
        A[j, i] = -2 * Jx

    for (i, j) in yb:
        # y-bond connects cell (i, j) to B(i, j-1)
        # apply twist if either site is on the boundary
        sgn = y_twist(i // 2, (j // 2)) if sector[1] else 1
        A[i, j] = 2 * Jy * sgn
        A[j, i] = -2 * Jy * sgn

    for (i, j) in zb:
        sgn = x_twist(i // 2, j // 2) if sector[0] else 1
        A[i, j] = 2 * Jz * sgn
        A[j, i] = -2 * Jz * sgn

    return A


def ground_energy(A):
    """E0 = -(1/2) sum of positive imaginary parts of eigenvalues."""
    eigvals = np.linalg.eigvals(A)
    imag = eigvals.imag
    positive = imag[imag > 1e-8]
    return -0.5 * positive.sum(), len(positive)


if __name__ == "__main__":
    print("=== Kitaev honeycomb via Majorana fermionization ===\n")

    print("Isotropic (Jx = Jy = Jz = 1), flux-free sector:")
    print(f"  {'L':>4} {'sites':>7} {'E0':>14} {'E0/site':>12} {'modes':>8}")
    print("  " + "-" * 52)
    for L in (2, 3, 4, 5, 6, 8, 10, 20):
        A = majorana_matrix(L, 1.0, 1.0, 1.0)
        E0, nm = ground_energy(A)
        n = 2 * L * L
        print(f"  {L:>4} {n:>7} {E0:>14.6f} {E0/n:>12.6f} {nm:>8}")

    print()
    print("Topological degeneracy check (L=6, four flux sectors):")
    print(f"  {'sector':>10} {'E0':>14} {'delta':>10}")
    print("  " + "-" * 38)
    E_ref = None
    for s in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        A = majorana_matrix(6, 1.0, 1.0, 1.0, sector=s)
        E0, _ = ground_energy(A)
        if E_ref is None:
            E_ref = E0
        delta = E0 - E_ref
        print(f"  {str(s):>10} {E0:>14.6f} {delta:>+10.6f}")

    print()
    print("Anisotropic phase (Jx = Jy = 1, Jz = 3):")
    print(f"  {'L':>4} {'E0':>14} {'E0/site':>12}")
    print("  " + "-" * 34)
    for L in (4, 6, 8, 10):
        A = majorana_matrix(L, 1.0, 1.0, 3.0)
        E0, _ = ground_energy(A)
        n = 2 * L * L
        print(f"  {L:>4} {E0:>14.6f} {E0/n:>12.6f}")

    print()
    print("Expected:")
    print("  Isotropic E0/site should converge to the Kitaev 2006 value")
    print("  (~-0.5 for the isotropic case, exact number cited in the paper).")
    print("  All four flux sectors should give identical E0 at L=6 --")
    print("  that is the 4-fold topological degeneracy of the B-phase.")
