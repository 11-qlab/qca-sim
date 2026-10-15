"""Kitaev honeycomb: 4-sector degeneracy test (CORRECTED twist).

Bug fixed: the twist is now applied to lattice coordinates (i, j)
inside the bond loop, not to site indices outside it.

Sector labels:
  sector[0] = 1 -> flip z-bonds with i=0 (cross x-boundary)
  sector[1] = 1 -> flip y-bonds with j=0 (cross y-boundary)

Both preserve every plaquette flux W_p (each hexagon gets 0 or 2
flipped bonds), so the W_p are unchanged.  The two large-loop
operators W_1, W_2 flip.  On a torus, all four sectors are
ground states of the Kitaev honeycomb -- that is the 4-fold
topological degeneracy of the B-phase.
"""
import numpy as np
import sys
sys.path.insert(0, "examples")

I2 = np.eye(2, dtype=complex)


def site_idx(L, sub, i, j):
    return 2 * ((i % L) * L + (j % L)) + sub


def majorana_matrix(L, Jx=1.0, Jy=1.0, Jz=1.0, sector=(0, 0)):
    """Build the Majorana coupling matrix A for the flux sector.

    sector = (sx, sy) in {0,1}^2
      sx = 1 -> flux through the x-cycle
      sy = 1 -> flux through the y-cycle
    """
    n = 2 * L * L
    A = np.zeros((n, n))

    for i in range(L):
        for j in range(L):
            A_idx = site_idx(L, 0, i, j)

            # x-bond: A(i,j) -- B(i,j)  (intra-cell, never twisted)
            B_x = site_idx(L, 1, i, j)
            A[A_idx, B_x] = 2 * Jx
            A[B_x, A_idx] = -2 * Jx

            # y-bond: A(i,j) -- B(i, j-1)  (crosses y-boundary when j=0)
            B_y = site_idx(L, 1, i, j - 1)
            sign_y = -1 if (sector[1] and j == 0) else 1
            A[A_idx, B_y] = 2 * Jy * sign_y
            A[B_y, A_idx] = -2 * Jy * sign_y

            # z-bond: A(i,j) -- B(i-1, j)  (crosses x-boundary when i=0)
            B_z = site_idx(L, 1, i - 1, j)
            sign_z = -1 if (sector[0] and i == 0) else 1
            A[A_idx, B_z] = 2 * Jz * sign_z
            A[B_z, A_idx] = -2 * Jz * sign_z

    return A


def ground_energy(A):
    eigvals = np.linalg.eigvals(A)
    imag = eigvals.imag
    positive = imag[imag > 1e-10]
    return -0.5 * positive.sum()


def sector_energies(L, Jx, Jy, Jz):
    return {s: ground_energy(majorana_matrix(L, Jx, Jy, Jz, sector=s))
            for s in [(0, 0), (1, 0), (0, 1), (1, 1)]}


def summarize(L, Jx, Jy, Jz, label):
    E = sector_energies(L, Jx, Jy, Jz)
    vals = list(E.values())
    spread = max(vals) - min(vals)
    print(f"  {label}  L={L:>2}  E0/site = {min(vals)/(2*L*L):.6f}  "
          f"spread = {spread:.3e}")
    return spread


if __name__ == "__main__":
    print("=== Kitaev honeycomb: 4-sector degeneracy (corrected) ===\n")

    print("Isotropic (Jx=Jy=Jz=1) -- gapless B-phase:")
    for L in (4, 5, 6, 7, 8, 9, 10):
        summarize(L, 1.0, 1.0, 1.0, "iso")

    print("\nAnisotropic (Jx=Jy=1, Jz=3) -- gapped B-phase:")
    for L in (4, 6, 8, 10, 12):
        summarize(L, 1.0, 1.0, 3.0, "gap")

    print("\nVery anisotropic (Jx=Jy=0.1, Jz=1) -- deeply gapped:")
    for L in (4, 6, 8, 10):
        summarize(L, 0.1, 0.1, 1.0, "deep")

    print("\nExpected with corrected twist:")
    print("  * All four sectors give the same E0 to machine precision.")
    print("  * The 4-fold degeneracy holds in BOTH phases.")
    print("  * Isotropic case: no O(1/L) splitting -- the degeneracy")
    print("    is exact, not asymptotic.")

    # Explicit sector-by-sector for one L to be sure
    print("\nFull sector table at L=8, isotropic:")
    E = sector_energies(8, 1.0, 1.0, 1.0)
    Emin = min(E.values())
    for s, e in E.items():
        print(f"  sector {str(s):>8}: E0 = {e:.10f}   delta = {e - Emin:+.3e}")
