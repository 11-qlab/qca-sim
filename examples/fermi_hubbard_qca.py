"""Fermi-Hubbard model on a QCA lattice.

    H = -t sum_{<ij>,s} (c†_{i,s} c_{j,s} + h.c.)
        + U sum_i n_{i,up} n_{i,down}

Qubit layout (Jordan-Wigner):
    qubit 2*i   = site i, spin up
    qubit 2*i+1 = site i, spin down

c_k = Z_0 Z_1 ... Z_{k-1} * sigma^-_k

Work in the fixed-particle-number sector N_up = N_down = N_sites/2
(half filling) to keep the Hilbert space tractable.

QCA-native gate decomposition:
  - Interaction U n_up n_down = (U/4)(I - Z_up - Z_down + Z_up Z_down)
    The Z_up Z_down term is a native sigma_z sigma_z QCA coupling.
  - Hopping -t (c†_i c_j + h.c.) becomes, under JW, a two-qubit
    X X / Y Y term dressed by a Z-string.  Every piece is a
    single-qubit rotation plus native ZZ couplings.

Benchmark (Lieb-Wu exact for 1D, U/t=4): E0/N ~ -0.5 (spinful).
"""
from __future__ import annotations
import numpy as np
import time
from itertools import combinations
from scipy.sparse import csr_matrix, kron, identity, eye
from scipy.sparse.linalg import eigsh

I2 = csr_matrix(np.eye(2, dtype=complex))
X  = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y  = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
Z  = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
Sm = csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))  # sigma^-


def kron_tensor(ops):
    while len(ops) > 1:
        new = [kron(ops[i], ops[i + 1], format="csr")
               for i in range(0, len(ops) - 1, 2)]
        if len(ops) % 2 == 1:
            new.append(ops[-1])
        ops = new
    return ops[0]


def jw_c(n, k):
    """c_k as a sparse matrix on n qubits (JW convention)."""
    ops = []
    for q in range(n):
        if q < k:
            ops.append(Z)
        elif q == k:
            ops.append(Sm)
        else:
            ops.append(I2)
    return kron_tensor(ops)


def jw_cdag(n, k):
    return jw_c(n, k).conj().T.tocsr()


def project_to_sector(H_full, n, n_up, n_down):
    """Project H to the sector with n_up up-spins and n_down down-spins.

    The qubit basis: for spin up, qubits 0, 2, 4, ...; for spin down,
    1, 3, 5, ...
    """
    dim = 2 ** n
    states = []
    for basis in range(dim):
        bits = [(basis >> q) & 1 for q in range(n)]
        up = sum(bits[2 * i] for i in range(n // 2))
        dn = sum(bits[2 * i + 1] for i in range(n // 2))
        if up == n_up and dn == n_down:
            states.append(basis)
    states = np.array(states, dtype=np.int64)
    sub = H_full[states][:, states].tocsr()
    return sub, states


def site_qubit(i, spin):
    """Qubit index for site i, spin 0 (up) or 1 (down)."""
    return 2 * i + spin


def fermi_hubbard_1d(n_sites, t=1.0, U=4.0, periodic=False):
    """Build H in the full 2^n_sites * 2 Hilbert space, then project."""
    n = 2 * n_sites
    dim = 2 ** n

    # Precompute annihilation operators
    c = [jw_c(n, q) for q in range(n)]
    cdag = [x.conj().T.tocsr() for x in c]

    # Hopping
    pairs = [(i, i + 1) for i in range(n_sites - 1)]
    if periodic:
        pairs.append((n_sites - 1, 0))

    H_full = csr_matrix((dim, dim), dtype=complex)
    for (i, j) in pairs:
        for spin in (0, 1):
            qi = site_qubit(i, spin)
            qj = site_qubit(j, spin)
            hop = cdag[qi] @ c[qj]
            H_full = H_full - t * (hop + hop.conj().T)

    # Interaction
    for i in range(n_sites):
        qu = site_qubit(i, 0)
        qd = site_qubit(i, 1)
        nu = cdag[qu] @ c[qu]
        nd = cdag[qd] @ c[qd]
        H_full = H_full + U * (nu @ nd)

    # Project to half-filling
    n_up = n_down = n_sites // 2
    H, states = project_to_sector(H_full, n, n_up, n_down)
    return H, states, n


def fermi_hubbard_2d(L, t=1.0, U=4.0, periodic=False):
    """2D L x L square lattice Fermi-Hubbard."""
    n_sites = L * L
    n = 2 * n_sites
    dim = 2 ** n

    c = [jw_c(n, q) for q in range(n)]
    cdag = [x.conj().T.tocsr() for x in c]

    def sidx(i, j):
        return (i % L) * L + (j % L)

    pairs = []
    for i in range(L):
        for j in range(L):
            a = sidx(i, j)
            if j + 1 < L or periodic:
                pairs.append((a, sidx(i, j + 1)))
            if i + 1 < L or periodic:
                pairs.append((a, sidx(i + 1, j)))

    H_full = csr_matrix((dim, dim), dtype=complex)
    for (i, j) in pairs:
        for spin in (0, 1):
            qi = site_qubit(i, spin)
            qj = site_qubit(j, spin)
            hop = cdag[qi] @ c[qj]
            H_full = H_full - t * (hop + hop.conj().T)

    for i in range(n_sites):
        qu = site_qubit(i, 0)
        qd = site_qubit(i, 1)
        nu = cdag[qu] @ c[qu]
        nd = cdag[qd] @ c[qd]
        H_full = H_full + U * (nu @ nd)

    n_up = n_down = n_sites // 2
    H, states = project_to_sector(H_full, n, n_up, n_down)
    return H, states, n


def doublon_density(H, states, n, n_sites):
    """<D>/N = (1/N) sum_i <n_up_i n_down_i>."""
    dim = 2 ** n
    D = csr_matrix((dim, dim), dtype=complex)
    for i in range(n_sites):
        qu = site_qubit(i, 0)
        qd = site_qubit(i, 1)
        nu = jw_c(n, qu).conj().T @ jw_c(n, qu)
        nd = jw_c(n, qd).conj().T @ jw_c(n, qd)
        D = D + nu @ nd
    D_sub = D[states][:, states].tocsr()
    w, v = eigsh(H, k=1, which="SA")
    psi = v[:, 0]
    return float(np.real(psi.conj() @ (D_sub @ psi))) / n_sites


if __name__ == "__main__":
    print("=== Fermi-Hubbard on a QCA lattice ===\n")

    print("1D chain, half filling, U/t = 4 (Lieb-Wu benchmark):")
    print(f"  {'N_sites':>8} {'qubits':>8} {'dim(sector)':>12} "
          f"{'E0/N':>12} {'D/N':>10} {'time':>6}")
    print("  " + "-" * 62)

    for n_sites in (4, 6, 8):
        t0 = time.time()
        H, states, nq = fermi_hubbard_1d(n_sites, t=1.0, U=4.0)
        w, v = eigsh(H, k=1, which="SA")
        E0 = w[0]
        D = doublon_density(H, states, nq, n_sites)
        dt = time.time() - t0
        print(f"  {n_sites:>8} {nq:>8} {H.shape[0]:>12} "
              f"{E0/n_sites:>12.6f} {D:>10.6f} {dt:>5.1f}s")

    print()
    print("1D chain, U sweep at N=8 sites:")
    print(f"  {'U/t':>6} {'E0/N':>12} {'D/N':>10}")
    print("  " + "-" * 30)
    for U in (0.0, 1.0, 2.0, 4.0, 8.0):
        H, states, nq = fermi_hubbard_1d(8, t=1.0, U=U)
        w, v = eigsh(H, k=1, which="SA")
        D = doublon_density(H, states, nq, 8)
        print(f"  {U:>6.1f} {w[0]/8:>12.6f} {D:>10.6f}")

    print()
    print("2D square lattice, half filling, U/t = 4:")
    print(f"  {'L':>4} {'N_sites':>8} {'qubits':>8} {'dim(sector)':>12} "
          f"{'E0/N':>12} {'time':>6}")
    print("  " + "-" * 56)
    for L in (2,):
        t0 = time.time()
        H, states, nq = fermi_hubbard_2d(L, t=1.0, U=4.0)
        w, v = eigsh(H, k=1, which="SA")
        E0 = w[0]
        dt = time.time() - t0
        print(f"  {L:>4} {L*L:>8} {nq:>8} {H.shape[0]:>12} "
              f"{E0/(L*L):>12.6f} {dt:>5.1f}s")

    print()
    print("QCA-native gate decomposition:")
    print("  Interaction U n_up n_down")
    print("    = (U/4) (I - Z_up)(I - Z_down)")
    print("    = (U/4)(I - Z_up - Z_down + Z_up Z_down)")
    print("    -> native sigma_z sigma_z QCA coupling + single-qubit Rz.")
    print("  Hopping -t(c†_i c_j + h.c.)")
    print("    -> JW: -t/2 (X_i Z_string X_j + Y_i Z_string Y_j)")
    print("    -> single-qubit rotations + ZZ couplings along the string.")
    print()
    print("Benchmark (Lieb-Wu exact for 1D, U/t=4):")
    print("  E0/N -> -0.5 in the thermodynamic limit.")
    print("  Finite-size corrections are O(1/N) and oscillatory.")
    print("  Doublon density D/N decreases with U -- Mott physics.")
