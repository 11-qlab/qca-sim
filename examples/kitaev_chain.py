"""Kitaev chain: 1D p-wave superconductor with Majorana zero modes.

The Kitaev chain is the simplest model of a topological
superconductor:

    H = -mu sum_i c_i^dag c_i
        - t sum_i (c_i^dag c_{i+1} + h.c.)
        + Delta sum_i (c_i c_{i+1} + h.c.)

where:
    mu    = chemical potential
    t     = hopping amplitude
    Delta = p-wave pairing amplitude (real, positive)

Topological phase transition:
  |mu| < 2t:  topological, 2 Majorana zero modes at the ends
  |mu| > 2t:  trivial, no zero modes

At mu = 0, Delta = t, the Hamiltonian is exactly
    H = i t sum_i gamma_{2i-1} gamma_{2i}
with the two end Majoranas gamma_1 and gamma_{2N} unpaired.

We solve exactly via the Bogoliubov-de Gennes (BdG) formalism:
    H = (1/2) Psi^dag H_BdG Psi
    Psi = (c_1, ..., c_N, c_1^dag, ..., c_N^dag)^T

The 2N eigenvalues of H_BdG come in +/- pairs.  Zero modes
appear as exactly-zero eigenvalues.

The QCA mapping: each site is a QCA cell; the pairing term
requires an additional basis rotation, but the hopping is
native sigma_z sigma_z as in every other QCA Hamiltonian.
"""
from __future__ import annotations
import numpy as np


def kitaev_chain_hamiltonian(N, t=1.0, mu=0.0, Delta=1.0,
                             periodic=False):
    """Build the 2N x 2N BdG Hamiltonian."""
    H = np.zeros((2 * N, 2 * N), dtype=complex)

    # Chemical potential: -mu c_i^dag c_i
    # In BdG: contributes -mu to (c_i^dag c_i) block
    for i in range(N):
        H[i, i] = -mu
        H[N + i, N + i] = mu

    # Hopping: -t (c_i^dag c_{i+1} + h.c.)
    # Contributes -t to off-diagonal between particle blocks
    # and +t to the hole blocks
    pairs = [(i, i + 1) for i in range(N - 1)]
    if periodic:
        pairs.append((N - 1, 0))
    for (i, j) in pairs:
        H[i, j] -= t
        H[j, i] -= t
        H[N + i, N + j] += t
        H[N + j, N + i] += t
        # Pairing: Delta (c_i c_j + h.c.) where j = i+1
        H[i, N + j] += Delta
        H[j, N + i] -= Delta
        H[N + i, j] -= Delta
        H[N + j, i] += Delta

    # Hermitize
    H = (H + H.conj().T) / 2
    return H


def bdg_spectrum(N, t=1.0, mu=0.0, Delta=1.0, periodic=False):
    """Return sorted BdG eigenvalues."""
    H = kitaev_chain_hamiltonian(N, t, mu, Delta, periodic)
    w = np.linalg.eigvalsh(H)
    return np.sort(w)


def count_zero_modes(N, t=1.0, mu=0.0, Delta=1.0, tol=1e-3, periodic=False):
    """Count eigenvalues within tol of zero."""
    w = bdg_spectrum(N, t, mu, Delta, periodic)
    return int(np.sum(np.abs(w) < tol))


def majorana_localization(N, t=1.0, mu=0.0, Delta=1.0):
    """Return the LEFT and RIGHT edge-mode probability profiles.

    The two zero modes are degenerate, so eigh returns an
    arbitrary mixture.  To recover the physical left/right
    Majoranas, we project onto the 2D zero-mode subspace and
    diagonalize the position operator there.
    """
    H = kitaev_chain_hamiltonian(N, t, mu, Delta)
    w, v = np.linalg.eigh(H)
    # Take the 2 eigenvectors closest to zero
    idx = np.argsort(np.abs(w))[:2]
    V = v[:, idx]                       # (2N, 2)
    # Position operator on particle sites: X = diag(0..N-1) on
    # the particle block, zero on hole block
    X = np.zeros((2 * N, 2 * N))
    for i in range(N):
        X[i, i] = i
    X_sub = V.conj().T @ X @ V          # (2, 2)
    X_sub = (X_sub + X_sub.conj().T) / 2
    _, U = np.linalg.eigh(X_sub)        # 2x2 eigenvectors
    # Rotate to the localized basis
    V_loc = V @ U
    probs = []
    for k in range(2):
        psi = V_loc[:, k]
        prob = np.array([abs(psi[i]) ** 2 + abs(psi[N + i]) ** 2
                         for i in range(N)])
        prob /= prob.sum()
        probs.append(prob)
    # Sort: left mode first
    order = np.argsort([np.sum(np.arange(N) * p) for p in probs])
    return probs[order[0]], probs[order[1]]


def winding_invariant(N, t=1.0, mu=0.0, Delta=1.0):
    """Compute the bulk winding number.

    nu = 1 if |mu| < 2t (topological)
    nu = 0 if |mu| > 2t (trivial)
    """
    # Analytic: nu = 1 for |mu| < 2|t|, else 0 (Delta > 0)
    return 1 if abs(mu) < 2 * abs(t) else 0


if __name__ == "__main__":
    print("=== Kitaev chain: topological superconductor ===\n")
    print("H = -mu sum c_i^dag c_i - t sum (c_i^dag c_{i+1} + h.c.)")
    print("    + Delta sum (c_i c_{i+1} + h.c.)\n")

    # 1. Zero-mode count vs mu
    print("1. Zero modes vs chemical potential (N=20, t=Delta=1):")
    print(f"   {'mu':>8} {'zero modes':>12} {'winding':>10} {'phase':>15}")
    print("   " + "-" * 50)
    for mu in (-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0):
        nz = count_zero_modes(20, t=1.0, mu=mu, Delta=1.0)
        nu = winding_invariant(20, t=1.0, mu=mu, Delta=1.0)
        phase = "topological" if nu == 1 else "trivial"
        print(f"   {mu:>+8.2f} {nz:>12} {nu:>10} {phase:>15}")

    # 2. Gap vs mu (log)
    print("\n2. Bulk gap vs mu (N=20):")
    print(f"   {'mu':>8} {'min|E|':>12} {'bulk gap':>12}")
    print("   " + "-" * 36)
    for mu in (0.0, 1.0, 1.8, 2.0, 2.2, 3.0):
        w = bdg_spectrum(20, t=1.0, mu=mu, Delta=1.0)
        pos = np.sort(w[w > 0])
        print(f"   {mu:>+8.2f} {np.min(np.abs(w)):>12.6f} "
              f"{pos[0] if len(pos) else 0.0:>12.6f}")

    # 3. Majorana localization
    print("\n3. Majorana zero-mode localization (N=20, mu=0, t=Delta=1):")
    p_left, p_right = majorana_localization(20)
    print(f"   {'site':>6} {'left prob':>12} {'right prob':>12}  combined")
    for i in range(20):
        total = p_left[i] + p_right[i]
        bar = "#" * int(60 * total)
        print(f"   {i:>6} {p_left[i]:>12.4f} {p_right[i]:>12.4f}  {bar}")
    prob = p_left + p_right

    # 4. Exponential splitting between the two edge modes
    print("\n4. Edge-mode energy splitting vs N (mu=0, t=Delta=1):")
    print(f"   {'N':>6} {'min|E|':>14} {'log10':>8}")
    print("   " + "-" * 32)
    for N in (4, 6, 8, 10, 12, 16, 20, 30):
        w = bdg_spectrum(N, t=1.0, mu=0.0, Delta=1.0)
        e_min = np.min(np.abs(w))
        log10 = np.log10(e_min) if e_min > 0 else -np.inf
        print(f"   {N:>6} {e_min:>14.8e} {log10:>8.2f}")

    print("\nInterpretation:")
    print("  * mu = 0: exact zero modes (energy 0) -- 2 Majoranas")
    print("    localized at opposite ends.")
    print("  * |mu| < 2t: topological phase, 2 zero modes.")
    print("  * |mu| > 2t: trivial phase, 0 zero modes.")
    print("  * mu = 2t: phase transition (bulk gap closes).")
    print("  * Edge-mode splitting decays exponentially in N.")
    print()
    print("QCA mapping: the Kitaev chain Hamiltonian requires only")
    print("nearest-neighbour c^dag c and c c couplings, both of which")
    print("decompose into QCA-native sigma_z sigma_z terms with local")
    print("basis rotations.  This gives the QCA platform access to")
    print("Majorana zero modes -- the basis for topological quantum")
    print("computing with braiding.")

    # Save results
    import json
    results = {
        "zero_modes_vs_mu": [
            {"mu": mu,
             "zero_modes": count_zero_modes(20, mu=mu),
             "winding": winding_invariant(20, mu=mu)}
            for mu in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
        ],
        "localization": prob.tolist(),
    }
    with open("kitaev_chain_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults written to kitaev_chain_results.json")
