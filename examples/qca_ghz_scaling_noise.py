"""N-cell QCA chain: Scalability & Thermal Noise Analysis.

Hamiltonian:
    H = -J sum_i (sz_i sz_{i+1}) - g sum_i (sx_i)

We analyze:
1. Ground state fidelity with the N-qubit GHZ state as a function of N and g.
2. Spectral gap (Delta E = E1 - E0), which dictates adiabatic/cooling rates.
3. Thermal robustness: Fidelity of the thermal state rho = exp(-beta H) / Z
   with the target GHZ state at finite temperature T.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def get_operator_n(op, site, N):
    ops = [I2 for _ in range(N)]
    ops[site] = op
    res = ops[0]
    for o in ops[1:]:
        res = np.kron(res, o)
    return res


def H_chain_n(N, J=1.0, g=0.05):
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    # ZZ nearest-neighbor coupling
    for i in range(N - 1):
        H -= J * (get_operator_n(sz, i, N) @ get_operator_n(sz, i+1, N))
    # Transverse field
    for i in range(N):
        H -= g * get_operator_n(sx, i, N)
    return H


def get_ghz_state(N):
    ghz = np.zeros(2**N, dtype=complex)
    ghz[0] = 1.0 / np.sqrt(2)   # |00...0>
    ghz[-1] = 1.0 / np.sqrt(2)  # |11...1>
    return ghz


def thermal_state_fidelity(w, v, ghz, T):
    """Computes <psi_GHZ | rho | psi_GHZ> for rho = exp(-H/T) / Z."""
    if T <= 0:
        # Ground state fidelity
        psi0 = v[:, 0]
        return float(abs(ghz.conj() @ psi0) ** 2)
    
    # Shift energies for numerical stability
    E = w - w[0]
    boltzmann = np.exp(-E / T)
    Z = np.sum(boltzmann)
    probs = boltzmann / Z
    
    # rho = sum_k p_k |v_k><v_k|
    # Fidelity = sum_k p_k |<ghz|v_k>|^2
    overlaps = np.abs(v.conj().T @ ghz) ** 2
    return float(np.sum(probs * overlaps))


if __name__ == "__main__":
    print("=== N-Cell QCA Chain: Scaling & Thermal Robustness ===\n")
    
    # --- 1. Scaling with N at fixed g ---
    print("1. Ground state fidelity and spectral gap vs N (g = 0.05, J = 1.0):")
    print(f"   {'N':>4} | {'Gap (E1-E0)':>12} | {'F_GHZ':>10}")
    print("   " + "-" * 34)
    for N in (3, 4, 5, 6):
        H = H_chain_n(N, J=1.0, g=0.05)
        w, v = np.linalg.eigh(H)
        gap = float(w[1] - w[0])
        ghz = get_ghz_state(N)
        F = float(abs(ghz.conj() @ v[:, 0]) ** 2)
        print(f"   {N:>4} | {gap:>12.5f} | {F:>10.4f}")

    # --- 2. Thermal robustness for N = 4 ---
    print("\n2. Thermal robustness of N=4 GHZ state across temperatures T:")
    print(f"   {'Temperature T':>14} | {'Thermal Fidelity':>18}")
    print("   " + "-" * 37)
    N_test = 4
    H_test = H_chain_n(N_test, J=1.0, g=0.05)
    w_test, v_test = np.linalg.eigh(H_test)
    ghz_test = get_ghz_state(N_test)
    
    for T in (0.001, 0.01, 0.05, 0.1, 0.2, 0.5):
        f_therm = thermal_state_fidelity(w_test, v_test, ghz_test, T)
        print(f"   {T:>14.3f} | {f_therm:>18.4f}")

    print("\nSummary:")
    print("  - Larger N chains maintain near-unity GHZ fidelity as g -> 0.")
    print("  - Spectral gap decreases with N, highlighting the need for STA")
    print("    clocking or optimized cooling protocols to prevent critical slowing down.")
