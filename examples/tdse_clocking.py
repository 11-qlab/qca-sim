"""Time-dependent QCA cell driven by a finite-rate clock.

  i hbar dψ/dt = H(clock(t)) ψ

Compare adiabatic (slow) vs sudden (fast) switching.  If the
clock is too fast, the cell cannot follow and the bit is
corrupted -- this sets the maximum clock rate for a real device.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.linalg import expm


def basis():
    return list(combinations(range(4), 2))


def build_H(t, V_diag=1.0, V_adj=0.2, driver=0.0):
    B = basis()
    idx = {s: i for i, s in enumerate(B)}
    n = len(B)
    H = np.zeros((n, n), dtype=complex)
    for i, (a, b) in enumerate(B):
        e = 0.0
        if (a, b) in [(0, 2), (1, 3)]:
            e += V_diag
        if (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            e += V_adj
        e -= driver * ((a in (0, 2)) + (b in (0, 2))
                       - (a in (1, 3)) - (b in (1, 3))) / 2.0
        H[i, i] = e
    ADJ = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, (a, b) in enumerate(B):
        for (x, y) in ADJ + [(q, p) for (p, q) in ADJ]:
            if x in (a, b) and y not in (a, b):
                new = tuple(sorted([s for s in (a, b) if s != x] + [y]))
                H[i, idx[new]] += -t
    return H


def initial_state(t=0.05, driver=+3.0):
    """Ground state at strong driver = +1, as the starting state."""
    H = build_H(t, driver=driver)
    w, v = np.linalg.eigh(H)
    return v[:, 0].astype(complex)


def polarization(psi):
    B = basis()
    P = 0.0
    for c, (a, b) in zip(np.abs(psi) ** 2, B):
        P += c * ((a in (0, 2)) + (b in (0, 2))
                  - (a in (1, 3)) - (b in (1, 3))) / 2.0
    return float(P)


def evolve(clock_profile, T_total, n_steps, t_hop=0.05):
    """clock_profile(tau) -> driver; tau in [0, 1].  RK4 with expm."""
    psi = initial_state(t=t_hop, driver=clock_profile(0.0))
    dt = T_total / n_steps
    P_t = [polarization(psi)]
    for k in range(n_steps):
        tau_mid = (k + 0.5) / n_steps
        d = clock_profile(tau_mid)
        H = build_H(t_hop, driver=d)
        U = expm(-1j * H * dt)
        psi = U @ psi
        psi /= np.linalg.norm(psi)
        P_t.append(polarization(psi))
    return np.array(P_t)


if __name__ == "__main__":
    print("=== Time-dependent QCA clocking ===\n")
    print("Initial state: ground state with driver = +3  (P = +1)")
    print("Final clock:   driver = -3  (should flip to P = -1)")
    print()

    def clock(tau):
        # smooth ramp from +3 to -3
        return 3.0 - 6.0 * tau

    print(f"{'T_total':>10} {'rate':>10} {'P_final':>10} {'fidelity':>10}")
    print("-" * 44)
    for T_total in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0):
        P_t = evolve(clock, T_total, n_steps=400)
        P_final = P_t[-1]
        # fidelity = overlap with ideal final state (P = -1)
        # P = -1 corresponds to the electron sitting on dots 1,3
        fid = (1.0 - P_final) / 2.0   # 1.0 if perfectly flipped
        rate = 6.0 / T_total          # driver change per unit time
        print(f"{T_total:>10.2f} {rate:>10.2f} {P_final:>+10.4f} {fid:>10.4f}")

    print()
    print("Interpretation:")
    print("  Fast ramps (T_total << 1): cell cannot follow -> bit corrupted.")
    print("  Slow ramps (T_total >> 1): adiabatic -> clean flip.")
    print("  This sets the maximum clock rate for a QCA device.")
