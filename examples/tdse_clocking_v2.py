"""Time-dependent QCA cell — Landau-Zener-correct analysis.

Key insight: the adiabatic threshold is T >> 1/(2*pi*Delta^2)
where Delta is the minimum gap traversed.  For t_hop = 0.05,
Delta ~ 0.05, so T_adiabatic ~ 2000, not 200.

We also compare a linear ramp (crosses the gap at full speed)
with a smoothstep ramp (dwells near the gap minimum).
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
    H = np.zeros((len(B), len(B)), dtype=complex)
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


def polarization(psi):
    B = basis()
    P = 0.0
    for c, (a, b) in zip(np.abs(psi) ** 2, B):
        P += c * ((a in (0, 2)) + (b in (0, 2))
                  - (a in (1, 3)) - (b in (1, 3))) / 2.0
    return float(P)


def gap_at(driver, t=0.05):
    H = build_H(t, driver=driver)
    w = np.linalg.eigvalsh(H)
    return float(w[1] - w[0])


def initial_state(t=0.05, driver=+3.0):
    H = build_H(t, driver=driver)
    w, v = np.linalg.eigh(H)
    return v[:, 0].astype(complex)


def ramp_linear(tau):
    return 3.0 - 6.0 * tau


def ramp_smoothstep(tau):
    u = 3 * tau ** 2 - 2 * tau ** 3      # smoothstep
    return 3.0 - 6.0 * u


def evolve(ramp, T_total, t_hop=0.05, n_steps_per_unit=100):
    n_steps = max(int(T_total * n_steps_per_unit), 100)
    dt = T_total / n_steps
    psi = initial_state(t=t_hop, driver=ramp(0.0))
    P_t = np.empty(n_steps + 1)
    P_t[0] = polarization(psi)
    for k in range(n_steps):
        tau = (k + 0.5) / n_steps
        H = build_H(t_hop, driver=ramp(tau))
        psi = expm(-1j * H * dt) @ psi
        psi /= np.linalg.norm(psi)
        P_t[k + 1] = polarization(psi)
    return P_t


if __name__ == "__main__":
    print("=== QCA cell: gap at the clock crossing ===\n")
    print(f"{'driver':>8} {'gap':>10}")
    for d in (-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0):
        print(f"{d:>8.2f} {gap_at(d):>10.5f}")
    Delta_min = gap_at(0.0)
    print(f"\nMinimum gap Delta_min = {Delta_min:.5f}")
    print(f"Adiabatic threshold T >> 1/(2*pi*Delta^2) = "
          f"{1/(2*np.pi*Delta_min**2):.0f}\n")

    print("=== Fidelity vs ramp time (linear vs smoothstep) ===\n")
    print(f"{'T_total':>10} | {'linear P':>10} {'lin fid':>8} | "
          f"{'smooth P':>10} {'sm fid':>8}")
    print("-" * 60)
    for T in (50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0):
        P_lin = evolve(ramp_linear, T)[-1]
        P_sm  = evolve(ramp_smoothstep, T)[-1]
        fid_lin = (1.0 - P_lin) / 2.0
        fid_sm  = (1.0 - P_sm)  / 2.0
        print(f"{T:>10.0f} | {P_lin:>+10.4f} {fid_lin:>8.4f} | "
              f"{P_sm:>+10.4f} {fid_sm:>8.4f}")

    print("\nInterpretation:")
    print("  Linear ramp: crosses gap at constant speed; needs T ~ 1/Delta^2.")
    print("  Smoothstep: dwells at gap minimum; reaches adiabatic at lower T.")
    print("  The *shape* of the clock waveform matters as much as its rate.")
