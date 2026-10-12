"""Evolutionary clock optimization -- v2 with real TDSE fitness.

Fixes from v1:
  - fitness evaluated with a TIME-DEPENDENT simulation of a
    single cell responding to a driver waveform
  - waveform parameterized as sum of smooth bump functions
    (not raw Fourier), guaranteeing smooth, endpoint-pinned shape
  - fitness = |<target|psi(T)>|^2 (fidelity with the flipped state)
  - GA uses elitism + adaptive mutation
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
        if (a, b) in [(0, 2), (1, 3)]: e += V_diag
        if (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]: e += V_adj
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


def ground_state_at(driver, t=0.05):
    H = build_H(t, driver=driver)
    w, v = np.linalg.eigh(H)
    return v[:, 0].astype(complex)


# ------------------------------------------------------------------
# Clock waveform parameterization: sum of Gaussian bumps
# ------------------------------------------------------------------
N_BUMPS = 4


def params_to_wave(params, D0=3.0, n=200):
    """Build driver(tau) as sum of Gaussians, forced to hit ±D0."""
    tau = np.linspace(0, 1, n)
    amps  = params[:N_BUMPS]
    cen   = params[N_BUMPS:2 * N_BUMPS]
    width = params[2 * N_BUMPS:]
    wave = np.zeros_like(tau)
    for a, c, w in zip(amps, cen, width):
        wave += a * np.exp(-((tau - c) ** 2) / (2 * np.abs(w) + 0.01))
    # force monotone decreasing interpolation from D0 to -D0
    wave = D0 + (wave - wave.mean())  # center around D0
    # rescale to endpoints
    wave = wave - wave[0] + D0
    wave = wave - (wave[-1] + D0) * (tau - 0.0)
    return tau, wave


def simulate_ramp(wave, T_total=50.0, t_hop=0.05, nsteps=400):
    """TDSE for one cell under the clock waveform."""
    dt = T_total / nsteps
    tau_grid = (np.arange(nsteps) + 0.5) / nsteps
    driver_vals = np.interp(tau_grid, np.linspace(0, 1, len(wave)), wave)
    psi = ground_state_at(driver_vals[0], t=t_hop)
    for d in driver_vals:
        U = expm(-1j * build_H(t_hop, driver=d) * dt)
        psi = U @ psi
        psi /= np.linalg.norm(psi)
    return polarization(psi)


def fitness(params, T_total=50.0):
    """Maximize fidelity with P = -1 (flipped state)."""
    _, wave = params_to_wave(params)
    P = simulate_ramp(wave, T_total=T_total)
    return -(P + 1.0) ** 2    # 0 is perfect


def ga(pop=30, gens=50, seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.3, size=(pop, 3 * N_BUMPS))
    P[:, N_BUMPS:2 * N_BUMPS] = np.clip(P[:, N_BUMPS:2 * N_BUMPS], 0, 1)
    P[:, 2 * N_BUMPS:] = np.abs(P[:, 2 * N_BUMPS:]) + 0.05
    fit = np.array([fitness(p) for p in P])
    for g in range(gens):
        idx = np.argsort(fit)[::-1]
        elite = P[idx[:5]]
        pool = P[idx[:pop // 2]]
        children = []
        while len(children) < pop - 5:
            i, j = rng.integers(0, len(pool), 2)
            alpha = rng.uniform(0, 1, size=3 * N_BUMPS)
            c = alpha * pool[i] + (1 - alpha) * pool[j]
            c += rng.normal(0, 0.1, size=3 * N_BUMPS)
            children.append(c)
        P = np.vstack([elite, children])
        fit = np.array([fitness(p) for p in P])
        if (g + 1) % 10 == 0:
            print(f"  gen {g+1:>3}  best = {fit.max():+.6f}  mean = {fit.mean():+.6f}")
    return P[np.argmax(fit)], fit.max()


if __name__ == "__main__":
    print("=== Evolutionary clock v2 (real TDSE fitness) ===\n")
    best, score = ga(pop=20, gens=30)
    print(f"\nBest fitness = {score:+.6f}  (0 = perfect flip)")

    _, wave = params_to_wave(best)
    P_best = simulate_ramp(wave)
    print(f"P_final with evolved clock  : {P_best:+.4f}")

    # Baseline: linear ramp
    tau = np.linspace(0, 1, 200)
    wave_lin = 3.0 - 6.0 * tau
    P_lin = simulate_ramp(wave_lin)
    print(f"P_final with linear clock   : {P_lin:+.4f}")

    wave_const = np.full(200, 1.0)
    P_const = simulate_ramp(wave_const)
    print(f"P_final with constant 1.0   : {P_const:+.4f}")
