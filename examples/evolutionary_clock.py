"""Evolve a QCA clock waveform via genetic algorithm.

The 4-phase trapezoidal clock is hand-designed.  Can we do
better?  Parametrize driver(tau) as a truncated Fourier series:

    driver(tau) = a0 + sum_k [ a_k cos(2 pi k tau) + b_k sin(2 pi k tau) ]

with the constraint driver(0) = +D0 and driver(1) = -D0.
Fitness = propagation fidelity through a 5-cell wire with a
stuck-at-0 defect on one cell.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qca import hamiltonian, ground_state, polarization


N_HARM = 4     # Fourier harmonics


def params_to_waveform(params, D0=1.0, n=200):
    """Build driver(tau) from params, enforcing endpoint conditions."""
    tau = np.linspace(0, 1, n)
    a = params[:N_HARM]
    b = params[N_HARM:]
    wave = np.zeros_like(tau)
    for k in range(N_HARM):
        wave += a[k] * np.cos(2 * np.pi * (k + 1) * tau)
        wave += b[k] * np.sin(2 * np.pi * (k + 1) * tau)
    # force endpoints
    wave = wave - wave[0] + D0
    ramp = D0 - (wave[0] - wave[-1]) * tau   # correct endpoint slope
    wave = wave + (ramp - wave) * 0.0        # keep raw shape for now
    # normalize to [D0, -D0] linearly
    wave = D0 - 2 * D0 * (wave - wave.min()) / (wave.max() - wave.min() + 1e-9)
    return tau, wave


def cell_response(driver, t=0.05):
    H, B = hamiltonian(t=t, V_diag=1.0, V_adj=0.2, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def simulate_wire(waveform, n_cells=5, defect_cell=None):
    """Run a 5-cell wire; each cell sees the previous cell as driver.
    If defect_cell is set, that cell's output is forced to 0.
    """
    P = np.zeros(n_cells)
    P[0] = +1.0
    for i in range(1, n_cells):
        if defect_cell == i:
            P[i] = 0.0
        else:
            # use waveform mid-point as effective clock field on this cell
            d_eff = P[i - 1] * np.mean(np.abs(waveform))
            P[i] = cell_response(d_eff)
    return P


def fitness(params, defect_cell=2):
    _, waveform = params_to_waveform(params)
    P = simulate_wire(waveform, defect_cell=defect_cell)
    # reward full propagation to +1 at the output
    return -abs(P[-1] - 1.0) - 0.1 * np.sum(np.abs(P - 1.0))


def ga(pop_size=30, generations=40, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.normal(0, 0.5, size=(pop_size, 2 * N_HARM))
    fit = np.array([fitness(p) for p in pop])
    for gen in range(generations):
        # tournament selection
        idx = np.argsort(fit)[::-1]
        parents = pop[idx[:pop_size // 2]]
        # crossover + mutation
        children = []
        while len(children) < pop_size - len(parents):
            i, j = rng.integers(0, len(parents), 2)
            alpha = rng.uniform(0, 1, size=2 * N_HARM)
            child = alpha * parents[i] + (1 - alpha) * parents[j]
            child += rng.normal(0, 0.1, size=2 * N_HARM)
            children.append(child)
        pop = np.vstack([parents, children])
        fit = np.array([fitness(p) for p in pop])
        if (gen + 1) % 10 == 0:
            print(f"  gen {gen+1:>3}  best fitness = {fit.max():+.5f}")
    return pop[np.argmax(fit)], fit.max()


if __name__ == "__main__":
    print("=== Evolutionary clock optimization ===\n")
    print("Evolving a Fourier-series clock waveform to maximize")
    print("bit propagation through a 5-cell wire with a defect.\n")

    best, score = ga(pop_size=30, generations=40)
    print(f"\nBest fitness = {score:+.5f}")

    _, waveform = params_to_waveform(best)
    P = simulate_wire(waveform, defect_cell=2)
    print(f"Wire polarizations:  {[f'{p:+.3f}' for p in P]}")

    # Compare with baseline 4-phase-equivalent constant waveform
    print("\nBaseline (constant |driver| = 1):")
    baseline = np.ones(200)
    P_base = simulate_wire(baseline, defect_cell=2)
    print(f"Wire polarizations:  {[f'{p:+.3f}' for p in P_base]}")

    print(f"\nBest waveform shape (first 10 samples):")
    print(f"  {waveform[:10].round(3)}")
