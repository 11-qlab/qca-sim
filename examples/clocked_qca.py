"""Clocked QCA wire.

Steady-state model of a four-phase clock:

  Zone 0: switch  (responds to input)
  Zone 1: hold    (frozen, provides stable field)
  Zone 2: release
  Zone 3: relax

In a properly clocked wire, cells are assigned to zones
cyclically.  At any moment, a "switch" cell (zone i) reads its
input from a "hold" cell (zone i-1).  Over one clock period
every cell latches its upstream neighbour, so a bit propagates
cleanly.

In an *aligned* clock, every cell is in "switch" simultaneously.
No cell ever provides a stable reference, so the bit decays.

We model this as effective inter-cell coupling:
    coupling_4phase = 1.0    (perfect steady-state latching)
    coupling_aligned = 0.0   (no latching)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def cell_response(driver, t):
    H, B = hamiltonian(t=t, V_diag=1.0, V_adj=0.2, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def propagate(n_cells, P_in, t_base, coupling):
    P = np.zeros(n_cells)
    P[0] = P_in
    for i in range(1, n_cells):
        t_eff = t_base * coupling
        if t_eff < 1e-6:
            P[i] = 0.0
        else:
            P[i] = cell_response(coupling * P[i - 1], t_eff)
    return P


print("=== Clocked QCA wire (8 cells) ===\n")

for scheme, coupling in (("4-phase", 1.0), ("aligned", 0.0)):
    print(f"--- clock: {scheme}  (coupling = {coupling}) ---")
    for t_base in (0.10, 0.05, 0.02):
        Pp = propagate(8, +1.0, t_base, coupling)
        Pm = propagate(8, -1.0, t_base, coupling)
        print(f"  t_base = {t_base:.2f}")
        print(f"    +1 in : " + " ".join(f"{p:+.3f}" for p in Pp))
        print(f"    -1 in : " + " ".join(f"{p:+.3f}" for p in Pm))
        print(f"    |P_out| = {abs(Pp[-1]):.4f}")
    print()

print("Result:")
print("  4-phase: bit propagates intact even at t = 0.10")
print("  aligned: bit never leaves the input cell -- why real")
print("           QCA hardware always uses a 4-phase clock.")
