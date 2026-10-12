"""QCA wire: signal propagation along a chain of coupled cells.

Each cell responds to the field of its left neighbour. The
input cell is pinned at P = +1 (driver) or P = -1. A properly
functioning QCA wire propagates the bit without attenuation
in the low-tunnelling (bistable) regime.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def cell_response(driver, t, V_diag=1.0, V_adj=0.2):
    H, B = hamiltonian(t=t, V_diag=V_diag, V_adj=V_adj, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def propagate(n_cells, P_in, t, coupling=1.0):
    P = np.zeros(n_cells)
    P[0] = P_in
    for i in range(1, n_cells):
        P[i] = cell_response(coupling * P[i - 1], t)
    return P


print("=== QCA wire propagation ===\n")
for t in (0.30, 0.10, 0.05, 0.02):
    print(f"--- t = {t:.2f} ---")
    for P_in in (-1.0, +1.0):
        P = propagate(8, P_in, t)
        line = "  ".join(f"{p:+.3f}" for p in P)
        print(f"  in = {P_in:+.0f} : {line}")
    # measure attenuation: |P| at output cell
    P_plus  = propagate(8, +1.0, t)
    P_minus = propagate(8, -1.0, t)
    print(f"  |P_out| for +1 : {abs(P_plus[-1]):.4f}")
    print(f"  |P_out| for -1 : {abs(P_minus[-1]):.4f}")
    print()

print("Interpretation:")
print("  High t  -> signal attenuates (quantum leakage)")
print("  Low  t  -> bit propagates nearly unattenuated end-to-end")
