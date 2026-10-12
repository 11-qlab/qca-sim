"""QCA majority gate: out = majority(A, B, C).

The device cell sits at the centre of three input cells. Its
total driver field is the sum of the three input polarizations.
At low tunnelling the device cell saturates to the majority
value, giving the universal Boolean primitive MAJ(A,B,C).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def cell_response(driver, t, V_diag=1.0, V_adj=0.2):
    H, B = hamiltonian(t=t, V_diag=V_diag, V_adj=V_adj, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def majority_out(A, B, C, t, coupling=1.0):
    driver = coupling * (A + B + C)
    return cell_response(driver, t)


def truth_table(t):
    print(f"=== QCA majority gate truth table (t = {t:.2f}) ===\n")
    print(f"{'A':>3} {'B':>3} {'C':>3} | {'driver':>8} {'P_out':>8} | {'logic':>6}")
    print("-" * 46)
    correct = 0
    for A in (-1, +1):
        for B in (-1, +1):
            for C in (-1, +1):
                P = majority_out(A, B, C, t)
                logic = 1 if P > 0 else 0
                expected = 1 if (A + B + C) > 0 else 0
                ok = "OK" if logic == expected else "FAIL"
                correct += (logic == expected)
                print(f"{A:+3d} {B:+3d} {C:+3d} | {A+B+C:+8d} {P:+8.4f} | "
                      f"  {logic}   {ok}")
    print(f"\n{correct}/8 correct")
    return correct


for t in (0.30, 0.10, 0.05, 0.02):
    truth_table(t)
    print()
