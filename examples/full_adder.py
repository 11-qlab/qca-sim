"""QCA 1-bit full adder built from 5 majority gates.

    Cout = MAJ(A, B, Cin)
    W1   = MAJ(A, B, ~Cin)
    W2   = MAJ(A, ~B, Cin)
    W3   = MAJ(~A, B, Cin)
    Sum  = MAJ(W1, W2, W3)

This is the canonical QCA adder topology.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def cell_response(driver, t):
    H, B = hamiltonian(t=t, V_diag=1.0, V_adj=0.2, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def maj(a, b, c, t, coupling=1.0):
    """QCA majority gate: inputs in {-1,+1}, output in [-1,+1]."""
    return cell_response(coupling * (a + b + c), t)


def to_bit(p):
    return 1 if p > 0 else 0


def full_adder(a_bit, b_bit, cin_bit, t):
    """Return (Sum_bit, Cout_bit) via 5 majority gates."""
    A   = +1 if a_bit  else -1
    B   = +1 if b_bit  else -1
    Cin = +1 if cin_bit else -1

    Cout = maj(A, B, Cin, t)
    W1   = maj(A, B, -Cin, t)
    W2   = maj(A, -B, Cin, t)
    W3   = maj(-A, B, Cin, t)
    Sum  = maj(W1, W2, W3, t)

    return to_bit(Sum), to_bit(Cout)


def truth_table(t):
    print(f"=== QCA full adder (t = {t:.2f}) ===\n")
    print(f"{'A':>3} {'B':>3} {'Cin':>4} | {'Sum':>4} {'Cout':>5} | "
          f"{'exp S':>6} {'exp C':>6} | ok")
    print("-" * 56)
    ok = 0
    for A in (0, 1):
        for B in (0, 1):
            for Cin in (0, 1):
                S, C = full_adder(A, B, Cin, t)
                total = A + B + Cin
                exp_S, exp_C = total % 2, total // 2
                good = (S == exp_S) and (C == exp_C)
                ok += good
                print(f"{A:>3} {B:>3} {Cin:>4} | {S:>4} {C:>5} | "
                      f"{exp_S:>6} {exp_C:>6} | {'OK' if good else 'FAIL'}")
    print(f"\n{ok}/8 correct\n")


for t in (0.30, 0.10, 0.05, 0.02):
    truth_table(t)
