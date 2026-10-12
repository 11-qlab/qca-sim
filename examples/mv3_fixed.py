"""Fault-tolerant MV3 majority gate (parallel voter topology).

Layout (7 cells):
    Inputs A, B, C  -> 3 pairwise majority cells mAB, mAC, mBC
                    -> device cell sees all 6 signals

    driver_device = w1*(A+B+C) + w2*(mAB + mAC + mBC)

With weights (w1, w2) = (1, 2), losing any single input or
pairwise cell still leaves a decisive majority.

Contrast with the serial version: out = maj(A, B, maj(A,B,C)).
Serial MV3 tolerates a stuck-at-+1 or stuck-at--1 fault, but
NOT a "stuck-at-0" (omission) fault, because the outer gate
then degenerates to a 2-input voter that cannot break ties.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def cell_response(driver, t):
    H, B = hamiltonian(t=t, V_diag=1.0, V_adj=0.2, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def maj(a, b, c, t):
    """3-input majority in {-1,+1}, returns P in [-1,+1]."""
    return cell_response(a + b + c, t)


def pairwise_maj(a, b, t):
    """2-input voter: maj(a, b, +1) is A AND B;
                   maj(a, b, -1) is A OR B.
       For an unbiased pairwise 'voter', we use maj(a, b, 0) -> sign(a+b)."""
    return cell_response(a + b, t)


def mv3_serial(a, b, c, t):
    """Serial chain: out = maj(A, B, maj(A,B,C))."""
    inner = maj(a, b, c, t)
    return maj(a, b, inner, t)


def mv3_parallel(a, b, c, t, w1=1.0, w2=2.0):
    """Parallel voter: device sees A+B+C plus pairwise majorities."""
    mAB = pairwise_maj(a, b, t)
    mAC = pairwise_maj(a, c, t)
    mBC = pairwise_maj(b, c, t)
    driver = w1 * (a + b + c) + w2 * (mAB + mAC + mBC)
    return cell_response(driver, t)


def apply_fault(signals, faulty_index, fault_value=0.0):
    """Set one element of the signal list to the fault value."""
    s = list(signals)
    s[faulty_index] = fault_value
    return s


def run_table(t=0.05):
    print(f"=== MV3 topologies, t = {t} ===\n")
    header = (f"{'A':>3} {'B':>3} {'C':>3} | "
              f"{'std':>4} {'ser':>4} {'par':>4} | "
              f"{'serF':>5} {'parF':>5} | {'exp':>4}")
    print(header)
    print("-" * len(header))

    ser_ok = par_ok = ser_fault_ok = par_fault_ok = 0

    for A in (-1, +1):
        for B in (-1, +1):
            for C in (-1, +1):
                exp = 1 if (A + B + C) > 0 else 0

                std  = 1 if maj(A, B, C, t) > 0 else 0
                ser  = 1 if mv3_serial(A, B, C, t) > 0 else 0
                par  = 1 if mv3_parallel(A, B, C, t) > 0 else 0

                # inject a stuck-at-0 fault into the FIRST inner cell
                # (for serial: the inner majority cell;
                #  for parallel: the first pairwise cell -> weight drop)
                ser_fault_driver = A + B + 0.0     # inner stuck at 0
                ser_f = 1 if cell_response(ser_fault_driver, t) > 0 else 0

                mAB = pairwise_maj(A, B, t)
                mAC = pairwise_maj(A, C, t)
                mBC = pairwise_maj(B, C, t)
                # omit mAB (set to 0)
                par_fault_driver = (A + B + C) + 2.0 * (0.0 + mAC + mBC)
                par_f = 1 if cell_response(par_fault_driver, t) > 0 else 0

                ser_ok       += (ser == exp)
                par_ok       += (par == exp)
                ser_fault_ok += (ser_f == exp)
                par_fault_ok += (par_f == exp)

                print(f"{A:+3d} {B:+3d} {C:+3d} | "
                      f"{std:>4} {ser:>4} {par:>4} | "
                      f"{ser_f:>5} {par_f:>5} | {exp:>4}")

    print()
    print(f"  healthy  : serial {ser_ok}/8   parallel {par_ok}/8")
    print(f"  faulted  : serial {ser_fault_ok}/8   parallel {par_fault_ok}/8")
    print()
    print("  serial   fails on ties when the inner cell is omitted.")
    print("  parallel survives because the two other pairwise cells")
    print("  still deliver a decisive majority to the device cell.")


run_table(t=0.05)
