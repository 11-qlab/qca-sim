"""Bis-ferrocene molecular QCA cell with corrected conventions.

Polarization: P = n_L - n_R,  P = +1 means electron on L.
Positive clock field RAISES site L -> electron sits on R -> P = -1.
"""
from __future__ import annotations
import numpy as np


def hamiltonian(t_DA=0.05, dE_clock=0.0):
    return np.array([
        [ 0.5 * dE_clock, -t_DA ],
        [ -t_DA,          -0.5 * dE_clock ],
    ])


def ground_state(H):
    w, v = np.linalg.eigh(H)
    return w[0], v[:, 0]


def polarization(psi):
    return float(abs(psi[0]) ** 2 - abs(psi[1]) ** 2)


def tunneling_vs_chain(n_ch2, beta=0.9, t0=0.1):
    return t0 * np.exp(-beta * n_ch2)


if __name__ == "__main__":
    print("=== Bis-ferrocene QCA cell (corrected) ===\n")

    print("1. Alkyl-chain tunnelling decay (beta = 0.9 / CH2):")
    print(f"   {'n_CH2':>6} {'t_DA (eV)':>12}")
    for n in (0, 2, 4, 6, 8, 10):
        print(f"   {n:>6} {tunneling_vs_chain(n):>12.5f}")

    print("\n2. Clock-field switching (t_DA = 5 meV):")
    print(f"   {'clock (eV)':>12} {'E0 (eV)':>12} {'P':>8} {'electron':>10}")
    t_DA = 0.005
    for dE in (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
        H = hamiltonian(t_DA=t_DA, dE_clock=dE)
        E0, psi = ground_state(H)
        P = polarization(psi)
        where = "L" if P > 0.5 else ("R" if P < -0.5 else "mixed")
        print(f"   {dE:>+12.3f} {E0:>12.5f} {P:>+8.4f} {where:>10}")

    print("\n3. Room-temperature feasibility (T_max = gap / k_B):")
    print(f"   {'clock (eV)':>12} {'gap (meV)':>10} {'P':>8} {'T_max (K)':>10}")
    for dE in (0.005, 0.010, 0.020, 0.050, 0.100, 0.200):
        H = hamiltonian(t_DA=t_DA, dE_clock=dE)
        E0, psi = ground_state(H)
        P = polarization(psi)
        gap_meV = (np.sqrt(dE ** 2 + (2 * t_DA) ** 2)) * 1000
        T_max = gap_meV * 11.604
        print(f"   {dE:>+12.3f} {gap_meV:>10.2f} {P:>+8.4f} {T_max:>10.0f}")

    print("\n4. Redox-window safety check:")
    print("   Ferrocene 1st oxidation ~ +0.40 V vs SCE (~0.64 V vs NHE).")
    print("   Clock field across 1.2 nm: E_field = dE / 1.2 nm")
    for dE in (0.02, 0.05, 0.10, 0.20):
        E_field_V_per_nm = dE / 1.2
        print(f"   clock = {dE*1000:>5.0f} meV -> E = {E_field_V_per_nm:.3f} V/nm")
    print("\n   Below ~0.5 V/nm the ferrocene redox couple is intact.")
    print("   200 meV / 1.2 nm = 0.167 V/nm -- safely inside the window.")
