"""Decoherence of QCA polarizations (corrected).

At T = 0, the Ohmic coherence function is
    L(t) = (1 + (omega_c t)^2)^(-alpha/2)
which gives an EXACT closed-form T2:
    T2 = sqrt(exp(2/alpha) - 1) / omega_c

This is superexponentially large for small alpha because the
Ohmic bath has no low-frequency cutoff at T=0.  At finite
temperature, the Markovian result T2 = 1/(2 pi alpha kT)
is the physically relevant quantity.
"""
import numpy as np


def T2_zero_T_analytic(alpha, omega_c):
    """Exact closed form for Ohmic T=0."""
    return np.sqrt(np.exp(2.0 / alpha) - 1) / omega_c


def T2_markovian(alpha, kT):
    """High-T Markovian limit: T2 = 1 / (2 pi alpha kT)."""
    return 1.0 / (2 * np.pi * alpha * kT)


def T2_crossover(alpha, omega_c, kT):
    """Full numeric T2 from the Ohmic integral."""
    def L(t):
        wmax = 20 * omega_c
        w = np.linspace(1e-6, wmax, 400)
        J = alpha * w * np.exp(-w / omega_c)
        integ = (J / w**2) * (1 - np.cos(w * t)) / np.tanh(w / (2 * kT))
        return np.exp(-np.trapezoid(integ, w))

    lo, hi = 1e-6, 1e4 / omega_c
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if L(mid) > 1 / np.e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("=== Decoherence of QCA polarizations ===\n")
    omega_c = 10.0  # THz
    kT_300K = 300 * 0.0208
    kT_4K   = 4 * 0.0208

    print("Ohmic T=0 T2 (analytic, superexponential):")
    print(f"  {'alpha':>6} {'T2 (ps)':>18} {'log10(T2/ps)':>14}")
    print("  " + "-" * 40)
    for alpha in (0.01, 0.05, 0.10, 0.20, 0.50):
        t2 = T2_zero_T_analytic(alpha, omega_c)
        print(f"  {alpha:>6.2f} {t2:>18.3e} {np.log10(t2):>14.1f}")

    print("\n  -> T=0 T2 is astronomically long (no IR cutoff).")
    print("     In practice, finite T dominates below.")

    print(f"\nMarkovian limit at finite T (kT in THz):")
    print(f"  kT@300K = {kT_300K:.2f} THz, kT@4K = {kT_4K:.4f} THz\n")
    print(f"  {'alpha':>6} {'T2@4K (ps)':>14} {'T2@300K (ps)':>16}")
    print("  " + "-" * 40)
    for alpha in (0.001, 0.005, 0.01, 0.05, 0.1):
        t2_4 = T2_markovian(alpha, kT_4K)
        t2_300 = T2_markovian(alpha, kT_300K)
        print(f"  {alpha:>6.3f} {t2_4:>14.4f} {t2_300:>16.4f}")

    print("\n  -> Markovian T2 scales as 1/(alpha * T).")
    print("     At 300 K, alpha=0.01:  T2 = 2.55 ps")
    print("     At   4 K, alpha=0.01:  T2 = 191 ps")

    print("\nGHZ coherence under local dephasing (T2_GHZ = T2_qubit / N):")
    t2_qubit = T2_markovian(0.01, kT_300K)
    print(f"  Single-qubit T2 (alpha=0.01, 300K) = {t2_qubit:.3f} ps")
    print(f"  {'N':>4} {'T2_GHZ (ps)':>14} {'status':>16}")
    print("  " + "-" * 38)
    for N in (2, 3, 5, 9, 17):
        t2 = t2_qubit / N
        status = "OK for 100 fs clock" if t2 > 0.1 else "SHORTER than clock"
        print(f"  {N:>4} {t2:>14.4f} {status:>16}")

    print("\nConclusions:")
    print("  1. QCA clock cycle (~100 fs) is FASTER than T2 for small N.")
    print("  2. GHZ coherence scales as 1/N -> surface code needed for N > 5.")
    print("  3. Surface code threshold (p_th = 1e-2) makes logical T2")
    print("     longer than physical T2 by ~1/p_L, potentially 100-1000x.")
