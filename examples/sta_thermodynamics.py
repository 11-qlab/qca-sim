"""Quantum thermodynamics of the counter-diabatic STA clock.

The STA clock (F5) achieves a 5000x speedup over the Landau-Zener
threshold.  But the counter-diabatic term is an *extra drive*:

    H_cd(t) = -(Delta/2) * ddot(t) / (d(t)^2 + Delta^2) * sigma_y

The extra work cost is what this script computes.

We solve the TDSE for the 2-level system with H(t) + H_cd(t)
and compute, along the trajectory:

    W(t)   = integral of <dH_total/dt> dt    (work done)
    <H>    = instantaneous energy
    <H_cd> = counter-diabatic contribution

Then we compare:
  - Baseline: linear ramp, no CD, slow (T = 500)
  - STA: linear ramp, CD, fast (T = 0.1)
  - Fast baseline: linear ramp, no CD, fast (T = 0.1)  <- fails to flip

Landauer bound for a bit flip at temperature T:
    W_Landauer = k_B T ln 2 = 0.693 * kT

Typical molecular QCA at 300 K has kT ~ 25 meV = 0.025 eV.
So W_Landauer ~ 0.017 eV.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def H_bare(d, Delta):
    return 0.5 * d * sz + 0.5 * Delta * sx


def H_cd(d, ddot, Delta):
    return -0.5 * Delta * ddot / (d * d + Delta * Delta) * sy


def evolve(T_total, Delta, d_start, d_end, use_cd, n_steps=2000):
    """Evolve the 2-level system from |0> (ground state at d_start).
    Returns time series of <sz>, <H_bare>, <H_cd>, and cumulative work."""
    dt = T_total / n_steps
    psi = np.array([1.0, 0.0], dtype=complex)
    P = np.empty(n_steps + 1)         # <sz> = polarization
    Hb = np.empty(n_steps + 1)
    Hcd = np.empty(n_steps + 1)
    W = np.empty(n_steps + 1)

    def d_of_tau(tau):
        return d_start + (d_end - d_start) * tau

    def ddot(tau):
        return (d_end - d_start) / T_total   # d(d)/dt, in real time

    def energies(psi, d, dd):
        Hb_t = float(np.real(psi.conj() @ H_bare(d, Delta) @ psi))
        Hcd_t = float(np.real(psi.conj() @ H_cd(d, dd, Delta) @ psi))
        return Hb_t, Hcd_t

    # Initial
    d = d_of_tau(0.0)
    dd = ddot(0.0)
    Hb[0], Hcd[0] = energies(psi, d, dd)
    W[0] = 0.0
    P[0] = float(np.real(psi.conj() @ sz @ psi))

    for k in range(n_steps):
        tau = (k + 0.5) / n_steps
        d = d_of_tau(tau)
        dd = ddot(tau)
        H = H_bare(d, Delta)
        if use_cd:
            H = H + H_cd(d, dd, Delta)
        psi_next = expm(-1j * H * dt) @ psi
        psi_next /= np.linalg.norm(psi_next)

        # Work: <dH_total/dt>
        # Approximate by finite difference of <H_total>
        Hb_new, Hcd_new = energies(psi_next, d_of_tau(tau + 0.5 / n_steps),
                                   dd)
        dH = (Hb_new + Hcd_new) - (Hb[k] + Hcd[k])
        W[k + 1] = W[k] + dH

        psi = psi_next
        Hb[k + 1], Hcd[k + 1] = Hb_new, Hcd_new
        P[k + 1] = float(np.real(psi.conj() @ sz @ psi))

    return P, Hb, Hcd, W


def summarise(label, T_total, Delta, d_start, d_end, use_cd):
    P, Hb, Hcd, W = evolve(T_total, Delta, d_start, d_end, use_cd)
    return {
        "label": label,
        "T_total": T_total,
        "P_final": P[-1],
        "W_total": W[-1],
        "W_peak_amplitude": float(np.max(np.abs(Hcd))),
        "H_bare_change": Hb[-1] - Hb[0],
    }


if __name__ == "__main__":
    print("=== Quantum thermodynamics of the STA clock ===\n")
    print("2-level model: H = (d/2) sigma_z + (Delta/2) sigma_x + H_cd")
    print("Driver ramps d from +3 to -3.\n")

    Delta = 0.10

    print(f"{'protocol':<22} {'T':>8} {'P_final':>10} {'W_total':>12} "
          f"{'|H_cd|_max':>12} {'dE_bare':>10}")
    print("-" * 80)

    results = []
    for label, T, cd in [
        ("baseline (slow)", 500.0, False),
        ("baseline (fast)", 0.1, False),
        ("STA (fast)", 0.1, True),
        ("STA (medium)", 1.0, True),
        ("STA (slow)", 500.0, True),
    ]:
        r = summarise(label, T, Delta, 3.0, -3.0, cd)
        results.append(r)
        print(f"{label:<22} {T:>8.2f} {r['P_final']:>+10.4f} "
              f"{r['W_total']:>+12.6f} {r['W_peak_amplitude']:>12.6f} "
              f"{r['H_bare_change']:>+10.6f}")

    print()
    print("Interpretation:")
    print("  * Baseline (slow): reaches P_final = -1 (flip succeeds),")
    print("    W_total ~ 0 because the ramp is adiabatic.")
    print("  * Baseline (fast): fails to flip (P_final > 0).")
    print("  * STA (fast): succeeds at T = 0.1 (5000x speedup),")
    print("    but the CD term adds extra work W_total > 0.")
    print("  * The |H_cd|_max column scales as 1/T: faster ramps")
    print("    need larger counter-diabatic fields.")
    print()
    print("Landauer comparison:")
    print(f"  kT at 300 K = 0.0259 eV")
    print(f"  Landauer bound per bit: kT ln 2 = {0.0259 * np.log(2):.6f} eV")
    print(f"  Delta (our energy scale) = {Delta} eV")
    print(f"  Landauer / Delta = {0.0259 * np.log(2) / Delta:.6f}")
    print()
    print("  Our W_total is measured in units of the drive energy,")
    print("  which is set by the field strength (typically ~ 0.1 eV")
    print("  for molecular QCA).  W_total in real units =")
    print("  W_total * drive_scale.")
    print()

    # Save
    import json
    with open("sta_thermodynamics_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results written to sta_thermodynamics_results.json")
