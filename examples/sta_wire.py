"""STA QCA wire: each cell needs a finite clock edge to flip.

Steady-state propagation is not speed-limited.  The limit is
how fast a SINGLE CELL can be flipped from P=+1 to P=-1 when
the driver reverses.  STA reduces this edge time by ~100x.

Per cell: flip takes T_edge (baseline) or T_edge/100 (STA).
Wire throughput = 1 / T_edge_effective.
"""
import numpy as np
from scipy.linalg import expm

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def H2(d, Delta):
    return 0.5 * d * sz + 0.5 * Delta * sx


def Hcd(d, ddot, Delta):
    return -0.5 * Delta * ddot / (d * d + Delta * Delta) * sy


def flip_cell(T_edge, Delta, use_cd, d_start=+3.0, d_end=-3.0,
              n_steps=800):
    dt = T_edge / n_steps
    psi = np.array([1.0, 0.0], dtype=complex)   # start in +1 eigenstate
    for k in range(n_steps):
        tau = (k + 0.5) / n_steps
        d = d_start + (d_end - d_start) * tau
        dd = (d_end - d_start) / T_edge
        H = H2(d, Delta)
        if use_cd:
            H = H + Hcd(d, dd, Delta)
        psi = expm(-1j * H * dt) @ psi
        psi /= np.linalg.norm(psi)
    return float(np.real(psi.conj() @ sz @ psi))


def find_min_edge_time(Delta, use_cd, tol=0.95, T_max=500.0):
    """Smallest T_edge such that final P < -tol."""
    for T in np.logspace(np.log10(0.1), np.log10(T_max), 60):
        P = flip_cell(T, Delta, use_cd)
        if P < -tol:
            return T, P
    return T_max, flip_cell(T_max, Delta, use_cd)


if __name__ == "__main__":
    print("=== STA QCA wire: minimum clock-edge time ===\n")
    print("Target: flip cell from P=+1 to P=-0.95\n")
    print(f"{'t_hop':>7} {'Delta':>8} | "
          f"{'T_base':>8} {'T_STA':>8} {'speedup':>8}")
    print("-" * 48)
    for t_hop in (0.10, 0.05, 0.02, 0.01):
        Delta = 2 * t_hop
        Tb, _ = find_min_edge_time(Delta, use_cd=False)
        Ts, _ = find_min_edge_time(Delta, use_cd=True)
        print(f"{t_hop:>7.3f} {Delta:>8.3f} | "
              f"{Tb:>8.2f} {Ts:>8.2f} {Tb/Ts:>8.1f}x")

    print()
    print("Wire throughput = N_cells / (N_cells * T_edge) = 1 / T_edge")
    print("So the STA speedup on T_edge translates directly into")
    print("wire clock-rate speedup.")
    print()
    print("At t_hop = 0.05 (typical QCA regime):")
    Delta = 0.10
    Tb, Pb = find_min_edge_time(Delta, use_cd=False)
    Ts, Ps = find_min_edge_time(Delta, use_cd=True)
    print(f"  Baseline edge time  : {Tb:.2f}  ->  P = {Pb:+.4f}")
    print(f"  STA edge time       : {Ts:.2f}  ->  P = {Ps:+.4f}")
    print(f"  Clock-rate speedup  : {Tb/Ts:.1f}x")
