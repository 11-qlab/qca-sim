"""Logical CNOT between distance-3 patches (corrected)."""
import numpy as np


XA = np.array([[0,0,1,0],
               [0,0,0,1],
               [1,0,0,0],
               [0,1,0,0]], dtype=complex)
XB = np.array([[0,1,0,0],
               [1,0,0,0],
               [0,0,0,1],
               [0,0,1,0]], dtype=complex)
ZA = np.diag([1,1,-1,-1]).astype(complex)
ZB = np.diag([1,-1,1,-1]).astype(complex)
IA = np.eye(4, dtype=complex)


def logical_state(a, b):
    psi = np.zeros(4, dtype=complex)
    psi[a * 2 + b] = 1.0
    return psi


def logical_H_A(psi):
    """Logical Hadamard on qubit A."""
    out = psi.copy()
    # H swaps X_A and Z_A; on the |00>,|01>,|10>,|11> basis:
    #   (|0>+|1>)/√2  is the +1 X_A eigenstate
    #   |0> and |1> are Z_A eigenstates
    # Applying H_A: |0> -> (|0>+|1>)/√2, |1> -> (|0>-|1>)/√2
    new = np.zeros_like(psi)
    new[0] = (psi[0] + psi[2]) / np.sqrt(2)
    new[1] = (psi[1] + psi[3]) / np.sqrt(2)
    new[2] = (psi[0] - psi[2]) / np.sqrt(2)
    new[3] = (psi[1] - psi[3]) / np.sqrt(2)
    return new


def logical_CNOT(psi):
    """CNOT_A_to_B.  A = control, B = target."""
    return np.array([psi[0], psi[1], psi[3], psi[2]], dtype=complex)


def joint_exp(psi, op):
    """<psi| op |psi>."""
    return float(np.real(psi.conj() @ op @ psi))


if __name__ == "__main__":
    print("=== Logical CNOT between distance-3 patches ===\n")

    # Prep logical Bell: H_A on |00>_L, then CNOT
    psi = logical_state(0, 0)
    print("Start: |00>_L")
    psi = logical_H_A(psi)
    print(f"After H_A: <X_A> = {joint_exp(psi, XA):+.4f}, "
          f"<Z_A> = {joint_exp(psi, ZA):+.4f}")
    psi = logical_CNOT(psi)
    print("After CNOT_A->B:")
    print(f"  <Z_A Z_B> = {joint_exp(psi, ZA @ ZB):+.4f}   (ideal +1)")
    print(f"  <X_A X_B> = {joint_exp(psi, XA @ XB):+.4f}   (ideal +1)")
    print(f"  <Z_A>     = {joint_exp(psi, ZA):+.4f}   (ideal  0)")
    print(f"  <Z_B>     = {joint_exp(psi, ZB):+.4f}   (ideal  0)")

    print("\nTruth table (logical basis):")
    print(f"  {'in_L':>6}  {'out_L':>6}")
    for a, b in [(0,0), (0,1), (1,0), (1,1)]:
        psi = logical_state(a, b)
        psi = logical_CNOT(psi)
        idx = int(np.argmax(np.abs(psi)))
        print(f"  |{a}{b}>_L -> |{idx//2}{idx%2}>_L")

    print("\nLogical error rate after n_cycles=100 (p_th=1e-2, d=3):")
    print(f"  {'p_phys':>10} {'p_logical':>12} {'cycles to fail':>16}")
    print("  " + "-" * 42)
    p_th, d = 1e-2, 3
    for p in (5e-3, 1e-3, 5e-4, 1e-4, 1e-5):
        pL = (p / p_th) ** ((d+1)/2) * 100
        print(f"  {p:>10.1e} {pL:>12.2e} {1/max(pL,1e-12):>16.0f}")
