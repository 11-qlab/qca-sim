"""QCA cells as native entangling gates.

Observation: two ferromagnetically coupled QCA cells under a
transverse tunnelling field have a Bell-like ground state.
Applying the ZZ evolution for the right duration produces a
maximally entangled state from a product input.

H_coupling = -J sz_A sz_B
U(t)       = exp(i J t sz_A sz_B)
C(Jt)      = |sin(2 J t)|     (for the |+x +x> input)

Max concurrence at Jt = pi/4.  This is CNOT-equivalent up to
local pulses -- QCA coupling is a native 2-qubit entangler.
"""
import numpy as np
from itertools import combinations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
kron = np.kron
sy2 = kron(sy, sy)


def concurrence(psi):
    """Wootters concurrence for a pure 2-qubit state."""
    psi = psi / np.linalg.norm(psi)
    return float(abs(psi.conj() @ sy2 @ psi.conj()))


def reduced_A_entropy(psi):
    """S(A) for a 2-qubit pure state."""
    psi = psi / np.linalg.norm(psi)
    psi_mat = psi.reshape(2, 2)
    rho_A = psi_mat @ psi_mat.conj().T
    w = np.linalg.eigvalsh(rho_A)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log(w)))


def zz_evolve(J, t):
    """U = exp(i J t sz sz) on the 2-cell space."""
    phases = np.array([np.exp(1j * J * t), np.exp(-1j * J * t),
                       np.exp(-1j * J * t), np.exp(1j * J * t)])
    return np.diag(phases)


def H_A():
    return (kron(np.array([[1, 1], [1, -1]]) / np.sqrt(2), I2))


def H_B():
    return (kron(I2, np.array([[1, 1], [1, -1]]) / np.sqrt(2)))


if __name__ == "__main__":
    print("=== Native QCA entangling gate ===\n")

    # --- 1. Concurrence vs Jt from the |+x +x> input ---
    print("1. Concurrence C(t) from input |+x, +x>:\n")
    print(f"   {'Jt':>8} {'C':>8} {'S(A)':>8}")
    print("   " + "-" * 26)
    psi0 = np.ones(4, dtype=complex) / 2
    for Jt in (0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2,
               3*np.pi/4, np.pi):
        psi = zz_evolve(1.0, Jt) @ psi0
        C = concurrence(psi)
        S = reduced_A_entropy(psi)
        print(f"   {Jt:>8.4f} {C:>8.4f} {S:>8.4f}")

    # --- 2. Bell state at Jt = pi/4 ---
    print("\n2. Bell state at Jt = pi/4:")
    psi = zz_evolve(1.0, np.pi/4) @ psi0
    psi /= np.linalg.norm(psi)
    labels = ["|++>", "|+->", "|-+>", "|-->"]
    for lab, amp in zip(labels, psi):
        print(f"   {lab}:  {amp.real:+.4f} {amp.imag:+.4f}i")
    print(f"   C = {concurrence(psi):.4f}   S(A) = {reduced_A_entropy(psi):.4f}")

    # --- 3. CNOT construction ---
    print("\n3. CNOT from QCA native gates (H_B . ZZ(pi/4) . H_B):")
    print("   Truth table on the computational basis:")
    print(f"   {'in':>6} {'out':>6}")
    for i, lab in enumerate(["|00>", "|01>", "|10>", "|11>"]):
        psi_in = np.zeros(4, dtype=complex); psi_in[i] = 1.0
        psi_out = H_B() @ zz_evolve(1.0, np.pi/4) @ H_B() @ psi_in
        # reduce to bit string
        idx = np.argmax(np.abs(psi_out))
        bit = format(idx, "02b")
        print(f"   {lab:>6} ->|{bit}>   (C = {concurrence(psi_out):.4f})")
