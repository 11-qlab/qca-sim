"""Layer 2 diagnostic — the right invariant is CONCURRENCE, not F.

The state ZZ(pi/4)|++> after H_A ⊗ H_B is:
    (|00> + i|11>) / sqrt(2)
which is a Bell state, just not |Phi+>.  Fidelity with |Phi+>
is 0.5 by phase mismatch, but CONCURRENCE is 1.0 -- the state is
maximally entangled.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
H1 = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
kron = np.kron
sy2 = kron(sy, sy)


def concurrence(psi):
    psi = psi / np.linalg.norm(psi)
    return float(abs(psi.conj() @ sy2 @ psi.conj()))


def zz(theta):
    return np.diag(np.exp(1j*theta*np.array([1,-1,-1,1])))


# Input |++>
psi = np.ones(4, dtype=complex) / 2

# QCA-native entangler
psi = zz(np.pi/4) @ psi

print("Layer 2 diagnostic (corrected):\n")
print(f"  ZZ(pi/4)|++> state amplitudes:")
for lab, amp in zip(["|++>", "|+->", "|-+>", "|-->"], psi):
    print(f"    {lab}: {amp.real:+.4f} {amp.imag:+.4f}i")

# Now rotate to Z basis
psi_z = kron(H1, H1) @ psi
print(f"\n  After H_A ⊗ H_B rotation (Z basis):")
for lab, amp in zip(["|00>", "|01>", "|10>", "|11>"], psi_z):
    print(f"    {lab}: {amp.real:+.4f} {amp.imag:+.4f}i")

# Various fidelities
phi_plus = np.array([1,0,0,1], dtype=complex) / np.sqrt(2)
phi_plus_i = np.array([1,0,0,1j], dtype=complex) / np.sqrt(2)
F1 = abs(phi_plus.conj() @ psi_z)**2
F2 = abs(phi_plus_i.conj() @ psi_z)**2

print(f"\n  Fidelity with (|00>+|11>)/√2   : {F1:.4f}")
print(f"  Fidelity with (|00>+i|11>)/√2  : {F2:.4f}   <- the actual state")
print(f"  Concurrence (basis-independent): {concurrence(psi_z):.4f}")
print(f"  S(A) (reduced entropy)         :", end=" ")
m = psi_z.reshape(2, 2)
r = m @ m.conj().T
w = np.linalg.eigvalsh(r); w = w[w > 1e-12]
print(f"{-np.sum(w*np.log(w)):.4f}   (ideal ln 2 = 0.6931)")

print("\n  Conclusion:")
print("    QCA-native entangler produces a MAXIMALLY entangled")
print("    Bell state (C = 1, S = ln 2).  The specific Bell")
print("    variant is (|00>+i|11>)/√2, equivalent to |Phi+>")
print("    up to a single-qubit S gate.")
