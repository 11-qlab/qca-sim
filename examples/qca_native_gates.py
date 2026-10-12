"""QCA cells as native entangling gates (final, corrected)."""
import numpy as np

I2 = np.eye(2, dtype=complex)
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
kron = np.kron
sy = np.array([[0, -1j], [1j, 0]])
sy2 = kron(sy, sy)


def concurrence(psi):
    psi = psi / np.linalg.norm(psi)
    return float(abs(psi.conj() @ sy2 @ psi.conj()))


def reduced_A_entropy(psi):
    psi = psi / np.linalg.norm(psi)
    m = psi.reshape(2, 2)
    r = m @ m.conj().T
    w = np.linalg.eigvalsh(r); w = w[w > 1e-12]
    return float(-np.sum(w * np.log(w)))


def zz(theta):
    """exp(i * theta * sz_A sz_B)."""
    return np.diag(np.exp(1j * theta * np.array([1, -1, -1, 1])))


def CNOT():
    """Control = qubit A, target = qubit B. Basis |qA qB>."""
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=complex)


def CZ():
    return np.diag([1, 1, 1, -1]).astype(complex)


def HB():
    return kron(I2, H1)


def decompose_cz_from_zz():
    """Find single-qubit Rz rotations + ZZ(pi/4) that reproduce CZ.

    We solve numerically: for (alpha, beta) in a small grid, test if
        (Rz_A(alpha) ⊗ Rz_B(beta)) · ZZ(pi/4)
    equals CZ up to global phase.
    """
    def Rz(phi):
        return np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])

    for alpha in np.linspace(-np.pi, np.pi, 25):
        for beta in np.linspace(-np.pi, np.pi, 25):
            M = kron(Rz(alpha), Rz(beta)) @ zz(np.pi/4)
            # normalize to first element
            if abs(M[0, 0]) < 1e-9:
                continue
            Mn = M / M[0, 0]
            if np.allclose(Mn, CZ(), atol=1e-6):
                return (alpha, beta)
    return None


if __name__ == "__main__":
    print("=== Native QCA entangling gate ===\n")

    print("1. Concurrence C(Jt) from input |+x,+x>:")
    print(f"   {'Jt':>8} {'C':>8} {'S(A)':>8}")
    print("   " + "-" * 26)
    psi0 = np.ones(4, dtype=complex) / 2
    for Jt in (0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 3*np.pi/4, np.pi):
        psi = zz(Jt) @ psi0
        print(f"   {Jt:>8.4f} {concurrence(psi):>8.4f} {reduced_A_entropy(psi):>8.4f}")

    print("\n2. Bell state at Jt = π/4:")
    psi = zz(np.pi/4) @ psi0
    for lab, amp in zip(["|++>", "|+->", "|-+>", "|-->"], psi):
        print(f"   {lab}: {amp.real:+.4f} {amp.imag:+.4f}i")
    print(f"   C = {concurrence(psi):.4f}   S(A) = {reduced_A_entropy(psi):.4f}")

    print("\n3. CNOT truth table (direct construction):")
    print(f"   {'in':>6}  {'out':>6}  {'expected':>8}  ok")
    all_ok = True
    for i, lab in enumerate(["|00>", "|01>", "|10>", "|11>"]):
        psi_in = np.zeros(4, dtype=complex); psi_in[i] = 1.0
        psi_out = CNOT() @ psi_in
        idx = int(np.argmax(np.abs(psi_out)))
        exp = [0, 1, 3, 2][i]
        ok = idx == exp
        all_ok = all_ok and ok
        print(f"   {lab:>6}  |{format(idx,'02b')}>  |{format(exp,'02b')}>    "
              f"{'OK' if ok else 'FAIL'}")
    print(f"   all rows: {'PASS' if all_ok else 'FAIL'}")

    print("\n4. Verify CNOT = (I⊗H) · CZ · (I⊗H):")
    U_id = HB() @ CZ() @ HB()
    match = np.allclose(U_id, CNOT())
    print(f"   matches CNOT: {match}")

    print("\n5. QCA-native decomposition: CZ = Rz_A(α) Rz_B(β) · ZZ(π/4)")
    result = decompose_cz_from_zz()
    if result is None:
        print("   no (α, β) found in the searched grid")
    else:
        alpha, beta = result
        print(f"   α = {alpha:+.4f}, β = {beta:+.4f} radians")

    print("\n6. Bell state prepared by CNOT from |+0>:")
    psi_in = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    psi_bell = CNOT() @ psi_in
    print(f"   C = {concurrence(psi_bell):.4f}   S(A) = {reduced_A_entropy(psi_bell):.4f}")

    print("\nConclusion:")
    print("  QCA-native coupling σz⊗σz (ZZ(π/4)) is LOCALLY EQUIVALENT")
    print("  to CZ, and CZ is EQUIVALENT to CNOT up to single-qubit H.")
    print("  Every QCA wire is a chain of native entangling gates.")
