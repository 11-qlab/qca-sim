"""Statevector simulation of the toric code ground state.

Adaptive: chooses the largest L that fits in available RAM.
L=2 ->  8 qubits,  256-dim       ~2 KB
L=3 -> 18 qubits,  256K-dim      ~2 MB
L=4 -> 32 qubits,  4G-dim        ~64 GB  (server only)
L=5 -> 50 qubits,  1P-dim        ~16 PB  (impossible)

For 50 qubits, use the CHP (Clifford) simulator in
examples/clifford_toric.py -- exact but O(n^2) memory.
"""
from __future__ import annotations
import numpy as np
import psutil
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import eigsh

I2 = csr_matrix(np.eye(2, dtype=complex))
X  = csr_matrix(np.array([[0,1],[1,0]], dtype=complex))
Z  = csr_matrix(np.array([[1,0],[0,-1]], dtype=complex))


def pauli_string(n, terms):
    op = csr_matrix(np.array([[1.0]], dtype=complex))
    for q in range(n):
        p = I2
        for (qq, pp) in terms:
            if qq == q:
                p = X if pp == "X" else Z
                break
        op = kron(op, p, format="csr")
    return op


def build_toric(L):
    def h(i, j): return (i%L)*L + (j%L)
    def v(i, j): return L*L + (i%L)*L + (j%L)
    plaqs = [[h(i,j), v(i,(j+1)%L), h((i+1)%L,j), v(i,j)]
             for i in range(L) for j in range(L)]
    verts = [[h(i,j), h(i,(j-1)%L), v(i,j), v((i-1)%L,j)]
             for i in range(L) for j in range(L)]
    n = 2*L*L
    H = csr_matrix((2**n, 2**n), dtype=complex)
    for p in plaqs: H -= pauli_string(n, [(q,"Z") for q in p])
    for s in verts: H -= pauli_string(n, [(q,"X") for q in s])
    return H, n, plaqs, verts


def available_ram_gb():
    return psutil.virtual_memory().available / (1024**3)


def max_L_for_ram():
    """Return the largest L whose statevector fits in available RAM."""
    ram = available_ram_gb() * 0.7   # leave 30% headroom
    for L in (6, 5, 4, 3, 2):
        n = 2 * L * L
        bytes_needed = (2 ** n) * 16 / (1024**3)   # complex128 = 16 bytes
        if bytes_needed < ram:
            return L, bytes_needed
    return 2, (2**8) * 16 / (1024**3)


if __name__ == "__main__":
    print("=== Statevector toric code ground state ===\n")
    print(f"Available RAM: {available_ram_gb():.1f} GB\n")

    L_max, est_gb = max_L_for_ram()
    print(f"Largest feasible L from RAM: L={L_max} (~{est_gb:.2f} GB)\n")

    for L in (2, 3, 4, 5):
        n = 2 * L * L
        bytes_needed = (2 ** n) * 16 / (1024**3)
        if bytes_needed > available_ram_gb() * 0.7:
            if bytes_needed > 1000:
                print(f"L={L} ({n} qubits): {bytes_needed:.0f} GB "
                      f"({bytes_needed/1024:.1f} TB) -- SKIP (impossible)")
            else:
                print(f"L={L} ({n} qubits): {bytes_needed:.0f} GB -- SKIP (RAM)")
            continue

        print(f"\nL={L} ({n} qubits): {bytes_needed:.4f} GB")
        H, n, plaqs, verts = build_toric(L)
        dim = 2 ** n

        if dim <= 4096:
            w, v = np.linalg.eigh(H.toarray())
        else:
            w, v = eigsh(H, k=1, which="SA")
        psi = v[:, 0]

        E0 = w[0]
        print(f"  Ground energy = {E0:.6f}")
        print(f"  |psi[0]|^2 = {abs(psi[0])**2:.6f}")

        # Verify some stabilizers
        n_ok = 0
        for plaq in plaqs:
            op = pauli_string(n, [(q, "Z") for q in plaq])
            ev = float(np.real(psi.conj() @ (op @ psi)))
            if abs(ev - 1.0) < 1e-6:
                n_ok += 1
        for star in verts:
            op = pauli_string(n, [(q, "X") for q in star])
            ev = float(np.real(psi.conj() @ (op @ psi)))
            if abs(ev - 1.0) < 1e-6:
                n_ok += 1
        print(f"  Stabilizers at +1: {n_ok}/{len(plaqs)+len(verts)}")

    print("\n" + "="*60)
    print("For L=5 (50 qubits), dense statevector needs 16 PB.")
    print("Use examples/clifford_toric.py -- exact, O(n^2) memory.")
    print("="*60)
