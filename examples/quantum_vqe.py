"""VQE for the QCA cell ground state.

Works with qiskit if installed; otherwise falls back to a
pure-numpy exact diagonalization so the physics is testable
without any quantum SDK.

Qubit encoding (Jordan-Wigner, 4 spin-orbitals = 4 qubits):
    |n0 n1 n2 n3>  ->  |q3 q2 q1 q0>
    2-electron subspace = Hamming weight 2 states.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import SPSA
    from qiskit.circuit.library import TwoLocal
    from qiskit.opflow import PauliSumOp
    from qiskit.quantum_info import SparsePauliOp
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False


# --- Hamiltonian (same as cell_hubbard.py, but as an operator) ------
def hubbard_terms(t=0.3, V_diag=1.0, V_adj=0.2, driver=0.0):
    """Return (diag_energies, off_diag_hoppings) on the
    6-dimensional 2-electron Hilbert space.

    Basis index order matches cell_hubbard.basis().
    """
    B = list(combinations(range(4), 2))
    idx = {s: i for i, s in enumerate(B)}
    diag = np.zeros(len(B))
    for i, (a, b) in enumerate(B):
        e = 0.0
        if (a, b) in [(0, 2), (1, 3)]:
            e += V_diag
        if (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            e += V_adj
        e -= driver * ((a in (0, 2)) + (b in (0, 2))
                       - (a in (1, 3)) - (b in (1, 3))) / 2.0
        diag[i] = e

    hops = []
    ADJ = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, (a, b) in enumerate(B):
        for (x, y) in ADJ + [(q, p) for (p, q) in ADJ]:
            if x in (a, b) and y not in (a, b):
                new = tuple(sorted([s for s in (a, b) if s != x] + [y]))
                j = idx[new]
                hops.append((i, j, -t))
    return np.array(B), diag, hops


def exact_ground(B, diag, hops):
    H = np.diag(diag).astype(float)
    for i, j, v in hops:
        H[i, j] += v
    w, v = np.linalg.eigh(H)
    return w[0], v[:, 0]


# --- VQE on a quantum simulator ------------------------------------
def build_qiskit_hamiltonian(diag, hops, n_qubits=4):
    """Build the QCA cell Hamiltonian as a SparsePauliOp.

    The 6-state, 2-electron subspace is embedded in the 16-dim
    full Fock space.  Only Hamming-weight-2 states are used;
    the other 10 states are given a large positive penalty.
    """
    from qiskit.quantum_info import SparsePauliOp

    # Build the 16x16 matrix in the occupation basis
    H = np.zeros((16, 16))
    PENALTY = 5.0
    for k in range(16):
        if bin(k).count("1") != 2:
            H[k, k] = PENALTY

    B = list(combinations(range(4), 2))
    def occ_to_int(occ):
        n = 0
        for s in occ:
            n |= 1 << (3 - s)
        return n

    # Diagonal
    for i, (a, b) in enumerate(B):
        e = diag[i]
        H[occ_to_int((a, b)), occ_to_int((a, b))] = e

    # Off-diagonal hopping
    for i, j, v in hops:
        a, b = B[i]
        c, d = B[j]
        H[occ_to_int((a, b)), occ_to_int((c, d))] += v

    # Convert dense Hermitian matrix to Pauli sum
    labels = []
    coeffs = []
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    P = {"I": I, "X": X, "Y": Y, "Z": Z}

    for pauli_str in ("".join(p) for p in __import__("itertools").product("IXYZ", repeat=4)):
        M = np.array([[1.0]])
        for ch in pauli_str:
            M = np.kron(M, P[ch])
        c = np.trace(M @ H) / 16.0
        if abs(c) > 1e-10:
            labels.append(pauli_str)
            coeffs.append(c.real)

    return SparsePauliOp(labels, coeffs)


def run_vqe(driver=0.0, t=0.3):
    B, diag, hops = hubbard_terms(t=t, driver=driver)
    E0_exact, psi_exact = exact_ground(B, diag, hops)
    P_exact = 0.0
    for c, (a, b) in zip(np.abs(psi_exact) ** 2, B):
        P_exact += c * ((a in (0, 2)) + (b in (0, 2))
                        - (a in (1, 3)) - (b in (1, 3))) / 2.0

    print(f"[classical exact] driver={driver:+.2f}  E0 = {E0_exact:.6f}  P = {P_exact:+.4f}")

    if HAVE_QISKIT:
        print("[qiskit] qiskit detected -- running VQE on Aer simulator")
        H_op = build_qiskit_hamiltonian(diag, hops)
        ansatz = TwoLocal(4, ["ry"], "cx", reps=2, entanglement="linear")
        optimizer = SPSA(maxiter=200)
        vqe = VQE(ansatz, optimizer, quantum_instance=Aer.get_backend("aer_simulator_statevector"))
        result = vqe.compute_minimum_eigenvalue(H_op)
        print(f"[VQE]       driver={driver:+.2f}  E0 = {result.eigenvalue.real:.6f}")
    else:
        print("[qiskit] not installed -- install with: pip install qiskit qiskit-aer")
        print("         skipping VQE run; classical result shown above.")


if __name__ == "__main__":
    print("=== QCA cell ground state: exact vs VQE ===\n")
    for d in (-2.0, -1.0, 0.0, 1.0, 2.0):
        run_vqe(driver=d, t=0.3)
        print()
