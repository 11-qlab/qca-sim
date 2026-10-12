"""QCA Bell state with full 2-qubit tomography.

Prep: H(0); H(1); CZ(0,1); H(1)  ->  |Φ+> = (|00> + |11>)/√2

Tomography: 9 measurement settings in {X, Y, Z}^2.
For each setting (a, b), extract three Pauli expectations:
    <P_a ⊗ P_b>  (full 2-qubit)
    <P_a ⊗ I>    (marginal over qubit 1)
    <I ⊗ P_b>    (marginal over qubit 0)
Reconstruct: rho = (1/4) [ I⊗I + sum over P⊗Q of <PQ> P⊗Q ]
"""
import os, sys
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import DensityMatrix, concurrence as qk_conc
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False
    print("[warn] qiskit not installed.  pip install qiskit qiskit-aer")


PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def bell_with_tomography(a, b):
    """Prepare |Φ+> and rotate into basis (a, b) for measurement."""
    qc = QuantumCircuit(2, 2)
    # QCA-native Bell state: H(0); H(1); CZ; H(1)
    qc.h(0); qc.h(1)
    qc.cz(0, 1)
    qc.h(1)
    # Tomography rotation
    for q, base in ((0, a), (1, b)):
        if base == "X":
            qc.h(q)
        elif base == "Y":
            qc.sdg(q); qc.h(q)
        # Z = no-op (Z basis is the default computational basis)
    qc.measure([0, 1], [0, 1])
    return qc


def run_simulator(qc, shots=8192):
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    return sim.run(tqc, shots=shots).result().get_counts()


def run_real(qc, backend_name, shots=8192):
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError:
        print("[hardware] qiskit-ibm-runtime not installed")
        return None
    token = os.environ.get("IBMQ_TOKEN")
    if not token:
        print("[hardware] IBMQ_TOKEN not set")
        return None
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.backend(backend_name)
    tqc = transpile(qc, backend, optimization_level=3)
    sampler = SamplerV2(backend)
    job = sampler.run([tqc], shots=shots)
    print(f"[hardware] job id = {job.job_id()}")
    return job.result()[0].data.c.get_counts()


def expectation_from_counts(counts, shots, a, b):
    """Compute <P_a ⊗ P_b> from counts.

    Qiskit strings are q1 q0 (leftmost is highest qubit).
    a applies to qubit 0, b applies to qubit 1.
    """
    ev = 0.0
    for k, n in counts.items():
        q1 = int(k[0]); q0 = int(k[1])
        sign = 1
        if a != "I" and q0 == 1:
            sign *= -1
        if b != "I" and q1 == 1:
            sign *= -1
        ev += sign * n / shots
    return ev


def marginal_expectation(counts, shots, qubit, basis):
    """<P_basis> on `qubit`, marginalizing the other."""
    p0 = p1 = 0
    for k, n in counts.items():
        bit = int(k[1 - qubit])   # 0 -> q0, 1 -> q1
        if bit == 0:
            p0 += n
        else:
            p1 += n
    p0 /= shots; p1 /= shots
    return p0 - p1


def reconstruct_rho(tomo_counts, shots):
    """Full 2-qubit linear-inversion tomography.

    rho = (1/4) [ I + sum_{(P,Q) != (I,I)} <PQ> P ⊗ Q ]
    """
    rho = np.kron(PAULI["I"], PAULI["I"]) / 4.0

    # P⊗Q with P,Q ∈ {X,Y,Z} -- full 2-qubit from each setting
    for a in ("X", "Y", "Z"):
        for b in ("X", "Y", "Z"):
            ev = expectation_from_counts(tomo_counts[a + b], shots, a, b)
            rho += ev * np.kron(PAULI[a], PAULI[b]) / 4.0

    # P⊗I  -- use setting (P, Z), marginalize qubit 1
    for a in ("X", "Y", "Z"):
        ev = marginal_expectation(tomo_counts[a + "Z"], shots, 0, a)
        rho += ev * np.kron(PAULI[a], PAULI["I"]) / 4.0

    # I⊗Q  -- use setting (Z, Q), marginalize qubit 0
    for b in ("X", "Y", "Z"):
        ev = marginal_expectation(tomo_counts["Z" + b], shots, 1, b)
        rho += ev * np.kron(PAULI["I"], PAULI[b]) / 4.0

    return rho



def project_to_psd(rho):
    """Eigenvalue clip: keep only non-negative eigenvalues, re-trace to 1."""
    w, v = np.linalg.eigh(rho)
    w = np.clip(w.real, 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()           # renormalize trace to 1
    rho_psd = (v * w) @ v.conj().T
    # symmetrize and enforce Hermiticity
    return (rho_psd + rho_psd.conj().T) / 2


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def fidelity_bell(rho):
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    psi = bell.reshape(-1, 1)
    return float(np.real((psi.conj().T @ rho @ psi)[0, 0]))


if __name__ == "__main__":
    if not HAVE_QISKIT:
        sys.exit(1)

    print("=== QCA Bell state with full tomography ===\n")
    backend_name = sys.argv[1] if len(sys.argv) > 1 else None
    shots = 8192

    bases = [a + b for a in ("X", "Y", "Z") for b in ("X", "Y", "Z")]
    tomo_counts = {}
    for bas in bases:
        qc = bell_with_tomography(bas[0], bas[1])
        counts = run_real(qc, backend_name, shots) if backend_name \
                 else run_simulator(qc, shots)
        if counts is None:
            sys.exit(1)
        tomo_counts[bas] = counts
        if bas in ("ZZ", "XX", "YY"):
            print(f"  [{bas}]  counts = {dict(sorted(counts.items()))}")

    rho = reconstruct_rho(tomo_counts, shots)
    print("\nReconstructed density matrix:")
    for row in rho:
        print("  " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}i" for x in row))

    print(f"\n  trace(rho)      = {np.real(np.trace(rho)):.4f}")
    print(f"  purity          = {purity(rho):.4f}   (ideal 1.0)")
    print(f"  fidelity |Phi+> = {fidelity_bell(rho):.4f}   (ideal 1.0)")
    rho_psd = project_to_psd(rho)
    try:
        C = float(qk_conc(DensityMatrix(rho_psd)))
        print(f"  concurrence      = {C:.4f}   (ideal 1.0)")
    except Exception as e:
        print(f"  concurrence computation failed: {e}")
