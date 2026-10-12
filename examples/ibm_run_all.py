"""Submit Bell + 3-qubit GHZ + full tomography to IBM Quantum.

Usage:
    pip install qiskit qiskit-aer qiskit-ibm-runtime
    export IBMQ_TOKEN=<token>
    python3 examples/ibm_run_all.py ibm_brisbane
"""
import os, sys
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import DensityMatrix, concurrence as qk_conc
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False
    print("[warn] qiskit not installed")


PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def bell_prep(meas_A="Z", meas_B="Z"):
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.h(1); qc.cz(0, 1); qc.h(1)
    if meas_A == "X": qc.h(0)
    elif meas_A == "Y": qc.sdg(0); qc.h(0)
    if meas_B == "X": qc.h(1)
    elif meas_B == "Y": qc.sdg(1); qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc


def ghz_prep(meas=None):
    """GHZ: (|000> + |111>)/√2 from QCA-native CZ gates."""
    qc = QuantumCircuit(3, 3)
    qc.h(0); qc.h(1); qc.h(2)
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.h(0)  # disentangle into GHZ
    if meas:
        for q, b in enumerate(meas):
            if b == "X": qc.h(q)
            elif b == "Y": qc.sdg(q); qc.h(q)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def run(qc, backend_name=None, shots=8192):
    if backend_name is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        return sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    token = os.environ.get("IBMQ_TOKEN")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.backend(backend_name)
    tqc = transpile(qc, backend, optimization_level=3)
    sampler = SamplerV2(backend)
    job = sampler.run([tqc], shots=shots)
    print(f"  [hw] job {job.job_id()} submitted")
    return job.result()[0].data.c.get_counts()


def parities(counts, shots, n=2):
    """Return dict of P(bitstring) for a 2-qubit measurement."""
    return {format(i, f"0{n}b"): counts.get(format(i, f"0{n}b"), 0) / shots
            for i in range(2**n)}


def main():
    if not HAVE_QISKIT:
        return
    backend = sys.argv[1] if len(sys.argv) > 1 else None
    shots = 8192

    print("=== IBM run: Bell + GHZ + tomography ===\n")

    # Bell tomography (9 settings)
    print("Bell state tomography:")
    bases = [a + b for a in "XYZ" for b in "XYZ"]
    tomo = {b: run(bell_prep(b[0], b[1]), backend, shots) for b in bases}

    # Compute <XX>, <YY>, <ZZ>, <ZI>, <IZ>
    def ev2(counts, a, b):
        probs = parities(counts, shots, n=2)
        s = 0
        for key, p in probs.items():
            q1, q0 = int(key[0]), int(key[1])
            sign = 1
            if a != "I" and q0 == 1: sign *= -1
            if b != "I" and q1 == 1: sign *= -1
            s += sign * p
        return s

    XX = ev2(tomo["XX"], "X", "X")
    YY = ev2(tomo["YY"], "Y", "Y")
    ZZ = ev2(tomo["ZZ"], "Z", "Z")
    print(f"  <XX> = {XX:+.4f}   <YY> = {YY:+.4f}   <ZZ> = {ZZ:+.4f}")
    C_bound = max(0.0, abs(XX) + abs(YY) + abs(ZZ) - 1) / 2
    print(f"  concurrence (Horodecki bound) = {C_bound:.4f}")

    # GHZ
    print("\nGHZ state (3 qubits):")
    ghz_counts = run(ghz_prep(), backend, shots)
    top = sorted(ghz_counts.items(), key=lambda kv: -kv[1])[:4]
    print(f"  top outcomes: {top}")
    p_000 = ghz_counts.get("000", 0) / shots
    p_111 = ghz_counts.get("111", 0) / shots
    print(f"  P(000) = {p_000:.4f}   P(111) = {p_111:.4f}")
    print(f"  GHZ fidelity lower bound = {p_000 + p_111:.4f}")


if __name__ == "__main__":
    main()
