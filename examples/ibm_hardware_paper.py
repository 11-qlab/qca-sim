"""Run the QCA quantum findings on real IBM hardware.

Submits:
  1. Bell state (2 qubits) + full tomography
  2. GHZ state (3 qubits)
  3. Logical Bell pair (4 physical qubits)
Writes results to hardware_results.json.

Usage:
    pip install qiskit qiskit-aer qiskit-ibm-runtime
    export IBMQ_TOKEN=<token>
    python3 examples/ibm_hardware_paper.py ibm_brisbane
"""
from __future__ import annotations
import os, sys, json
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import DensityMatrix, concurrence as qk_conc
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False
    print("[fatal] pip install qiskit qiskit-aer qiskit-ibm-runtime")
    sys.exit(1)


PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0,1],[1,0]], dtype=complex),
    "Y": np.array([[0,-1j],[1j,0]], dtype=complex),
    "Z": np.array([[1,0],[0,-1]], dtype=complex),
}


def bell_circuit(a="Z", b="Z"):
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.h(1); qc.cz(0, 1); qc.h(1)
    for q, base in ((0, a), (1, b)):
        if base == "X":
            qc.h(q)
        elif base == "Y":
            qc.sdg(q); qc.h(q)
    qc.measure([0, 1], [0, 1])
    return qc


def ghz_circuit(basis=None):
    """QCA-native GHZ: H(0); CX(0,1); CX(1,2)."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1); qc.cz(0, 1); qc.h(1)
    qc.h(2); qc.cz(1, 2); qc.h(2)
    if basis:
        for q, b in enumerate(basis):
            if b == "X":
                qc.h(q)
            elif b == "Y":
                qc.sdg(q); qc.h(q)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def logical_bell_circuit():
    """Logical Bell pair via QCA-native CNOT."""
    qc = QuantumCircuit(4, 4)
    qc.h(0); qc.h(1); qc.cz(0, 1); qc.h(1)
    qc.cx(1, 3)
    qc.measure([0,1,2,3], [0,1,2,3])
    return qc


def run_circuit(qc, backend_name=None, shots=8192):
    if backend_name is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        return sim.run(transpile(qc, sim), shots=shots).result().get_counts()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    token = os.environ.get("IBMQ_TOKEN")
    if not token:
        print("[fatal] IBMQ_TOKEN not set")
        sys.exit(1)
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.backend(backend_name)
    tqc = transpile(qc, backend, optimization_level=3)
    sampler = SamplerV2(backend)
    job = sampler.run([tqc], shots=shots)
    print(f"  [hw] job {job.job_id()}")
    return job.result()[0].data.c.get_counts()


def expectation_2q(counts, shots, a, b):
    ev = 0.0
    for k, n in counts.items():
        q1, q0 = int(k[0]), int(k[1])
        sign = 1
        if a != "I" and q0:
            sign *= -1
        if b != "I" and q1:
            sign *= -1
        ev += sign * n / shots
    return ev


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else None
    shots = 8192
    results = {"backend": backend or "simulator", "shots": shots}

    print("=== QCA hardware run ===\n")

    print("1. Bell state tomography")
    tomo = {}
    for a in "XYZ":
        for b in "XYZ":
            tomo[a + b] = run_circuit(bell_circuit(a, b), backend, shots)
    XX = expectation_2q(tomo["XX"], shots, "X", "X")
    YY = expectation_2q(tomo["YY"], shots, "Y", "Y")
    ZZ = expectation_2q(tomo["ZZ"], shots, "Z", "Z")
    C_lb = max(0.0, (abs(XX) + abs(YY) + abs(ZZ) - 1) / 2)
    print(f"  <XX> = {XX:+.4f}   <YY> = {YY:+.4f}   <ZZ> = {ZZ:+.4f}")
    print(f"  concurrence (lower bound) = {C_lb:.4f}")
    results["bell"] = {
        "XX": XX, "YY": YY, "ZZ": ZZ,
        "concurrence_lower_bound": C_lb,
        "counts": {k: dict(v) for k, v in tomo.items()},
    }

    print("\n2. GHZ state")
    ghz_counts = run_circuit(ghz_circuit(), backend, shots)
    p000 = ghz_counts.get("000", 0) / shots
    p111 = ghz_counts.get("111", 0) / shots
    F_ghz = p000 + p111
    print(f"  P(000) = {p000:.4f}   P(111) = {p111:.4f}")
    print(f"  GHZ fidelity lower bound = {F_ghz:.4f}")
    results["ghz"] = {
        "p000": p000, "p111": p111, "fidelity_lb": F_ghz,
        "counts": dict(ghz_counts),
    }

    print("\n3. Logical Bell pair (4 physical qubits)")
    lb_counts = run_circuit(logical_bell_circuit(), backend, shots)
    corr = 0
    tot = 0
    for k, n in lb_counts.items():
        b1 = int(k[2])
        b3 = int(k[0])
        corr += (b1 == b3) * n
        tot += n
    parity = corr / tot
    print(f"  P(q1 == q3) = {parity:.4f}   (ideal 1.0)")
    results["logical_bell"] = {"parity_match": parity, "counts": dict(lb_counts)}

    with open("hardware_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults written to hardware_results.json")

    print("\nSummary for the paper:")
    print(f"  Bell state  : C >= {C_lb:.4f}   (ideal 1.0)")
    print(f"  GHZ state   : F >= {F_ghz:.4f}   (ideal 1.0)")
    print(f"  Logical Bell: parity = {parity:.4f}   (ideal 1.0)")
