"""Logical Bell pair via transversal CNOT (fixed prep).

Fix: prep |+>_L directly with a cat-state circuit:
    H(A_0); CX(A_0, A_1); CX(A_0, A_2)
This gives (|000> + |111>)/sqrt(2) exactly, no ancilla needed.

Then transversal CNOT(A_i, B_i) for i = 0, 1, 2 produces
the logical Bell state (|00>_L + |11>_L)/sqrt(2), which has
<Z_A Z_B> = <X_A X_B> = +1.

6 qubits total (3 per patch), 5 CX gates, depth ~6.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile


N_PER = 3
N_TOTAL = 2 * N_PER   # 6


def build_circuit(z_basis=True):
    """Prep |+>_L on A, |0>_L on B, apply transversal CNOT."""
    d = QuantumRegister(N_TOTAL, "d")
    c = ClassicalRegister(N_TOTAL, "c")
    qc = QuantumCircuit(d, c)

    # 1. Prep patch A: cat state |+>_L = (|000> + |111>)/sqrt(2)
    qc.h(d[0])
    qc.cx(d[0], d[1])
    qc.cx(d[0], d[2])

    # 2. Patch B is |0>_L = |000>, already initialized

    # 3. Transversal CNOT: CX(A_i, B_i) for i = 0, 1, 2
    for i in range(N_PER):
        qc.cx(d[i], d[N_PER + i])

    # 4. Basis rotation for X measurement
    if not z_basis:
        for q in range(N_TOTAL):
            qc.h(d[q])

    # 5. Measure all data
    for q in range(N_TOTAL):
        qc.measure(d[q], c[q])

    return qc


def parse(bs):
    """Rightmost bit = qubit 0.
    d[0..2] = patch A, d[3..5] = patch B."""
    s = bs.replace(" ", "")[::-1]
    return [int(s[i]) for i in range(6)]


def za_zb_eigenvalue(bs):
    """Compute (-1)^(Z_A + Z_B) where Z_A = Z_{A0}, Z_B = Z_{B0}."""
    d = parse(bs)
    return (-1) ** (d[0] + d[3])


def xa_xb_eigenvalue(bs):
    """Compute (-1)^(X_A + X_B) where X_A = X_{A0}X_{A1}X_{A2},
    X_B = X_{B0}X_{B1}X_{B2}.  After H rotation, each measured
    Z bit is the corresponding X eigenvalue."""
    d = parse(bs)
    xa_parity = (d[0] + d[1] + d[2]) % 2
    xb_parity = (d[3] + d[4] + d[5]) % 2
    return (-1) ** (xa_parity + xb_parity)


def analyse(counts, evaluator):
    total = 0
    n = 0
    for bs, shots in counts.items():
        total += evaluator(bs) * shots
        n += shots
    return total / n if n else 0.0, n


def connect():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        print("[mode] simulator")
        return None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN", "")
    service = None
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] {ch}"); break
        except Exception as e:
            print(f"[ibm] {ch}: {e}")
    if service is None:
        return None
    name = os.environ.get("QHAL_IBM_BACKEND", "ibm_marrakesh")
    try:
        backend = service.backend(name)
    except Exception:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"[ibm] {backend.name}  queue={backend.status().pending_jobs}")
    return backend


def run_sim(qc, shots):
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    tqc = transpile(qc, sim, optimization_level=2)
    return sim.run(tqc, shots=shots).result().get_counts()


def run_hw(qc, backend, shots):
    from qiskit_ibm_runtime import SamplerV2
    tqc = transpile(qc, backend, optimization_level=3)
    tq = sum(tqc.count_ops().get(k, 0) for k in ("cx", "ecr", "cz"))
    print(f"  depth = {tqc.depth()}, 2q = {tq}")
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    print(f"  job {job.job_id()}")
    return job.result()[0].data.c.get_counts()


if __name__ == "__main__":
    print("=== Logical Bell pair via transversal CNOT ===\n")
    print("Circuit:")
    print("  1. Prep A: H(A_0), CX(A_0, A_1), CX(A_0, A_2) -> |+>_L")
    print("  2. Prep B: |000> -> |0>_L")
    print("  3. Transversal CNOT: CX(A_i, B_i) for i=0,1,2")
    print("  Expected: <Z_A Z_B> = <X_A X_B> = +1\n")

    shots = 8192
    backend = connect()

    # Z-basis
    qc_z = build_circuit(z_basis=True)
    if backend is None:
        counts_z = run_sim(qc_z, shots)
    else:
        counts_z = run_hw(qc_z, backend, shots)
    zz, n_z = analyse(counts_z, za_zb_eigenvalue)
    print(f"Z-basis ({n_z} shots): <Z_A Z_B> = {zz:+.4f}\n")

    # X-basis
    qc_x = build_circuit(z_basis=False)
    if backend is None:
        counts_x = run_sim(qc_x, shots)
    else:
        counts_x = run_hw(qc_x, backend, shots)
    xx, n_x = analyse(counts_x, xa_xb_eigenvalue)
    print(f"X-basis ({n_x} shots): <X_A X_B> = {xx:+.4f}\n")

    print("=== Summary ===")
    print(f"  <Z_A Z_B> = {zz:+.4f}  (ideal +1)")
    print(f"  <X_A X_B> = {xx:+.4f}  (ideal +1)")
    if zz > 0.95 and xx > 0.95:
        print("  -> PASS: logical Bell state verified.")
    else:
        print("  -> partial")

    with open("logical_bell_results.json", "w") as f:
        json.dump({"zz": zz, "xx": xx, "shots": shots}, f, indent=2)
    print("\nResults written to logical_bell_results.json")
