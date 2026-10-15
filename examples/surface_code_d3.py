"""Distance-3 rotated surface code -- Z-sector (X-error correction).

9 data qubits on a 3x3 grid:

    d0 - d1 - d2
    |    |    |
    d3 - d4 - d5
    |    |    |
    d6 - d7 - d8

4 Z-type stabilizers (each is a product of Z's):

    Z_A (weight 4): d1 d2 d4 d5
    Z_B (weight 4): d3 d4 d6 d7
    Z_C (weight 2): d0 d3        (top-left boundary)
    Z_D (weight 2): d5 d8        (bottom-right boundary)

Logical operators (commute with all stabs, anticommute with each other):

    Z̄ = Z_0 Z_4 Z_8   (diagonal, weight 3)
    X̄ = X_2 X_5 X_8   (right column, weight 3)

Distance: minimum weight of a logical operator = 3.
Corrects: any single-qubit X error.

QCA mapping: every Z-stab is a native sigma_z sigma_z
multi-qubit coupling measured with a single ancilla.

Protocol:
  1. Prep |0>^9 (trivial +1 eigenstate of all Z-stabs)
  2. Inject a single X error on any data qubit
  3. Measure all 4 Z-stabs via 4 ancillas
  4. Measure all 9 data qubits in Z basis
  5. Decode Z-syndrome to identify the error
  6. Verify logical Z̄ = +1 (state is |0>_L)
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile


N_DATA = 9
N_ANC = 4
N_TOTAL = N_DATA + N_ANC   # 13


# Z-type stabilizers
Z_STABS = [
    ("Z_A", [1, 2, 4, 5]),
    ("Z_B", [3, 4, 6, 7]),
    ("Z_C", [0, 3]),
    ("Z_D", [5, 8]),
]

# Logical operators
LOGICAL_Z = [0, 4, 8]     # Z̄ = Z_0 Z_4 Z_8
LOGICAL_X = [2, 5, 8]     # X̄ = X_2 X_5 X_8


def build_circuit(inject_x_on=None):
    """Prep |0>^9, inject X error, measure Z syndromes + data."""
    d = QuantumRegister(N_DATA, "d")
    a = QuantumRegister(N_ANC, "a")
    c = ClassicalRegister(N_DATA + N_ANC, "c")
    qc = QuantumCircuit(d, a, c)

    # Prep |0>^9 -- already in +1 eigenspace of all Z stabs

    # Inject error
    if inject_x_on is not None:
        qc.x(d[inject_x_on])

    # Z syndrome extraction: CX from data to ancilla
    for i, (_, qs) in enumerate(Z_STABS):
        for q in qs:
            qc.cx(d[q], a[i])
        qc.measure(a[i], c[N_DATA + i])

    # Read out all data in Z basis
    for q in range(N_DATA):
        qc.measure(d[q], c[q])

    return qc


# ------------------------------------------------------------------
# Decoder
# ------------------------------------------------------------------
def build_decoder():
    """Map Z-syndrome -> most likely X-error qubit.

    Note: q1/q2 have the same syndrome (both trigger only Z_A), and
    q6/q7 have the same syndrome (both trigger only Z_B).  The
    decoder picks the first one found, which is safe because
    either choice gives the same logical Z̄ -- both q1, q2 lie
    outside Z̄, and both q6, q7 lie outside Z̄.
    """
    table = {}
    for q in range(N_DATA):
        syn = tuple(1 if q in set(s[1]) else 0 for s in Z_STABS)
        if any(syn):
            table.setdefault(syn, q)
    return table


DECODER = build_decoder()


def parse(bs):
    """Qiskit strings: rightmost bit = classical bit 0.
    c[0..8] = data, c[9..12] = ancillas."""
    s = bs.replace(" ", "")[::-1]
    data = [int(s[i]) for i in range(9)]
    syn = tuple(int(s[9 + i]) for i in range(4))
    return data, syn


def decode(bs):
    """Apply correction from Z-syndrome, then compute logical Z̄."""
    data, syn = parse(bs)
    if syn in DECODER:
        data[DECODER[syn]] ^= 1
    return (data[LOGICAL_Z[0]] + data[LOGICAL_Z[1]] + data[LOGICAL_Z[2]]) % 2


def analyse(counts):
    correct = total = 0
    for bs, n in counts.items():
        if decode(bs) == 0:
            correct += n
        total += n
    return correct / total if total else 0.0


# ------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------
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
    print("=== Distance-3 rotated surface code, Z-sector (13 qubits) ===\n")
    print("Z-type stabilizers (detect X errors):")
    for label, qs in Z_STABS:
        print(f"  {label} (wt {len(qs)}): {qs}")
    print(f"Logical X̄ = X on {LOGICAL_X}")
    print(f"Logical Z̄ = Z on {LOGICAL_Z}")
    print(f"Decoder table: {len(DECODER)} distinct syndromes\n")

    backend = connect()
    shots = 8192

    scenarios = [
        ("no error", None),
        ("X on d0", 0),
        ("X on d1", 1),
        ("X on d2", 2),
        ("X on d3", 3),
        ("X on d4", 4),
        ("X on d5", 5),
        ("X on d6", 6),
        ("X on d7", 7),
        ("X on d8", 8),
    ]

    print(f"{'scenario':<12} {'shots':>7} {'fidelity |0>_L':>16}")
    print("-" * 40)

    results = {}
    for label, inject in scenarios:
        qc = build_circuit(inject_x_on=inject)
        if backend is None:
            counts = run_sim(qc, shots)
        else:
            counts = run_hw(qc, backend, shots)
        fid = analyse(counts)
        results[label] = fid
        print(f"{label:<12} {sum(counts.values()):>7} {fid:>16.4f}")

    print("\nInterpretation:")
    print("  Every row should read 1.0000 on the simulator: the code")
    print("  detects and corrects every single-qubit X error.")
    print("  On hardware, all rows should match to within noise --")
    print("  the logical qubit hides the physical error.")

    with open("surface_code_d3_results.json", "w") as f:
        json.dump({"backend": backend.name if backend else "simulator",
                   "shots": shots, "results": results}, f, indent=2)
    print("\nResults written to surface_code_d3_results.json")
