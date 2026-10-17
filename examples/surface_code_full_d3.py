"""F26: Full distance-3 rotated surface code (X and Z sectors).

This extends F16 (Z-sector only) to the full code that corrects
both bit-flip (X) and phase-flip (Z) errors.

Layout: 9 data qubits on a 3x3 grid.
    0 - 1 - 2
    |   |   |
    3 - 4 - 5
    |   |   |
    6 - 7 - 8

Z-stabilizers (detect X errors, measured in Z basis):
    Z_A = Z(1,2,4,5)   bulk
    Z_B = Z(3,4,6,7)   bulk
    Z_C = Z(0,3)       left boundary
    Z_D = Z(5,8)       right boundary

X-stabilizers (detect Z errors, measured in X basis):
    X_A = X(0,1,3,4)   bulk
    X_B = X(4,5,7,8)   bulk
    X_C = X(1,2)       top boundary
    X_D = X(6,7)       bottom boundary

Logical operators:
    Z_L = Z(0,4,8)     main diagonal
    X_L = X(2,5,8)     right column
    Z_L ∩ X_L = {8}, so they anticommute -- correct.

Both X and Z stabs commute (verified by construction):
  every X-stab overlaps every Z-stab at even qubits.

Protocol:
  Test A (detects X errors): prep |0>_L = |0>^9, inject X,
    measure Z-syndrome, decode, verify Z_L = +1.
  Test B (detects Z errors): prep |+>_L = H^9|0>^9, inject Z,
    measure X-syndrome, decode, verify X_L = +1.

Together, they show the full d=3 code corrects any single-qubit
Pauli error (X via test A, Z via test B, Y via either component).

Simulator: all 20 scenarios should give fidelity 1.0.
Hardware: expect 0.85-0.95 per scenario.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

N = 9
N_ANC = 8

X_STABS = [
    ("X_A", [0, 1, 3, 4]),
    ("X_B", [4, 5, 7, 8]),
    ("X_C", [1, 2]),
    ("X_D", [6, 7]),
]
Z_STABS = [
    ("Z_A", [1, 2, 4, 5]),
    ("Z_B", [3, 4, 6, 7]),
    ("Z_C", [0, 3]),
    ("Z_D", [5, 8]),
]
X_L = [2, 5, 8]
Z_L = [0, 4, 8]


def build_decoders():
    x_dec = {}
    for q in range(N):
        syn = tuple(1 if q in set(X_STABS[i][1]) else 0 for i in range(4))
        if any(syn) and syn not in x_dec:
            x_dec[syn] = q
    z_dec = {}
    for q in range(N):
        syn = tuple(1 if q in set(Z_STABS[i][1]) else 0 for i in range(4))
        if any(syn) and syn not in z_dec:
            z_dec[syn] = q
    return x_dec, z_dec

X_DECODE, Z_DECODE = build_decoders()


def build_circuit(logical_state='0', inject=None):
    """Build the full d=3 surface code circuit.

    logical_state: '0' for |0>_L = |0>^9, '+' for |+>_L = H^9|0>^9.
    inject: (pauli, qubit) or None.
    """
    d = QuantumRegister(N, "d")
    aX = QuantumRegister(4, "aX")
    aZ = QuantumRegister(4, "aZ")
    c = ClassicalRegister(N + N_ANC, "c")
    qc = QuantumCircuit(d, aX, aZ, c)

    # Prep
    if logical_state == '+':
        for q in range(N):
            qc.h(d[q])

    # Inject single-qubit error
    if inject is not None:
        pauli, q = inject
        if pauli == 'X': qc.x(d[q])
        elif pauli == 'Z': qc.z(d[q])
        elif pauli == 'Y': qc.y(d[q])

    # Z-syndrome (CX from data to aZ)
    for i, (_, plaq) in enumerate(Z_STABS):
        for q in plaq:
            qc.cx(d[q], aZ[i])
    for i in range(4):
        qc.measure(aZ[i], c[N + i])

    # X-syndrome (H(aX), CX(aX, data), H(aX), measure)
    for i, (_, plaq) in enumerate(X_STABS):
        qc.h(aX[i])
        for q in plaq:
            qc.cx(aX[i], d[q])
        qc.h(aX[i])
    for i in range(4):
        qc.measure(aX[i], c[N + 4 + i])

    # Final readout
    if logical_state == '0':
        # Z_L = Z(0,4,8), measure all in Z basis
        for q in range(N):
            qc.measure(d[q], c[q])
    elif logical_state == '+':
        # X_L = X(2,5,8), apply H then measure
        for q in X_L:
            qc.h(d[q])
        for q in range(N):
            qc.measure(d[q], c[q])

    return qc


def analyse(counts, logical_state):
    """Apply syndrome correction, then check logical operator."""
    total = 0
    correct = 0
    for bs, n in counts.items():
        bits = bs.replace(" ", "")[::-1]
        data = [int(bits[i]) for i in range(N)]
        z_syn = tuple(int(bits[N + i]) for i in range(4))
        x_syn = tuple(int(bits[N + 4 + i]) for i in range(4))

        if logical_state == '0':
            # X-error detected by Z-syndrome
            if z_syn in Z_DECODE:
                data[Z_DECODE[z_syn]] ^= 1
            par = (data[Z_L[0]] + data[Z_L[1]] + data[Z_L[2]]) % 2
        else:
            # Z-error detected by X-syndrome
            if x_syn in X_DECODE:
                data[X_DECODE[x_syn]] ^= 1
            par = (data[X_L[0]] + data[X_L[1]] + data[X_L[2]]) % 2

        if par == 0:
            correct += n
        total += n
    return correct / total if total else 0.0


def connect_ibm():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        return None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN", "")
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] {ch}")
            break
        except Exception:
            service = None
    if service is None:
        return None
    name = os.environ.get("QHAL_IBM_BACKEND", "ibm_fez")
    try:
        backend = service.backend(name)
    except Exception:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"[ibm] {backend.name} queue={backend.status().pending_jobs}")
    return backend


def run_sim(qc, shots=8192):
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    tqc = transpile(qc, sim, optimization_level=2)
    return sim.run(tqc, shots=shots).result().get_counts()


def run_hw(qc, backend, shots=8192):
    from qiskit_ibm_runtime import SamplerV2
    tqc = transpile(qc, backend, optimization_level=3)
    tq = sum(tqc.count_ops().get(k, 0) for k in ("cx", "ecr", "cz"))
    print(f"  depth={tqc.depth()}, 2q={tq}")
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    print(f"  job {job.job_id()}")
    return job.result()[0].data.c.get_counts()


if __name__ == "__main__":
    print("=== F26: Full d=3 rotated surface code ===\n")
    print(f"X_L = {X_L}, Z_L = {Z_L}")
    print(f"X_L ∩ Z_L = {sorted(set(X_L) & set(Z_L))} (should be size 1)")
    print(f"Z-decode table: {len(Z_DECODE)} entries")
    print(f"X-decode table: {len(X_DECODE)} entries\n")

    backend = connect_ibm()
    shots = 8192

    # Test A: Z-sector (X errors)
    print("Test A: prep |0>_L, inject X, correct via Z-syndrome, verify Z_L")
    print(f"{'scenario':<14} {'fidelity':>10}")
    print("-" * 26)
    results_A = {}
    scenarios_A = [("no error", None)] + [(f"X on q{q}", ("X", q)) for q in range(N)]
    for label, inject in scenarios_A:
        qc = build_circuit('0', inject)
        counts = run_sim(qc, shots) if backend is None else run_hw(qc, backend, shots)
        fid = analyse(counts, '0')
        results_A[label] = fid
        print(f"{label:<14} {fid:>10.4f}")

    print()
    print("Test B: prep |+>_L, inject Z, correct via X-syndrome, verify X_L")
    print(f"{'scenario':<14} {'fidelity':>10}")
    print("-" * 26)
    results_B = {}
    scenarios_B = [("no error", None)] + [(f"Z on q{q}", ("Z", q)) for q in range(N)]
    for label, inject in scenarios_B:
        qc = build_circuit('+', inject)
        counts = run_sim(qc, shots) if backend is None else run_hw(qc, backend, shots)
        fid = analyse(counts, '+')
        results_B[label] = fid
        print(f"{label:<14} {fid:>10.4f}")

    # Summary
    min_A = min(results_A.values())
    min_B = min(results_B.values())
    overall = min(min_A, min_B)

    print()
    print("=== Summary ===")
    print(f"  Test A (X errors): min fidelity = {min_A:.4f}")
    print(f"  Test B (Z errors): min fidelity = {min_B:.4f}")
    print(f"  Overall min      = {overall:.4f}")
    if overall > 0.95:
        print("  -> PASS: full d=3 code corrects X and Z errors")
    elif overall > 0.7:
        print("  -> PARTIAL: code works with some hardware noise")
    else:
        print("  -> FAIL: check syndrome or decoder")

    with open("surface_full_d3_results.json", "w") as f:
        json.dump({"test_A_Zsector": results_A,
                   "test_B_Xsector": results_B,
                   "min": overall,
                   "backend": backend.name if backend else "simulator"}, f, indent=2)
    print("\nResults written to surface_full_d3_results.json")
