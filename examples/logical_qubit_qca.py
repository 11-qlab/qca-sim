"""Distance-3 repetition code logical qubit on 5 qubits.

The simplest fault-tolerant logical qubit.  Encodes one logical
bit into three physical qubits:

    |0>_L = |000>
    |1>_L = |111>

Two ancillas measure the ZZ stabilizers:
    S1 = Z0 Z1     (ancilla a0)
    S2 = Z1 Z2     (ancilla a1)

A single X error on any data qubit flips the syndrome:
    X on d0 -> (S1, S2) = (1, 0)
    X on d1 -> (S1, S2) = (1, 1)
    X on d2 -> (S1, S2) = (0, 1)

Majority vote decodes: measure the data, take the majority as
the logical bit.  A single X error is corrected.

This is a genuine logical qubit with distance 3, encoding a
logical bit that survives any single physical X error.

QCA mapping: all stabilizers are native sigma_z sigma_z
couplings.  The whole circuit is 5 qubits, depth ~10 raw.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile


N_DATA = 3
N_ANC = 2
N_QUBITS = N_DATA + N_ANC


# ------------------------------------------------------------------
# Circuit: prep |0>_L, inject X error, extract syndrome, measure
# ------------------------------------------------------------------
def build_circuit(inject_on=None):
    """Build the 5-qubit repetition code experiment.

    Sequence:
      1. Prepare |0>_L = |000> (data qubits start in |0>)
      2. Optionally inject X on data qubit `inject_on`
      3. Extract syndromes via 2 ancillas
      4. Measure all 3 data qubits
      5. Offline: decode and verify logical fidelity
    """
    d = QuantumRegister(N_DATA, "d")
    a = QuantumRegister(N_ANC, "a")
    c = ClassicalRegister(N_DATA + N_ANC, "c")
    qc = QuantumCircuit(d, a, c)

    # 1. Prep |000> -- data already in |0>, nothing to do

    # 2. Inject error (before syndrome extraction, simulating
    #    a physical error that occurred during idle or gates)
    if inject_on is not None:
        qc.x(d[inject_on])

    # 3. Syndrome extraction
    # S1 = Z0 Z1 -> ancilla a[0]
    qc.cx(d[0], a[0])
    qc.cx(d[1], a[0])
    qc.measure(a[0], c[3])

    # S2 = Z1 Z2 -> ancilla a[1]
    qc.cx(d[1], a[1])
    qc.cx(d[2], a[1])
    qc.measure(a[1], c[4])

    # 4. Measure data
    for q in range(N_DATA):
        qc.measure(d[q], c[q])

    return qc


# ------------------------------------------------------------------
# Decoder: majority vote with syndrome-based correction
# ------------------------------------------------------------------
def decode(syndrome, data_bits):
    """Given syndrome (s1, s2) and raw data bits, return corrected
    logical bit.

    Syndrome table:
      (0, 0): no error -> logical = data[1] (or majority)
      (1, 0): X on d0  -> flip d0, then take majority
      (1, 1): X on d1  -> flip d1
      (0, 1): X on d2  -> flip d2
    """
    s1, s2 = syndrome
    d = list(data_bits)
    if (s1, s2) == (0, 0):
        pass
    elif (s1, s2) == (1, 0):
        d[0] ^= 1
    elif (s1, s2) == (1, 1):
        d[1] ^= 1
    elif (s1, s2) == (0, 1):
        d[2] ^= 1
    # Majority vote
    return 1 if sum(d) >= 2 else 0


def parse_bits(bitstring):
    """Qiskit strings: rightmost = classical bit 0.
    c[0..2] = data d0..d2, c[3] = ancilla a0, c[4] = ancilla a1."""
    s = bitstring.replace(" ", "")[::-1]
    data = [int(s[0]), int(s[1]), int(s[2])]
    anc = [int(s[3]), int(s[4])]
    return data, anc


def analyse(counts, expected_logical=0):
    """Compute logical error rate after decoding."""
    total = 0
    correct = 0
    decoded_hist = {0: 0, 1: 0}
    for bs, n in counts.items():
        data, anc = parse_bits(bs)
        logical = decode(tuple(anc), data)
        decoded_hist[logical] += n
        total += n
        if logical == expected_logical:
            correct += n
    return correct / total if total else 0.0, decoded_hist, total


# ------------------------------------------------------------------
# Backend connection
# ------------------------------------------------------------------
def connect():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        print("[mode] simulator")
        return None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN", "")
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] {ch}")
            break
        except Exception as e:
            print(f"[ibm] {ch}: {e}")
            service = None
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
    two_q = sum(tqc.count_ops().get(k, 0) for k in ("cx", "ecr", "cz"))
    print(f"  transpiled depth = {tqc.depth()}, 2q = {two_q}")
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    print(f"  job {job.job_id()}")
    return job.result()[0].data.c.get_counts()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Repetition code logical qubit (d=3) ===\n")
    print(f"Layout: {N_DATA} data + {N_ANC} ancilla = {N_QUBITS} qubits")
    print("  S1 = Z(d0) Z(d1)  on ancilla a0")
    print("  S2 = Z(d1) Z(d2)  on ancilla a1")
    print("  Logical |0>_L = |000>, |1>_L = |111>\n")

    backend = connect()
    shots = 8192

    # Test 4 scenarios: no error, X on d0, X on d1, X on d2
    scenarios = [
        ("no error", None, 0),
        ("X on d0", 0, 0),
        ("X on d1", 1, 0),
        ("X on d2", 2, 0),
    ]

    print(f"{'scenario':<12} {'kept':>6} {'logical fidelity':>18} "
          f"{'decoded |0>':>12} {'decoded |1>':>12}")
    print("-" * 68)

    results = {}
    for label, inject_on, expected in scenarios:
        qc = build_circuit(inject_on=inject_on)
        if backend is None:
            counts = run_sim(qc, shots)
        else:
            counts = run_hw(qc, backend, shots)

        fidelity, hist, total = analyse(counts, expected)
        results[label] = {
            "fidelity": fidelity,
            "hist": hist,
            "total": total,
        }
        print(f"{label:<12} {total:>6} {fidelity:>18.4f} "
              f"{hist[0]:>12} {hist[1]:>12}")

    # Physical vs logical comparison
    print(f"\nPhysical error rate on Marrakesh:")
    print(f"  single-qubit gate error ~ 1e-4")
    print(f"  CX error               ~ 7e-3")
    print(f"  Circuit has 4 CX -> physical error ~ {1 - (1-0.007)**4:.4f}")

    print(f"\nLogical fidelity after decoding:")
    for label, r in results.items():
        print(f"  {label:<12} {r['fidelity']:.4f}")

    print(f"\nInterpretation:")
    print(f"  For the 'no error' and single-X-error cases, the code")
    print(f"  should decode to the correct logical value in EVERY shot")
    print(f"  (fidelity 1.0) in the absence of noise.  On hardware,")
    print(f"  fidelity will be reduced by physical gate errors, but the")
    print(f"  logical fidelity should still be HIGHER than the naive")
    print(f"  physical error rate on a single CX.")

    with open("logical_qubit_results.json", "w") as f:
        json.dump({
            "backend": backend.name if backend else "simulator",
            "shots": shots,
            "results": results,
        }, f, indent=2)
    print("\nResults written to logical_qubit_results.json")
