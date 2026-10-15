"""Threshold analysis with MWPM decoding (if pymatching available)."""
try:
    import pymatching
    HAVE_MWPM = True
except ImportError:
    HAVE_MWPM = False
    print("[skip] pip install pymatching")

if not HAVE_MWPM:
    raise SystemExit(0)

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

N_DATA = 9
N_ANC = 4
Z_STABS = [
    ("Z_A", [1, 2, 4, 5]),
    ("Z_B", [3, 4, 6, 7]),
    ("Z_C", [0, 3]),
    ("Z_D", [5, 8]),
]
LOGICAL_Z = [0, 4, 8]


def build_circuit():
    d = QuantumRegister(N_DATA, "d")
    a = QuantumRegister(N_ANC, "a")
    c = ClassicalRegister(N_DATA + N_ANC, "c")
    qc = QuantumCircuit(d, a, c)
    for i, (_, qs) in enumerate(Z_STABS):
        for q in qs:
            qc.cx(d[q], a[i])
        qc.measure(a[i], c[N_DATA + i])
    for q in range(N_DATA):
        qc.measure(d[q], c[q])
    return qc


def syndrome_to_bitstring(syn):
    """Convert syndrome tuple to a bit index for pymatching."""
    return sum((1 << i) for i, b in enumerate(syn) if b)


def build_matching():
    """Build a Matching graph for the 4-syndrome code.

    Each data qubit q induces a syndrome pattern; the edge
    connects the corresponding syndrome nodes (or to a
    boundary node if the pattern has odd weight).
    """
    # Syndrome index -> detector index (0..3)
    edges = []
    for q in range(N_DATA):
        syn = tuple(1 if q in set(s[1]) else 0 for s in Z_STABS)
        det_indices = [i for i, b in enumerate(syn) if b]
        if len(det_indices) == 2:
            edges.append((det_indices[0], det_indices[1], q))
        elif len(det_indices) == 1:
            # Boundary edge
            edges.append((det_indices[0], 4, q))   # virtual boundary node 4
    # Build pymatching graph
    m = pymatching.Matching()
    for a, b, _ in edges:
        m.add_edge(a, b)
    return m, edges


def logical_error_from_mwpm(bs, m, edges):
    s = bs.replace(" ", "")[::-1]
    syn = tuple(int(s[9 + i]) for i in range(4))
    if not any(syn):
        data_bits = [int(s[i]) for i in range(9)]
        return (data_bits[0] + data_bits[4] + data_bits[8]) % 2
    try:
        correction = m.decode([sum((1 << i) for i, b in enumerate(syn) if b)])[0]
    except Exception:
        return 1
    # Map matching output back to data corrections
    data_bits = [int(s[i]) for i in range(9)]
    for idx, (a, b, q) in enumerate(edges):
        if correction & (1 << idx):
            data_bits[q] ^= 1
    return (data_bits[0] + data_bits[4] + data_bits[8]) % 2


def make_noisy_sim(p_2q):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ["cx"])
    return AerSimulator(noise_model=noise)


if __name__ == "__main__":
    print("=== Threshold with MWPM decoding ===\n")
    qc = build_circuit()
    m, edges = build_matching()
    shots = 4096

    print(f"{'p_2q':>8} {'p_L':>10} {'ratio':>8}")
    print("-" * 30)
    for p in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        sim = make_noisy_sim(p)
        tqc = transpile(qc, sim, optimization_level=2)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        errs = sum(n for bs, n in counts.items()
                   if logical_error_from_mwpm(bs, m, edges) == 1)
        pL = errs / shots
        print(f"{p:>8.4f} {pL:>10.4f} {pL/p if p>0 else float('nan'):>8.3f}")
