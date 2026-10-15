"""Steane [[7,1,3]] code -- correct implementation.

The Steane code is a CSS code.  Its |0>_L is the uniform
superposition of the 8 even-weight codewords of the [7,4,3]
Hamming code:

|0>_L = (1/sqrt(8)) [ |0000000> + |1010101> + |0110011> + |1100110>
                   + |0001111> + |1011010> + |0111100> + |1101001> ]

We prepare |0>_L via StatePreparation (exact), then verify:
  - Z stabilizer syndrome is trivial (000)
  - Logical Z (Z_L = Z^7) is +1
  - X_L (transversal X) flips logical Z to -1
  - Single-qubit X error triggers non-trivial syndrome

Stabilizer generators:
  X-type: IIIXXXX  IXXIIXX  XIXIXIX
  Z-type: IIIZZZZ  IZZIIZZ  ZIZIZIZ

Logical operators:
  X_L = XXXXXXX
  Z_L = ZZZZZZZ
"""
from __future__ import annotations
import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation


# Even-weight codewords of [7,4,3] Hamming code (Steane codewords)
CODEWORDS = [
    [0,0,0,0,0,0,0],
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [1,1,0,0,1,1,0],
    [0,0,0,1,1,1,1],
    [1,0,1,1,0,1,0],
    [0,1,1,1,1,0,0],
    [1,1,0,1,0,0,1],
]

# Bit ordering: qubit 0 is the RIGHTMOST bit in Qiskit's statevector
# Statevector index = sum_i 2^i * bit_i
# We build the amplitude vector accordingly.
def build_logical_zero_statevector():
    """Return a 128-dim complex vector for |0>_L.

    Qiskit convention: |q6 q5 q4 q3 q2 q1 q0>; the integer index
    is sum 2^i * q_i where i is the qubit label.
    Our codewords are given as [q0, q1, ..., q6]; we need to
    convert to Qiskit's index.
    """
    vec = np.zeros(2 ** 7, dtype=complex)
    for cw in CODEWORDS:
        idx = 0
        for i, bit in enumerate(cw):
            if bit:
                idx |= (1 << i)
        vec[idx] += 1.0
    return vec / np.linalg.norm(vec)


# Stabilizer generators as Pauli strings (bit 0 = qubit 0)
# Format: string[i] = Pauli on qubit i
X_GENS = [
    "XXX X   ",
    "X  XX  X",   # will normalise below
    "X X X X ",
]
# Actually let's be explicit:
# IIIXXXX: qubits 3,4,5,6 have X, others I
# IXXIIXX: qubits 1,2,5,6 have X
# XIXIXIX: qubits 0,2,4,6 have X
X_STABS = [
    [3,4,5,6],
    [1,2,5,6],
    [0,2,4,6],
]
Z_STABS = [
    [3,4,5,6],
    [1,2,5,6],
    [0,2,4,6],
]


def build_circuit(inject_error=None, transversal=None):
    """Build: prep |0>_L, optional error, optional transversal, measure Z."""
    qc = QuantumCircuit(7, 7)

    # Prep |0>_L exactly
    psi = build_logical_zero_statevector()
    qc.append(StatePreparation(psi), range(7))

    # Inject error (before transversal)
    if inject_error is not None:
        pauli, qubit = inject_error
        if pauli == "X": qc.x(qubit)
        elif pauli == "Z": qc.z(qubit)
        elif pauli == "Y": qc.y(qubit)

    # Transversal gate
    if transversal == "H":
        for q in range(7):
            qc.h(q)
    elif transversal == "X":
        for q in range(7):
            qc.x(q)
    elif transversal == "Z":
        for q in range(7):
            qc.z(q)

    # Measure Z basis
    for q in range(7):
        qc.measure(q, q)
    return qc


def parse_bits(bs):
    """Qiskit string: leftmost = qubit 6.  Return list bits[i] = qubit i."""
    s = bs.replace(" ", "")
    return [int(s[6 - i]) for i in range(7)]


def z_syndrome(bits):
    syn = []
    for gen in Z_STABS:
        par = sum(bits[q] for q in gen) % 2
        syn.append(par)
    return tuple(syn)


def x_syndrome_from_z_measurement(bits):
    """When we measure in Z basis we can't see X stabilizers directly.
    Return None.  (For a real X-syndrome we need a separate X-basis
    measurement.)"""
    return None


def logical_z(bits):
    """Z_L = Z on all 7 qubits."""
    return sum(bits) % 2


def analyse_z_basis(counts):
    """Post-select on trivial Z syndrome; return logical Z distribution
    and syndrome histogram."""
    logical_hist = {0: 0, 1: 0}
    syndrome_hist = {}
    total = 0
    for bs, n in counts.items():
        bits = parse_bits(bs)
        syn = z_syndrome(bits)
        l = logical_z(bits)
        syndrome_hist[syn] = syndrome_hist.get(syn, 0) + n
        logical_hist[l] += n
        total += n
    return syndrome_hist, logical_hist, total


def run_sim(qc, shots=8192):
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    tqc = transpile(qc, sim, optimization_level=2)
    return sim.run(tqc, shots=shots).result().get_counts()


if __name__ == "__main__":
    print("=== Steane [[7,1,3]] code (corrected) ===\n")
    print("|0>_L = uniform superposition of 8 even-weight codewords.")
    print()

    shots = 8192

    scenarios = [
        ("|0>_L no error", None, None),
        ("X_L (logical X)", None, "X"),
        ("Z_L (logical Z)", None, "Z"),
        ("X error q0", ("X", 0), None),
        ("X error q3", ("X", 3), None),
        ("Z error q0", ("Z", 0), None),
        ("Y error q2", ("Y", 2), None),
        ("transversal H", None, "H"),
    ]

    print(f"{'scenario':<20} {'trivial syn%':>13} {'<Z_L>':>10} "
          f"{'non-trivial syndrome distribution':>35}")
    print("-" * 82)

    for label, err, trans in scenarios:
        qc = build_circuit(inject_error=err, transversal=trans)
        counts = run_sim(qc, shots)
        syn_hist, l_hist, total = analyse_z_basis(counts)
        trivial = syn_hist.get((0, 0, 0), 0) / total
        zL = (l_hist[0] - l_hist[1]) / total

        # show top 3 non-trivial syndromes
        nontrivial = [(s, c) for s, c in syn_hist.items() if s != (0, 0, 0)]
        nontrivial.sort(key=lambda x: -x[1])
        top3 = ", ".join(f"{s}:{c}" for s, c in nontrivial[:3])
        if not top3:
            top3 = "(none)"

        print(f"{label:<20} {100*trivial:>12.2f}% {zL:>+10.4f}  {top3}")

    print()
    print("Interpretation:")
    print("  * |0>_L no error: trivial syndrome 100%, <Z_L> = +1")
    print("  * X_L: trivial syndrome (X_L commutes with Z_stabs),")
    print("         <Z_L> = -1 (state is now |1>_L)")
    print("  * X error qN: non-trivial syndrome (code detects it)")
    print("  * Z error qN: trivial syndrome, <Z_L> = +1 (Z error is")
    print("         invisible in Z basis; a Z_L is unaffected)")
    print("  * Y error: both non-trivial syndrome AND <Z_L> = -1")
    print("  * transversal H: <Z_L> = 0 (state is now |+>_L-like)")
    print()

    with open("steane_results.json", "w") as f:
        import json
        json.dump({"shots": shots}, f)
