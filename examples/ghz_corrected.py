"""GHZ prep corrected.

QCA-native sequence:
    H(0)                    |+00>
    H(1)                    |++0>
    CZ(0,1)                 entangled pair (0,1)
    H(1)                    uncompute qubit 1
    H(2)                    |GHZ_01, +>
    CZ(1,2)                 extend to qubit 2
    H(2)                    uncompute qubit 2

Result: (|000> + |111>)/√2.
"""
import os, sys
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False
    print("[warn] qiskit not installed")


def ghz_native(meas=None):
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1); qc.cz(0, 1); qc.h(1)   # CX(0,1)
    qc.h(2); qc.cz(1, 2); qc.h(2)   # CX(1,2)
    if meas:
        for q, b in enumerate(meas):
            if b == "X": qc.h(q)
            elif b == "Y": qc.sdg(q); qc.h(q)
    qc.measure([0,1,2], [0,1,2])
    return qc


def run(qc, shots=8192):
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    return sim.run(transpile(qc, sim), shots=shots).result().get_counts()


if __name__ == "__main__":
    if not HAVE_QISKIT:
        sys.exit(1)
    print("=== 3-qubit GHZ (QCA-native) ===\n")
    shots = 8192
    counts = run(ghz_native(), shots)
    print("counts (all non-zero):")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k}: {v:>5}  ({v/shots:.4f})")
    p000 = counts.get("000", 0) / shots
    p111 = counts.get("111", 0) / shots
    print(f"\nP(000) + P(111) = {p000 + p111:.4f}   (ideal 1.0)")
    print(f"GHZ fidelity    = {p000 + p111:.4f}")

    print("\nStabilizer expectations (all should be +1):")
    for basis in ("XXX", "ZZI", "IZZ"):
        c = run(ghz_native(basis), shots)
        ev = 0
        for k, v in c.items():
            sign = 1
            for q, b in enumerate(basis):
                if b != "I" and int(k[2-q]) == 1:
                    sign *= -1
            ev += sign * v / shots
        print(f"  <{basis}> = {ev:+.4f}")
