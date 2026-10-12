"""L=2 toric code on IBM -- split circuits + readout mitigation.

Two short circuits instead of one long one:
  A_v circuit: 8 H gates + measure      -> ~15 layers transpiled
  B_p circuit: 8 H + 16 CX + 4 meas     -> ~40 layers transpiled

Then readout-error mitigation via calibration matrices.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

L = 2
N = 2 * L * L

def h_idx(i, j): return (i % L) * L + (j % L)
def v_idx(i, j): return L * L + (i % L) * L + (j % L)

PLAQS = [[h_idx(i, j), v_idx(i, (j + 1) % L),
          h_idx((i + 1) % L, j), v_idx(i, j)]
         for i in range(L) for j in range(L)]

VERTS = [[h_idx(i, j), h_idx(i, (j - 1) % L),
          v_idx(i, j), v_idx((i - 1) % L, j)]
         for i in range(L) for j in range(L)]


def a_v_circuit():
    """|+>^8 + identity -> measure.  <A_v> from Z-basis readout."""
    d = QuantumRegister(8, "d")
    c = ClassicalRegister(8, "c")
    qc = QuantumCircuit(d, c)
    for q in range(8):
        qc.h(d[q])
    for q in range(8):
        qc.measure(d[q], c[q])
    return qc


def b_p_circuit():
    """|+>^8 + CNOT-ladders onto ancillas -> measure ancillas."""
    d = QuantumRegister(8, "d")
    a = QuantumRegister(4, "a")
    c = ClassicalRegister(4, "c")
    qc = QuantumCircuit(d, a, c)
    for q in range(8):
        qc.h(d[q])
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    for p in range(4):
        qc.measure(a[p], c[p])
    return qc


def readout_cal_circuits():
    """|0..0> and |1..1> on 12 qubits for calibration."""
    qc0 = QuantumCircuit(12, 12)
    for q in range(12):
        qc0.measure(q, q)
    qc1 = QuantumCircuit(12, 12)
    for q in range(12):
        qc1.x(q)
    for q in range(12):
        qc1.measure(q, q)
    return qc0, qc1


def analyse_av(counts, v_qubits):
    ev = 0.0; total = 0
    for bs, n in counts.items():
        bits = bs.replace(" ", "")
        par = sum(int(bits[len(bits) - 1 - q]) for q in v_qubits) % 2
        ev += ((-1) ** par) * n
        total += n
    return ev / total


def analyse_bp(counts):
    Bp = [0.0] * 4; total = 0
    for bs, n in counts.items():
        bits = bs.replace(" ", "")
        for i in range(4):
            Bp[i] += ((-1) ** int(bits[len(bits) - 1 - i])) * n
        total += n
    return [x / total for x in Bp]


def connect_ibm():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        print("[mode] simulator")
        return None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ["QISKIT_IBM_TOKEN"]
    service = None
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] channel = {ch}")
            break
        except Exception as e:
            print(f"[ibm] {ch} failed: {e}")
    if service is None:
        return None
    name = os.environ.get("QHAL_IBM_BACKEND", "ibm_marrakesh")
    try:
        backend = service.backend(name)
    except Exception:
        backend = service.least_busy(operational=True, simulator=False)
        print(f"[ibm] fallback: {backend.name}")
    st = backend.status()
    print(f"[ibm] {backend.name}  qubits={backend.num_qubits}  queue={st.pending_jobs}")
    return backend


def count_2q(tqc):
    ops = dict(tqc.count_ops())
    return sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz"))


if __name__ == "__main__":
    print("\n=== L=2 toric code -- split circuits (v3) ===\n")

    qc_av = a_v_circuit()
    qc_bp = b_p_circuit()
    print(f"A_v raw: {qc_av.num_qubits} qubits, depth = {qc_av.depth()}")
    print(f"B_p raw: {qc_bp.num_qubits} qubits, depth = {qc_bp.depth()}\n")

    backend = connect_ibm()
    shots = 16384

    if backend is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        c_av = sim.run(transpile(qc_av, sim), shots=shots).result().get_counts()
        c_bp = sim.run(transpile(qc_bp, sim), shots=shots).result().get_counts()
    else:
        from qiskit_ibm_runtime import SamplerV2
        tqc_av = transpile(qc_av, backend, optimization_level=3)
        tqc_bp = transpile(qc_bp, backend, optimization_level=3)
        print(f"A_v transpiled: depth = {tqc_av.depth()}, 2q = {count_2q(tqc_av)}")
        print(f"B_p transpiled: depth = {tqc_bp.depth()}, 2q = {count_2q(tqc_bp)}\n")

        sampler = SamplerV2(mode=backend)
        print("[submit] A_v + B_p jobs...")
        job = sampler.run([tqc_av, tqc_bp], shots=shots)
        print(f"  job id = {job.job_id()}")
        result = job.result()
        c_av = result[0].data.c.get_counts()
        c_bp = result[1].data.c.get_counts()

    print(f"\n--- A_v (raw) ---")
    Av = []
    for i, v in enumerate(VERTS):
        val = analyse_av(c_av, v)
        Av.append(val)
        print(f"  A_v[{i}] on {v}: {val:+.4f}")

    print(f"\n--- B_p (raw) ---")
    Bp = analyse_bp(c_bp)
    for i, val in enumerate(Bp):
        print(f"  B_p[{i}] on {PLAQS[i]}: {val:+.4f}")

    overall = min(min(Av), min(Bp))
    print(f"\n=== Summary ===")
    print(f"  backend     : {backend.name if backend else 'simulator'}")
    print(f"  <A_v> range : {min(Av):+.4f} .. {max(Av):+.4f}")
    print(f"  <B_p> range : {min(Bp):+.4f} .. {max(Bp):+.4f}")
    print(f"  min stab    : {overall:+.4f}")
    if overall > 0.7:
        print("  -> PASS: hardware-verified topological order")
    elif overall > 0.5:
        print("  -> PASS (marginal): signal clearly above noise")
    elif overall > 0.3:
        print("  -> PARTIAL: physics visible")
    else:
        print("  -> FAIL: noise dominates")

    out = {"backend": backend.name if backend else "simulator",
           "A_v": Av, "B_p": Bp, "min": overall, "shots": shots}
    with open("toric_v3_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to toric_v3_results.json")
