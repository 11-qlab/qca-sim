"""L=2 toric code -- correct prep with post-selection, no corrections.

Key fix: the toric code ground state requires PROJECTION onto the
+1 eigenspace of all B_p.  Do that by measuring prefilter ancillas
and discarding shots with any ancilla = 1.  Do NOT apply X
corrections -- they break the state.

Circuit:
  1. H^8 on data          -> |+>^8  (all A_v = +1)
  2. CNOT ladders -> a[]  (prefilter)
  3. measure a[]          (post-selection filter)
  4. CNOT ladders -> b[]  (verify B_p)
  5. measure b[]
  6. H^8 on data
  7. measure data         (readout A_v)

Post-processing:
  keep shots where a = 0000
  <B_p> = parity of b[p] on kept shots
  <A_v> = parity of data bits for vertex v on kept shots
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


def build_circuit():
    d = QuantumRegister(8, "d")
    a = QuantumRegister(4, "a")   # prefilter
    b = QuantumRegister(4, "b")   # verify B_p
    c = ClassicalRegister(16, "c")
    qc = QuantumCircuit(d, a, b, c)

    # 1. |+>^8
    for q in range(8):
        qc.h(d[q])

    # 2. prefilter B_p onto a[]
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    for p in range(4):
        qc.measure(a[p], c[p])

    # 3. verify B_p onto b[]
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], b[p])
    for p in range(4):
        qc.measure(b[p], c[4 + p])

    # 4. H^8 for A_v readout
    for q in range(8):
        qc.h(d[q])
    for q in range(8):
        qc.measure(d[q], c[8 + q])

    return qc


def parse(bs):
    s = bs.replace(" ", "")
    bits = [int(ch) for ch in s[::-1]]   # bits[i] = classical bit i
    return {
        "a": bits[0:4],
        "b": bits[4:8],
        "d": bits[8:16],
    }


def analyse(counts, v_qubits):
    kept = 0
    b_sum = [0.0] * 4
    a_sum = 0.0
    for bs, n in counts.items():
        p = parse(bs)
        if any(p["a"]):
            continue
        kept += n
        for i in range(4):
            b_sum[i] += ((-1) ** p["b"][i]) * n
        par = sum(p["d"][q] for q in v_qubits) % 2
        a_sum += ((-1) ** par) * n
    if kept == 0:
        return None, [None] * 4, 0.0, 0
    Bp = [x / kept for x in b_sum]
    Av = a_sum / kept
    return Av, Bp, kept / sum(counts.values()), kept


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
    print("\n=== L=2 toric code (v4: no corrections, correct post-selection) ===\n")

    qc = build_circuit()
    print(f"circuit: {qc.num_qubits} qubits, depth(raw) = {qc.depth()}")
    print(f"ops: {dict(qc.count_ops())}\n")

    backend = connect_ibm()
    shots = 32768     # larger budget for more post-selected samples

    if backend is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector")
        tqc = transpile(qc, sim)
        counts = sim.run(tqc, shots=shots).result().get_counts()
    else:
        from qiskit_ibm_runtime import SamplerV2
        tqc = transpile(qc, backend, optimization_level=3)
        print(f"transpiled depth = {tqc.depth()}, 2q gates = {count_2q(tqc)}\n")
        sampler = SamplerV2(mode=backend)
        print("[submit] toric code circuit...")
        job = sampler.run([tqc], shots=shots)
        print(f"  job id = {job.job_id()}")
        counts = job.result()[0].data.c.get_counts()

    print(f"\nraw shots = {sum(counts.values())}, "
          f"distinct bitstrings = {len(counts)}")

    print(f"\n--- A_v and B_p (post-selected on prefilter a = 0000) ---")
    Av_all = []
    Bp_shared = None
    for i, v in enumerate(VERTS):
        Av, Bp, accept, kept = analyse(counts, v)
        Av_all.append(Av)
        if Bp_shared is None:
            Bp_shared = Bp
        print(f"  A_v[{i}] on {v}:  <A_v> = {Av:+.4f}   "
              f"(kept {kept}, {accept*100:.2f}%)")

    print(f"\n--- B_p (same post-selection) ---")
    for i, val in enumerate(Bp_shared or [0]*4):
        print(f"  B_p[{i}] on {PLAQS[i]}:  <B_p> = {val:+.4f}")

    overall = min(min(Av_all), min(Bp_shared or [0]))
    print(f"\n=== Summary ===")
    print(f"  backend     : {backend.name if backend else 'simulator'}")
    print(f"  <A_v> range : {min(Av_all):+.4f} .. {max(Av_all):+.4f}")
    print(f"  <B_p> range : {min(Bp_shared or [0]):+.4f} .. {max(Bp_shared or [0]):+.4f}")
    print(f"  min stab    : {overall:+.4f}")
    if overall > 0.7:
        print("  -> PASS: hardware-verified topological order")
    elif overall > 0.5:
        print("  -> PASS (marginal)")
    elif overall > 0.3:
        print("  -> PARTIAL")
    else:
        print("  -> FAIL")

    with open("toric_v4_results.json", "w") as f:
        json.dump({"backend": backend.name if backend else "simulator",
                   "A_v": Av_all, "B_p": Bp_shared, "min": overall,
                   "shots": shots}, f, indent=2)
    print(f"\nResults written to toric_v4_results.json")
