"""L=2 toric code ground state on IBM -- single-circuit post-selection.

Circuit (16 qubits: 8 data + 4 prefilter + 4 verify):
  1. H on all 8 data -> |+>^8 (this IS +1 eigenspace of all A_v)
  2. CNOT-ladder for each B_p onto prefilter ancillas a[0..3]
  3. CNOT-ladder for each B_p onto verify ancillas b[0..3]
  4. H on all 8 data (rotate X basis into Z for readout)
  5. Measure everything.

Post-process:
  - Keep shots where a = 0000 (state is in all B_p = +1 eigenspace)
  - <B_p> = parity of b[p] in surviving shots (should be +1)
  - <A_v> = parity of the 4 data bits for vertex v (should be +1)
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
    a = QuantumRegister(4, "a")   # prefilter ancillas
    b = QuantumRegister(4, "b")   # verify ancillas
    c = ClassicalRegister(16, "c")
    qc = QuantumCircuit(d, a, b, c)

    # 1. |+>^8
    for q in range(8):
        qc.h(d[q])

    # 2. Prefilter B_p onto a[]
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    for p in range(4):
        qc.measure(a[p], c[p])

    # 3. Verify B_p onto b[]
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], b[p])
    for p in range(4):
        qc.measure(b[p], c[4 + p])

    # 4. H on data for X-basis A_v measurement
    for q in range(8):
        qc.h(d[q])
    for q in range(8):
        qc.measure(d[q], c[8 + q])

    return qc


def parse_bitstring(bs, nbits=16):
    """Return dict: a[0..3], b[0..3], d[0..7]."""
    bits = [int(ch) for ch in bs[::-1]]     # bits[i] = classical bit i
    return {
        "a": bits[0:4],
        "b": bits[4:8],
        "d": bits[8:16],
    }


def analyse(counts, target_A_v_qubits):
    """Post-select a = 0, compute <B_p> and <A_v>."""
    kept = 0
    Bp_sum = [0.0] * 4
    Av_sum = 0.0
    for bs, n in counts.items():
        s = bs.replace(" ", "")
        p = parse_bitstring(s)
        if any(p["a"]):
            continue
        kept += n
        for i in range(4):
            Bp_sum[i] += ((-1) ** p["b"][i]) * n
        par = sum(p["d"][q] for q in target_A_v_qubits) % 2
        Av_sum += ((-1) ** par) * n
    if kept == 0:
        return None, [None]*4, None, 0
    Bp = [x / kept for x in Bp_sum]
    Av = Av_sum / kept
    return Av, Bp, kept / sum(counts.values()), kept


def run_sim(qc, shots):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim, optimization_level=2)
    return sim.run(tqc, shots=shots).result().get_counts()


def run_ibm(qc, backend, shots):
    from qiskit_ibm_runtime import SamplerV2
    tqc = transpile(qc, backend, optimization_level=3)
    print(f"  transpiled depth = {tqc.depth()}, "
          f"CX = {tqc.count_ops().get('cx', 0)}")
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    print(f"  job id = {job.job_id()}")
    return job.result()[0].data.c.get_counts()


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


if __name__ == "__main__":
    print("=== L=2 toric code (single-circuit, post-selection) ===\n")

    qc = build_circuit()
    print(f"circuit: {qc.num_qubits} qubits, depth(raw) = {qc.depth()}")
    print(f"ops: {dict(qc.count_ops())}\n")

    backend = connect_ibm()
    shots = 8192

    if backend is None:
        counts = run_sim(qc, shots)
    else:
        counts = run_ibm(qc, backend, shots)

    print(f"\ntotal shots returned = {sum(counts.values())}")
    print(f"distinct bitstrings  = {len(counts)}")

    print("\nPer-vertex A_v measurements:")
    A_v_all = []
    for v_idx_, v_qubits in enumerate(VERTS):
        Av, Bp, accept, kept = analyse(counts, v_qubits)
        A_v_all.append(Av)
        print(f"  A_v[{v_idx_}] on {v_qubits}:  <A_v> = {Av:+.4f}   "
              f"(kept {kept} shots, {accept*100:.1f}%)")

    # Reuse the last analysis for B_p (same post-selection)
    _, Bp_all, _, _ = analyse(counts, VERTS[0])

    print("\nPer-plaquette B_p (from verify ancillas):")
    for i, val in enumerate(Bp_all):
        print(f"  B_p[{i}] on {PLAQS[i]}:  <B_p> = {val:+.4f}")

    min_stab = min(min(A_v_all), min(Bp_all))
    print(f"\nSummary:")
    print(f"  <A_v> range : {min(A_v_all):+.4f} .. {max(A_v_all):+.4f}")
    print(f"  <B_p> range : {min(Bp_all):+.4f} .. {max(Bp_all):+.4f}")
    print(f"  min         : {min_stab:+.4f}")
    if min_stab > 0.7:
        print("  -> PASS: topological order survives on hardware")
    elif min_stab > 0.3:
        print("  -> MARGINAL: partial verification, errors visible")
    else:
        print("  -> FAIL: state prep or readout corrupted")

    with open("toric_v2_results.json", "w") as f:
        json.dump({"A_v": A_v_all, "B_p": Bp_all,
                   "min": min_stab,
                   "backend": backend.name if backend else "simulator"}, f, indent=2)
    print("\nResults written to toric_v2_results.json")
