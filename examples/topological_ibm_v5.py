"""L=2 toric code on IBM -- with fixed layout + readout mitigation.

Fixes v4's 4x transpilation overhead:
  1. Explicit initial_layout places data + ancillas on adjacent qubits.
  2. Readout error calibration + reweighting.
  3. Reuses same 16-qubit circuit structure.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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
    a = QuantumRegister(4, "a")
    b = QuantumRegister(4, "b")
    c = ClassicalRegister(16, "c")
    qc = QuantumCircuit(d, a, b, c)
    for q in range(8):
        qc.h(d[q])
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    for p in range(4):
        qc.measure(a[p], c[p])
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], b[p])
    for p in range(4):
        qc.measure(b[p], c[4 + p])
    for q in range(8):
        qc.h(d[q])
    for q in range(8):
        qc.measure(d[q], c[8 + q])
    return qc


def calibration_circuits():
    qc0 = QuantumCircuit(16, 16)
    for q in range(16):
        qc0.measure(q, q)
    qc1 = QuantumCircuit(16, 16)
    for q in range(16):
        qc1.x(q)
    for q in range(16):
        qc1.measure(q, q)
    return qc0, qc1


def fit_layout(backend):
    """Pick 16 physical qubits forming a connected cluster."""
    # Try to find a connected 16-node subgraph of the coupling map
    cm = backend.coupling_map
    if cm is None:
        return None
    adj = {}
    for u, v in cm.get_edges():
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Greedy BFS from the qubit with best readout error
    props = backend.properties()
    readout_err = {q: (1.0 if props is None
                       else props.readout_error(q)) for q in adj}
    start = min(readout_err, key=readout_err.get)
    cluster = [start]
    seen = {start}
    frontier = [start]
    while len(cluster) < 16 and frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen:
                    cluster.append(v); seen.add(v); nxt.append(v)
                    if len(cluster) >= 16:
                        break
            if len(cluster) >= 16:
                break
        frontier = nxt
    return cluster if len(cluster) >= 16 else None


def calibrate(backend, shots=8192):
    from qiskit_ibm_runtime import SamplerV2
    qc0, qc1 = calibration_circuits()
    sampler = SamplerV2(mode=backend)
    tqc0 = transpile(qc0, backend, optimization_level=2, initial_layout=LAYOUT)
    tqc1 = transpile(qc1, backend, optimization_level=2, initial_layout=LAYOUT)
    r = sampler.run([tqc0, tqc1], shots=shots).result()
    c0 = r[0].data.c.get_counts()
    c1 = r[1].data.c.get_counts()
    p00 = np.zeros(16); p11 = np.zeros(16)
    for bs, n in c0.items():
        bits = bs.replace(" ", "")
        for q in range(16):
            if bits[15 - q] == "0":
                p00[q] += n
    for bs, n in c1.items():
        bits = bs.replace(" ", "")
        for q in range(16):
            if bits[15 - q] == "1":
                p11[q] += n
    return p00 / shots, p11 / shots


def analyse(counts, v_qubits, p00=None, p11=None):
    kept = 0.0
    b_sum = [0.0]*4
    a_sum = 0.0
    for bs, n in counts.items():
        bits = [int(c) for c in bs.replace(" ","")[::-1]]
        a = bits[0:4]; b = bits[4:8]; d = bits[8:16]
        if any(a):
            continue
        w = n
        if p00 is not None:
            for q in range(16):
                bc = bits[q]
                p_correct = p00[q] if bc == 0 else p11[q]
                w *= 1.0 / max(p_correct, 0.5)
        kept += w
        for i in range(4):
            b_sum[i] += ((-1) ** b[i]) * w
        par = sum(d[q] for q in v_qubits) % 2
        a_sum += ((-1) ** par) * w
    if kept == 0:
        return None, [None]*4, 0.0, 0
    Bp = [x/kept for x in b_sum]
    Av = a_sum / kept
    return Av, Bp, kept / sum(counts.values()), kept


def connect():
    if os.environ.get("QHAL_USE_IBM","0") != "1":
        print("[sim]"); return None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ["QISKIT_IBM_TOKEN"]
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] channel = {ch}"); break
        except Exception as e:
            print(f"[ibm] {ch} failed: {e}"); service = None
    if service is None:
        return None
    name = os.environ.get("QHAL_IBM_BACKEND", "ibm_marrakesh")
    try:
        backend = service.backend(name)
    except Exception:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"[ibm] {backend.name}  qubits={backend.num_qubits}  "
          f"queue={backend.status().pending_jobs}")
    return backend


LAYOUT = None


if __name__ == "__main__":
    print("\n=== L=2 toric code (v5: layout + readout mitigation) ===\n")
    qc = build_circuit()
    print(f"circuit: {qc.num_qubits} qubits, raw depth = {qc.depth()}\n")

    backend = connect()

    if backend is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector")
        counts = sim.run(transpile(qc, sim), shots=32768).result().get_counts()
        p00 = p11 = None
    else:
        # Find a 16-qubit cluster
        LAYOUT = fit_layout(backend)
        if LAYOUT is None:
            print("[layout] could not find cluster, using default")
        else:
            print(f"[layout] chosen physical qubits: {LAYOUT}")

        pm = generate_preset_pass_manager(optimization_level=3,
                                          backend=backend,
                                          initial_layout=LAYOUT)
        tqc = pm.run(qc)
        ops = dict(tqc.count_ops())
        tq = sum(v for k, v in ops.items() if k in ("cx","ecr","cz"))
        print(f"[layout] transpiled depth = {tqc.depth()}, 2q gates = {tq}\n")

        # Readout calibration
        print("[calib] measuring readout errors...")
        p00, p11 = calibrate(backend, shots=8192)
        print(f"[calib] 0|0: min={p00.min():.3f} mean={p00.mean():.3f}")
        print(f"[calib] 1|1: min={p11.min():.3f} mean={p11.mean():.3f}\n")

        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        print("[submit] main circuit...")
        job = sampler.run([tqc], shots=32768)
        print(f"  job id = {job.job_id()}\n")
        counts = job.result()[0].data.c.get_counts()

    print(f"raw shots = {sum(counts.values())}\n")

    print("--- RAW (no mitigation) ---")
    Av_raw = []
    for i, v in enumerate(VERTS):
        Av, Bp, acc, kept = analyse(counts, v)
        Av_raw.append(Av)
    Bp_raw = analyse(counts, VERTS[0])[1]
    print(f"  A_v: " + " ".join(f"{x:+.4f}" for x in Av_raw))
    print(f"  B_p: " + " ".join(f"{x:+.4f}" for x in Bp_raw))
    print(f"  min: {min(min(Av_raw), min(Bp_raw)):+.4f}")

    if p00 is not None:
        print("\n--- MITIGATED (readout correction) ---")
        Av_mit = []
        for i, v in enumerate(VERTS):
            Av, Bp, acc, kept = analyse(counts, v, p00, p11)
            Av_mit.append(Av)
        Bp_mit = analyse(counts, VERTS[0], p00, p11)[1]
        print(f"  A_v: " + " ".join(f"{x:+.4f}" for x in Av_mit))
        print(f"  B_p: " + " ".join(f"{x:+.4f}" for x in Bp_mit))
        final = min(min(Av_mit), min(Bp_mit))
        print(f"  min: {final:+.4f}")
        if final > 0.75:
            print("  -> PASS: hardware-verified topological order")
        elif final > 0.55:
            print("  -> PASS (marginal)")
        elif final > 0.35:
            print("  -> PARTIAL")
        else:
            print("  -> FAIL")
        with open("toric_v5_results.json", "w") as f:
            json.dump({"A_v_raw": Av_raw, "B_p_raw": Bp_raw,
                       "A_v_mit": Av_mit, "B_p_mit": Bp_mit,
                       "final_min": final,
                       "backend": backend.name}, f, indent=2)
    else:
        with open("toric_v5_results.json", "w") as f:
            json.dump({"A_v_raw": Av_raw, "B_p_raw": Bp_raw,
                       "min": min(min(Av_raw), min(Bp_raw)),
                       "backend": "simulator"}, f, indent=2)
    print("\nWrote toric_v5_results.json")
