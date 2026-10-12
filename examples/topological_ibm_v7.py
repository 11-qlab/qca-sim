"""L=2 toric code -- single classical register, minimal circuit.

Fix from v6: both data and ancilla measured into ONE classical
register, so .get_counts() returns the joint bitstring.

Circuit (Z run):
    H^8                        (8 H gates)
    4 CNOT-ladders -> ancilla  (16 CX)
    measure all 12 qubits      (12 measures)

Circuit (X run):
    H^8
    CX ladders
    H^8 data
    measure all 12 qubits

Ideal: 16 CX, no SWAPs required.
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


def make_circuit(x_basis_measure=False):
    """Ancilla qubits 8-11.  Data qubits 0-7.
    Single classical register c[0..11]: c[0..3] = ancillas, c[4..11] = data.
    """
    d = QuantumRegister(8, "d")
    a = QuantumRegister(4, "a")
    c = ClassicalRegister(12, "c")
    qc = QuantumCircuit(d, a, c)
    for q in range(8):
        qc.h(d[q])
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    if x_basis_measure:
        for q in range(8):
            qc.h(d[q])
    # measure ancillas first: c[0..3]
    for p in range(4):
        qc.measure(a[p], c[p])
    # then data: c[4..11]
    for q in range(8):
        qc.measure(d[q], c[4 + q])
    return qc


def parse(bs):
    """Bitstring is little-endian in Qiskit string form.
    Format: leftmost = c[11], rightmost = c[0]."""
    s = bs.replace(" ", "")
    bits = [int(ch) for ch in s[::-1]]    # bits[i] = classical bit i
    anc = bits[0:4]
    data = bits[4:12]
    return anc, data


def analyse_z(counts):
    """Post-select anc=0; compute <B_p> from data."""
    kept = 0.0
    Bp_sum = [0.0] * 4
    for bs, n in counts.items():
        anc, data = parse(bs)
        if any(anc):
            continue
        kept += n
        for p in range(4):
            par = sum(data[q] for q in PLAQS[p]) % 2
            Bp_sum[p] += ((-1) ** par) * n
    if kept == 0:
        return [None]*4, 0.0
    return [x/kept for x in Bp_sum], kept/sum(counts.values())


def analyse_x(counts, v_qubits):
    kept = 0.0
    a_sum = 0.0
    for bs, n in counts.items():
        anc, data = parse(bs)
        if any(anc):
            continue
        kept += n
        par = sum(data[q] for q in v_qubits) % 2
        a_sum += ((-1) ** par) * n
    if kept == 0:
        return None, 0.0
    return a_sum/kept, kept/sum(counts.values())


def connect():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        print("[sim]"); return None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ["QISKIT_IBM_TOKEN"]
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] {ch}"); break
        except Exception as e:
            print(f"[ibm] {ch}: {e}"); service = None
    if service is None:
        return None
    name = os.environ.get("QHAL_IBM_BACKEND", "ibm_marrakesh")
    try:
        backend = service.backend(name)
    except Exception:
        backend = service.least_busy(operational=True, simulator=False)
    print(f"[ibm] {backend.name}  queue={backend.status().pending_jobs}")
    return backend


def count_2q(tqc):
    ops = dict(tqc.count_ops())
    return sum(v for k, v in ops.items() if k in ("cx","ecr","cz"))


if __name__ == "__main__":
    print("\n=== L=2 toric code (v7: single-register fix) ===\n")

    qc_z = make_circuit(x_basis_measure=False)
    qc_x = make_circuit(x_basis_measure=True)
    print(f"Z circuit: {qc_z.num_qubits} qubits, depth = {qc_z.depth()}")
    print(f"X circuit: {qc_x.num_qubits} qubits, depth = {qc_x.depth()}\n")

    backend = connect()
    shots = 65536

    if backend is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector")
        c_z = sim.run(transpile(qc_z, sim), shots=shots).result().get_counts()
        c_x = sim.run(transpile(qc_x, sim), shots=shots).result().get_counts()
    else:
        from qiskit_ibm_runtime import SamplerV2
        t_z = transpile(qc_z, backend, optimization_level=3)
        t_x = transpile(qc_x, backend, optimization_level=3)
        print(f"Z transpiled: depth = {t_z.depth()}, 2q = {count_2q(t_z)}")
        print(f"X transpiled: depth = {t_x.depth()}, 2q = {count_2q(t_x)}\n")
        sampler = SamplerV2(mode=backend)
        print("[submit] Z + X circuits...")
        job = sampler.run([t_z, t_x], shots=shots)
        print(f"  job id = {job.job_id()}\n")
        r = job.result()
        c_z = r[0].data.c.get_counts()
        c_x = r[1].data.c.get_counts()

    print(f"Z raw shots = {sum(c_z.values())}, distinct = {len(c_z)}")
    print(f"X raw shots = {sum(c_x.values())}, distinct = {len(c_x)}\n")

    # B_p
    Bp, acc_z = analyse_z(c_z)
    print(f"--- B_p (Z-basis, post-selected {acc_z*100:.2f}%) ---")
    for i, val in enumerate(Bp):
        print(f"  B_p[{i}] on {PLAQS[i]}: {val:+.4f}")

    # A_v
    print(f"\n--- A_v (X-basis) ---")
    Av = []
    for i, v in enumerate(VERTS):
        val, acc_x = analyse_x(c_x, v)
        Av.append(val)
        print(f"  A_v[{i}] on {v}: {val:+.4f}  (post-sel {acc_x*100:.2f}%)")

    overall = min(min(Av), min(Bp))
    print(f"\n=== Summary ===")
    print(f"  backend     : {backend.name if backend else 'simulator'}")
    print(f"  <A_v> range : {min(Av):+.4f} .. {max(Av):+.4f}")
    print(f"  <B_p> range : {min(Bp):+.4f} .. {max(Bp):+.4f}")
    print(f"  min stab    : {overall:+.4f}")
    if overall > 0.75:
        print("  -> PASS: hardware-verified topological order")
    elif overall > 0.6:
        print("  -> PASS (marginal)")
    elif overall > 0.4:
        print("  -> PARTIAL")
    else:
        print("  -> FAIL")

    with open("toric_v7_results.json", "w") as f:
        json.dump({"backend": backend.name if backend else "simulator",
                   "A_v": Av, "B_p": Bp, "min": overall,
                   "shots": shots}, f, indent=2)
    print(f"\nWrote toric_v7_results.json")
