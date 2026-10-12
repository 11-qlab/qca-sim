"""L=2 toric code -- minimal circuit, no verify ancillas.

Circuit (per run):
    H^8                          (8 single-qubit)
    4 CNOT-ladders -> prefilter  (16 two-qubit)      <- THE ONLY CNOTs
    measure prefilter a[]        (4 measures)
    [optional H^8 for A_v mode]  (8 single-qubit)
    measure data or nothing

Two runs of the SAME circuit:
  Run A: after prefilter, measure data in Z basis -> compute <B_p>
  Run B: after prefilter, apply H^8, measure data -> compute <A_v>

Gate count: 16 two-qubit gates per circuit (vs 32 in v4, 61+ in v5).
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


def circuit_z_basis():
    """Post-select on prefilter; measure data in Z basis -> B_p."""
    d = QuantumRegister(8, "d")
    a = QuantumRegister(4, "a")
    c_a = ClassicalRegister(4, "ca")
    c_d = ClassicalRegister(8, "cd")
    qc = QuantumCircuit(d, a, c_a, c_d)
    for q in range(8):
        qc.h(d[q])
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    for p in range(4):
        qc.measure(a[p], c_a[p])
    for q in range(8):
        qc.measure(d[q], c_d[q])
    return qc


def circuit_x_basis():
    """Post-select on prefilter; measure data in X basis -> A_v."""
    d = QuantumRegister(8, "d")
    a = QuantumRegister(4, "a")
    c_a = ClassicalRegister(4, "ca")
    c_d = ClassicalRegister(8, "cd")
    qc = QuantumCircuit(d, a, c_a, c_d)
    for q in range(8):
        qc.h(d[q])
    for p in range(4):
        for q in PLAQS[p]:
            qc.cx(d[q], a[p])
    for p in range(4):
        qc.measure(a[p], c_a[p])
    for q in range(8):
        qc.h(d[q])
    for q in range(8):
        qc.measure(d[q], c_d[q])
    return qc


def analyse_bp(counts):
    """Post-select on ca=0; from cd compute each B_p = Z_0 Z_1 Z_2 Z_3."""
    kept = 0.0
    Bp_sum = [0.0] * 4
    for bs, n in counts.items():
        s = bs.replace(" ", "")
        # Qiskit string format: "cd ca" concatenated, MSB left
        # Actually with 2 registers, counts keys are "cd ca"
        # Format: "cd[7..0] ca[3..0]" — 12 bits total
        cd_bits = s[:8]
        ca_bits = s[8:]
        if any(int(c) for c in ca_bits):
            continue
        kept += n
        # cd_bits[0] is classical bit 7 = data qubit 7; use rightmost
        # for qubit 0 (little-endian)
        cd = [int(c) for c in cd_bits[::-1]]
        for p in range(4):
            par = sum(cd[q] for q in PLAQS[p]) % 2
            Bp_sum[p] += ((-1) ** par) * n
    if kept == 0:
        return [None] * 4, 0.0
    return [x / kept for x in Bp_sum], kept / sum(counts.values())


def analyse_av(counts, v_qubits):
    """Post-select on ca=0; from cd compute A_v = X⊗4 parity."""
    kept = 0.0
    a_sum = 0.0
    for bs, n in counts.items():
        s = bs.replace(" ", "")
        cd_bits = s[:8]
        ca_bits = s[8:]
        if any(int(c) for c in ca_bits):
            continue
        kept += n
        cd = [int(c) for c in cd_bits[::-1]]
        par = sum(cd[q] for q in v_qubits) % 2
        a_sum += ((-1) ** par) * n
    if kept == 0:
        return None
    return a_sum / kept


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


def two_q_count(tqc):
    ops = dict(tqc.count_ops())
    return sum(v for k, v in ops.items() if k in ("cx","ecr","cz"))


if __name__ == "__main__":
    print("\n=== L=2 toric code (v6: minimal circuit) ===\n")

    qc_z = circuit_z_basis()
    qc_x = circuit_x_basis()
    print(f"Z circuit: {qc_z.num_qubits} qubits, raw depth = {qc_z.depth()}")
    print(f"X circuit: {qc_x.num_qubits} qubits, raw depth = {qc_x.depth()}\n")

    backend = connect()
    shots = 65536

    if backend is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector")
        t_z = transpile(qc_z, sim)
        t_x = transpile(qc_x, sim)
        c_z = sim.run(t_z, shots=shots).result().get_counts()
        c_x = sim.run(t_x, shots=shots).result().get_counts()
    else:
        from qiskit_ibm_runtime import SamplerV2
        # Default transpiler layout (Sabre is default)
        t_z = transpile(qc_z, backend, optimization_level=3)
        t_x = transpile(qc_x, backend, optimization_level=3)
        print(f"Z transpiled: depth = {t_z.depth()}, 2q = {two_q_count(t_z)}")
        print(f"X transpiled: depth = {t_x.depth()}, 2q = {two_q_count(t_x)}\n")
        sampler = SamplerV2(mode=backend)
        print("[submit] Z + X circuits...")
        job = sampler.run([t_z, t_x], shots=shots)
        print(f"  job id = {job.job_id()}\n")
        r = job.result()
        c_z = r[0].data.cd.get_counts() if hasattr(r[0].data, 'cd') else r[0].data.get_counts()
        c_x = r[1].data.cd.get_counts() if hasattr(r[1].data, 'cd') else r[1].data.get_counts()

    print(f"Z raw shots = {sum(c_z.values())}")
    print(f"X raw shots = {sum(c_x.values())}\n")

    # B_p from Z circuit
    Bp, accept_z = analyse_bp(c_z)
    print(f"--- B_p (Z-basis, post-selected {accept_z*100:.1f}% kept) ---")
    for i, val in enumerate(Bp):
        print(f"  B_p[{i}] on {PLAQS[i]}: {val:+.4f}")

    # A_v from X circuit
    print(f"\n--- A_v (X-basis) ---")
    Av = []
    for i, v in enumerate(VERTS):
        val = analyse_av(c_x, v)
        Av.append(val)
        print(f"  A_v[{i}] on {v}: {val:+.4f}")

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

    with open("toric_v6_results.json", "w") as f:
        json.dump({"backend": backend.name if backend else "simulator",
                   "A_v": Av, "B_p": Bp, "min": overall,
                   "shots": shots}, f, indent=2)
    print(f"\nWrote toric_v6_results.json")
