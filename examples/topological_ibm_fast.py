"""L=2 toric code ground state on IBM hardware -- shallow prep.

Circuit:
  1. H on all 8 data qubits -> |+>^8  (+1 eigenstate of all A_v)
  2. For each plaquette, use one ancilla to measure B_p
     and feed-forward a conditional X correction.
  3. Result: +1 eigenstate of all A_v AND all B_p = toric GS.

Depth: ~25 layers (vs 1325 for StatePreparation).
Uses 8 data + 4 ancilla = 12 qubits.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

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

print(f"L={L}: {N} data qubits, {len(PLAQS)} plaquettes, "
      f"{len(VERTS)} vertices, {len(PLAQS)} ancilla")


# ------------------------------------------------------------------
# Shallow state prep with dynamic circuits
# ------------------------------------------------------------------
def toric_prep():
    """Prepare one toric code ground state with feed-forward."""
    data = QuantumRegister(N, "d")
    anc = QuantumRegister(len(PLAQS), "a")
    c_anc = ClassicalRegister(len(PLAQS), "ca")
    qc = QuantumCircuit(data, anc, c_anc)

    # 1. |+>^8 (this is +1 eigenstate of all A_v)
    for q in data:
        qc.h(q)

    # 2. Measure each B_p via ancilla and correct
    for p_idx, plaq in enumerate(PLAQS):
        for q_idx in plaq:
            qc.cx(data[q_idx], anc[p_idx])
        qc.measure(anc[p_idx], c_anc[p_idx])
        with qc.if_test((c_anc[p_idx], 1)):
            qc.x(data[plaq[0]])

    return qc, data, c_anc


# ------------------------------------------------------------------
# Append a stabilizer measurement to the prep circuit
# ------------------------------------------------------------------
def with_stab_measurement(qubits, kind):
    """Return a circuit that preps |GS> then measures <kind on qubits>."""
    data = QuantumRegister(N, "d")
    anc = QuantumRegister(len(PLAQS), "a")
    c_anc = ClassicalRegister(len(PLAQS), "ca")
    c_data = ClassicalRegister(N, "cd")
    qc = QuantumCircuit(data, anc, c_anc, c_data)

    # State prep (identical to toric_prep)
    for q in data:
        qc.h(q)
    for p_idx, plaq in enumerate(PLAQS):
        for q_idx in plaq:
            qc.cx(data[q_idx], anc[p_idx])
        qc.measure(anc[p_idx], c_anc[p_idx])
        with qc.if_test((c_anc[p_idx], 1)):
            qc.x(data[plaq[0]])

    # Stabilizer measurement
    if kind == "X":
        for q in qubits:
            qc.h(data[q])
    for q in qubits:
        qc.measure(data[q], c_data[q])

    return qc


# ------------------------------------------------------------------
# Simulator + hardware runner
# ------------------------------------------------------------------
def run_simulator(qc, shots=8192):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim, optimization_level=2)
    return sim.run(tqc, shots=shots).result().get_counts()


def run_hardware(qc, backend, shots=8192):
    from qiskit_ibm_runtime import SamplerV2
    tqc = transpile(qc, backend, optimization_level=3)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    print(f"    job {job.job_id()}")
    return job.result()[0].data.cd.get_counts()


def parity_from_counts(counts, qubits):
    """<product of Z on `qubits`> from a Z-basis measurement."""
    ev = 0.0
    total = 0
    for bitstr, n in counts.items():
        bits = bitstr.replace(" ", "")
        # Qiskit bit strings: rightmost char = qubit 0
        par = 0
        for q in qubits:
            par += int(bits[len(bits) - 1 - q])
        ev += ((-1) ** (par % 2)) * n
        total += n
    return ev / total


def connect_ibm():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        print("[mode] simulator")
        return None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("[warn] runtime missing -> simulator")
        return None
    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        print("[warn] token missing -> simulator")
        return None
    service = None
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] connected via {ch}")
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
        print(f"[ibm] fallback to {backend.name}")
    st = backend.status()
    print(f"[ibm] backend = {backend.name}  qubits={backend.num_qubits}  "
          f"queue={st.pending_jobs}")
    return backend


if __name__ == "__main__":
    print("\n=== L=2 toric code on IBM hardware (shallow prep) ===\n")

    # Show depth of prep
    prep_circ, _, _ = toric_prep()
    print(f"Prep circuit depth (raw) = {prep_circ.depth()}")
    print(f"Prep circuit ops         = {dict(prep_circ.count_ops())}")

    backend = connect_ibm()
    shots = 8192

    # Compute per-plaquette and per-vertex stabilizers
    results = {"A_v": [], "B_p": []}

    print(f"\nPlaquette stabilizers <B_p> ({shots} shots each):")
    for i, plaq in enumerate(PLAQS):
        qc = with_stab_measurement(plaq, "Z")
        if backend is None:
            counts = run_simulator(qc, shots)
        else:
            print(f"  submitting B_p[{i}]...", flush=True)
            counts = run_hardware(qc, backend, shots)
        ev = parity_from_counts(counts, plaq)
        results["B_p"].append(ev)
        print(f"    B_p[{i}] on {plaq}:  <B_p> = {ev:+.4f}")

    print(f"\nVertex stabilizers <A_v> ({shots} shots each):")
    for i, star in enumerate(VERTS):
        qc = with_stab_measurement(star, "X")
        if backend is None:
            counts = run_simulator(qc, shots)
        else:
            print(f"  submitting A_v[{i}]...", flush=True)
            counts = run_hardware(qc, backend, shots)
        ev = parity_from_counts(counts, star)
        results["A_v"].append(ev)
        print(f"    A_v[{i}] on {star}:  <A_v> = {ev:+.4f}")

    out = {
        "backend": backend.name if backend else "simulator",
        "shots": shots,
        "A_v": results["A_v"],
        "B_p": results["B_p"],
    }
    with open("toric_ibm_fast_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary ===")
    print(f"  backend  : {out['backend']}")
    print(f"  <A_v>    : {min(results['A_v']):+.4f} .. {max(results['A_v']):+.4f}")
    print(f"  <B_p>    : {min(results['B_p']):+.4f} .. {max(results['B_p']):+.4f}")
    print(f"  min      : {min(min(results['A_v']), min(results['B_p'])):+.4f}")
    print(f"\nResults written to toric_ibm_fast_results.json")
