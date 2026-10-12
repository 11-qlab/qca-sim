"""Toric code ground state on IBM Kingston -- direct state prep.

Strategy:
  1. Build L=2 toric code Hamiltonian (8 qubits).
  2. Exact diagonalization -> one specific ground state |GS>.
  3. StatePreparation circuit for |GS>.
  4. Transpile + run on backend.
  5. Measure all stabilizers <A_v>, <B_p>.  Ideal = +1.

No VQE.  Direct state prep is exact and shallow enough for
Kingston (transpiled depth ~ hundreds, still within T2 budget).

Environment:
    QHAL_USE_IBM=1
    QHAL_IBM_BACKEND=ibm_kingston
    QISKIT_IBM_TOKEN=<token>
    QISKIT_IBM_CHANNEL=ibm_quantum  (or ibm_cloud)
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import StatePreparation


# ------------------------------------------------------------------
# L=2 toric code
# ------------------------------------------------------------------
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

print(f"L={L}: {N} qubits, {len(PLAQS)} plaquettes, {len(VERTS)} vertices")


# ------------------------------------------------------------------
# Hamiltonian (dense, 8 qubits -> 256-dim is fine)
# ------------------------------------------------------------------
def pauli_dense(label: str) -> np.ndarray:
    """label is little-endian: rightmost char = qubit 0."""
    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0, 1], [1, 0]], dtype=complex)
    Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z  = np.array([[1, 0], [0, -1]], dtype=complex)
    P = {"I": I2, "X": X, "Y": Y, "Z": Z}
    # Qiskit Statevector convention: tensor product in order q_{n-1} x ... x q_0
    op = np.array([[1.0 + 0j]])
    for ch in label:      # leftmost char = highest qubit
        op = np.kron(op, P[ch])
    return op


def stabilizer_label(qubits, kind):
    """Build a Pauli-string label.  kind = 'Z' (plaquette) or 'X' (vertex).
    Rightmost char = qubit 0."""
    chars = ["I"] * N
    for q in qubits:
        chars[N - 1 - q] = kind       # <-- corrected indexing
    return "".join(chars)


def build_hamiltonian() -> np.ndarray:
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for plaq in PLAQS:
        H -= pauli_dense(stabilizer_label(plaq, "Z"))
    for star in VERTS:
        H -= pauli_dense(stabilizer_label(star, "X"))
    return H


# ------------------------------------------------------------------
# Exact ground state
# ------------------------------------------------------------------
def ground_state() -> np.ndarray:
    H = build_hamiltonian()
    w, v = np.linalg.eigh(H)
    # pick one ground state (there are 4)
    # choose the one with largest overlap with |+...+> to be canonical
    plus = np.ones(2 ** N, dtype=complex) / np.sqrt(2 ** N)
    candidates = v[:, :4]
    overlaps = np.abs(candidates.conj().T @ plus)
    best = int(np.argmax(overlaps))
    psi = candidates[:, best]
    # phase convention: first nonzero amplitude real positive
    nz = np.flatnonzero(np.abs(psi) > 1e-9)[0]
    phase = psi[nz] / abs(psi[nz])
    return psi / phase


# ------------------------------------------------------------------
# Stabilizer measurement circuits
# ------------------------------------------------------------------
def measure_stabilizer(qc_prep, qubits, kind, shots, backend):
    """Measure <product of Pauli kind on qubits>."""
    qc = qc_prep.copy()
    if kind == "X":
        for q in qubits:
            qc.h(q)
    # else: Z basis is default; nothing needed
    qc.measure_all()

    if backend is None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        result = sim.run(transpile(qc, sim), shots=shots).result()
    else:
        from qiskit_ibm_runtime import SamplerV2
        tqc = transpile(qc, backend, optimization_level=2)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=shots)
        print(f"    [job] {job.job_id()}")
        result = job.result()

    counts = result.get_counts() if backend is None else result[0].data.meas.get_counts()

    # Parity of the target qubits
    ev = 0.0
    for bitstr, n in counts.items():
        bits = bitstr.replace(" ", "")
        # Qiskit strings are big-endian: leftmost = highest qubit
        # rightmost char = qubit 0
        parity = 0
        for q in qubits:
            parity += int(bits[len(bits) - 1 - q])
        ev += ((-1) ** (parity % 2)) * n
    return ev / shots


# ------------------------------------------------------------------
# IBM connection
# ------------------------------------------------------------------
def connect():
    if os.environ.get("QHAL_USE_IBM", "0") != "1":
        print("[mode] simulator")
        return None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("[warn] qiskit-ibm-runtime missing -> simulator")
        return None
    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        print("[warn] QISKIT_IBM_TOKEN missing -> simulator")
        return None

    # Try each known channel until one works
    channels = []
    env_ch = os.environ.get("QISKIT_IBM_CHANNEL")
    if env_ch:
        channels.append(env_ch)
    for ch in ("ibm_quantum_platform", "ibm_cloud"):
        if ch not in channels:
            channels.append(ch)

    service = None
    last_err = None
    for ch in channels:
        try:
            service = QiskitRuntimeService(channel=ch, token=token)
            print(f"[ibm] connected via channel '{ch}'")
            break
        except Exception as e:
            last_err = e
            print(f"[ibm] channel '{ch}' failed: {e}")
    if service is None:
        print(f"[warn] all channels failed, last error: {last_err}")
        print("       falling back to simulator")
        return None

    name = os.environ.get("QHAL_IBM_BACKEND", "ibm_kingston")
    try:
        backend = service.backend(name)
    except Exception as e:
        print(f"[warn] {name} not accessible: {e}")
        backend = service.least_busy(operational=True, simulator=False)
        print(f"[ibm] using least_busy: {backend.name}")

    status = backend.status()
    print(f"[ibm] backend = {backend.name}  qubits={backend.num_qubits}  "
          f"queue={status.pending_jobs}")
    return backend


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== L=2 toric code on IBM Kingston ===\n")

    print("Building Hamiltonian...")
    H = build_hamiltonian()
    print("Computing exact ground state...")
    psi = ground_state()
    E0 = float(np.real(psi.conj() @ H @ psi))
    print(f"Exact ground energy = {E0:.6f}  (theory: -8.000000)")

    print("\nPreparing StatePreparation circuit...")
    prep = QuantumCircuit(N)
    prep.append(StatePreparation(psi), range(N))
    print(f"Prep circuit depth (raw) = {prep.depth()}")
    print(f"Prep circuit ops         = {dict(prep.count_ops())}")

    backend = connect()

    if backend is not None:
        print("\nTranspiling for backend...")
        tqc = transpile(prep, backend, optimization_level=3)
        print(f"  transpiled depth = {tqc.depth()}")
        print(f"  CX count         = {tqc.count_ops().get('cx', 0)}")
        # Reuse the transpiled prep for stabilizer measurements
        prep_for_meas = tqc
    else:
        # simulator: use raw prep
        prep_for_meas = prep

    shots = 4096
    print(f"\nMeasuring stabilizers ({shots} shots each)...")

    # On hardware, we need to re-transpile per measurement (H layers).
    # But the H layers are on single qubits; we can just append and re-transpile.
    def prep_circuit():
        qc = QuantumCircuit(N)
        qc.append(StatePreparation(psi), range(N))
        return qc

    results = {"A_v": [], "B_p": []}

    print("\nPlaquette stabilizers <B_p>:")
    for i, plaq in enumerate(PLAQS):
        ev = measure_stabilizer(prep_circuit(), plaq, "Z", shots, backend)
        results["B_p"].append(ev)
        print(f"  B_p[{i}] on qubits {plaq}: <B_p> = {ev:+.4f}")

    print("\nVertex stabilizers <A_v>:")
    for i, star in enumerate(VERTS):
        ev = measure_stabilizer(prep_circuit(), star, "X", shots, backend)
        results["A_v"].append(ev)
        print(f"  A_v[{i}] on qubits {star}: <A_v> = {ev:+.4f}")

    out = {
        "backend": backend.name if backend else "simulator",
        "E_exact": E0,
        "A_v": results["A_v"],
        "B_p": results["B_p"],
        "shots": shots,
        "N_qubits": N,
    }
    with open("toric_ibm_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary ===")
    print(f"  backend     : {out['backend']}")
    print(f"  E_exact     : {out['E_exact']:+.6f}")
    print(f"  <A_v> range : {min(results['A_v']):+.4f} .. {max(results['A_v']):+.4f}")
    print(f"  <B_p> range : {min(results['B_p']):+.4f} .. {max(results['B_p']):+.4f}")
    print(f"  min         : {min(min(results['A_v']), min(results['B_p'])):+.4f}")
    print(f"\nResults written to toric_ibm_results.json")
