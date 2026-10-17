[![PyPI](https://img.shields.io/pypi/v/qca-sim.svg)](https://pypi.org/project/qca-sim/)

# QCA Quantum

**Quantum-dot Cellular Automata as a native quantum computing platform.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## What this is

For 30 years, quantum-dot cellular automata (QCA) has been treated as a
*classical* computing paradigm — bits encoded in electron positions, coupling
via electrostatic repulsion, clocked by four-phase electric fields.

This repository shows that framing is incomplete. **QCA is a native quantum
platform.** Every cell is a qubit. Every inter-cell coupling is an entangler.
Every QCA wire is a chain of two-qubit gates. And the ground state of an
N-cell QCA chain is a GHZ state, available for free.

## Eight findings

| # | Result | Evidence |
|---|---|---|
| F1 | QCA cells are entangled qubits, not classical bits | Bell fidelity 0.9975 at J/g = 20 |
| F2 | σz ⊗ σz coupling is a native CNOT | Concurrence 1.0000, truth table PASS |
| F3 | N-cell chain ground state is a GHZ resource | S(A) = ln 2, F_GHZ = 0.9999 |
| F4 | Clock rate is capped by Landau-Zener | T_adiabatic ~ 1/(2πΔ²), verified vs TDSE |
| F5 | STA breaks the clock speed limit | **5000× speedup** (T_edge 500 → 0.1) |
| F6 | MV3 parallel voter tolerates faults | 8/8 correct with 1 dead cell |
| F7 | Distance-3 surface code from native gates | [[9,1,3]], p_th ≈ 1e-2, 32 CZ/cycle |
| F8 | Logical Bell pair at F = 1.0000 | <Z_A Z_B> = <X_A X_B> = +1 |

## Install

```bash
pip install qca-sim
```

Or from source:

```bash
git clone https://github.com/11-qlab/qca-quantum
cd qca-quantum
pip install -e .
pip install qiskit qiskit-aer qiskit-ibm-runtime   # optional, for HW
```

Quickstart

Run the full stack end-to-end:

```bash
python3 examples/qca_quantum_processor.py
```

Run any single experiment:

```bash
python3 examples/qca_bell_gate.py          # F1, F2
python3 examples/qca_ghz_3cell.py          # F3
python3 examples/sta_clock.py              # F4, F5
python3 examples/mv3_fixed.py              # F6
python3 examples/qca_surface_code.py       # F7
python3 examples/logical_cnot.py           # F8
```

Submit to real IBM Quantum hardware:

```bash
export IBMQ_TOKEN=<your token from quantum.ibm.com>
python3 examples/ibm_run_all.py ibm_brisbane
```

Layout

```
examples/
  qca_quantum_processor.py     # full-stack demo
  qca_bell_gate.py             # native CZ/CNOT + Bell state
  qca_ghz_3cell.py             # GHZ from chain ground state
  sta_clock.py                 # counter-diabatic QCA clock
  sta_wire.py                  # STA at the wire level
  mv3_fixed.py                 # fault-tolerant majority gate
  qca_surface_code.py          # distance-3 surface code
  logical_cnot.py              # transversal logical CNOT
  molecular_qca_bisfc.py       # bis-ferrocene molecular cell
  qiskit_bell_real.py          # 2-qubit tomography on HW
  ibm_run_all.py               # Bell + GHZ submission to IBM
src/qca/
  cell_hubbard.py              # exact diagonalization of QCA cell
  __init__.py
docs/
  FINDINGS.md                  # technical writeup
```

Key results

STA clock speedup

T_edge Baseline P STA P
0.10 +1.0000 -0.9978
1.00 +0.9952 -0.9988
50.0 +0.7869 -0.9978
500.0 -0.4023 -0.9977

Baseline needs T_edge ~ 500 for a clean flip. STA achieves it at T_edge = 0.1.

MV3 parallel voter under fault

 Healthy 1 dead cell
Standard 5-cell majority 8/8 6/8
MV3 parallel voter (7 cells) 8/8 8/8

Molecular feasibility

Clock field E-field T_max P
20 meV 0.017 V/nm 259 K -0.89
100 meV 0.083 V/nm 1166 K -0.995
200 meV 0.167 V/nm 2324 K -0.999

Room-temperature operation is comfortable. Redox-safe up to ~0.5 V/nm.

Citation

If you use this work, please cite:

```
@misc{qca-quantum-2026,
  title  = {QCA as a Native Quantum Platform: 5000x Clock Speedup and
            Fault-Tolerant Quantum Computing on Molecular Cellular Automata},
  author = {11-qlab},
  year   = {2026},
  url    = {https://github.com/11-qlab/qca-quantum}
}
```

License

MIT
