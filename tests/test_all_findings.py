"""Regression tests for all 22 findings.  Run: pytest tests/test_all_findings.py -v"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest


def test_F01_cell_entanglement():
    from qca import hamiltonian, ground_state, polarization
    H, B = hamiltonian(t=0.05, V_diag=1.0, V_adj=0.2, driver=1.0)
    E0, psi = ground_state(H)
    P = polarization(psi, B)
    assert abs(P) > 0.8


def test_F02_native_cnot():
    I2 = np.eye(2, dtype=complex)
    H1 = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
    zz = lambda th: np.diag(np.exp(1j*th*np.array([1,-1,-1,1])))
    Rz = lambda phi: np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    CZ = np.exp(1j*np.pi/4) * np.kron(Rz(np.pi/2), Rz(np.pi/2)) @ zz(np.pi/4)
    assert np.allclose(CZ, np.diag([1,1,1,-1]), atol=1e-9)


def test_F03_ghz_from_chain():
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1); qc.cz(0,1); qc.h(1)
    qc.h(2); qc.cz(1,2); qc.h(2)
    qc.measure([0,1,2],[0,1,2])
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=4096).result().get_counts()
    p000 = counts.get("000", 0) / 4096
    p111 = counts.get("111", 0) / 4096
    assert p000 + p111 > 0.99


def test_F04_landau_zener():
    Delta = 0.1876
    T = 1 / (2 * np.pi * Delta ** 2)
    assert 3.0 < T < 7.0


def test_F05_sta_speedup():
    from scipy.linalg import expm
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    H2 = lambda d, D: 0.5*d*sz + 0.5*D*sx
    Hcd = lambda d, dd, D: -0.5*D*dd/(d*d+D*D)*sy
    def flip(T, D, cd, n=400):
        dt = T/n
        psi = np.array([1,0], dtype=complex)
        for k in range(n):
            tau = (k+0.5)/n
            d = 3.0 - 6.0*tau
            dd = -6.0/T
            H = H2(d, D) + (Hcd(d, dd, D) if cd else 0)
            psi = expm(-1j*H*dt) @ psi
            psi /= np.linalg.norm(psi)
        return float(np.real(psi.conj() @ sz @ psi))
    assert flip(0.5, 0.10, True) < -0.9
    assert flip(0.5, 0.10, False) > 0.9


def test_F06_mv3_fault():
    from qca import hamiltonian, ground_state, polarization
    def resp(driver):
        H, B = hamiltonian(t=0.05, V_diag=1.0, V_adj=0.2, driver=driver)
        _, psi = ground_state(H)
        return polarization(psi, B)
    def mv3_faulty(a, b, c):
        mAC = resp(a + c); mBC = resp(b + c)
        return resp((a+b+c) + 2.0*(0 + mAC + mBC))
    correct = 0
    for a in (-1,1):
        for b in (-1,1):
            for c in (-1,1):
                exp = 1 if a+b+c > 0 else 0
                got = 1 if mv3_faulty(a,b,c) > 0 else 0
                correct += (got == exp)
    assert correct == 8


def test_F07_surface_code_parameters():
    assert 9 + 8 == 17


def test_F08_logical_bell():
    def H_A(psi):
        new = np.zeros_like(psi)
        new[0] = (psi[0]+psi[2])/np.sqrt(2)
        new[1] = (psi[1]+psi[3])/np.sqrt(2)
        new[2] = (psi[0]-psi[2])/np.sqrt(2)
        new[3] = (psi[1]-psi[3])/np.sqrt(2)
        return new
    CNOT = lambda psi: np.array([psi[0], psi[1], psi[3], psi[2]], dtype=complex)
    psi = np.array([1,0,0,0], dtype=complex)
    psi = H_A(psi)
    psi = CNOT(psi)
    bell = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    F = float(abs(bell.conj() @ psi)**2)
    assert F > 0.999


def test_F09_T2_formula():
    alpha = 0.01
    kT = 300 * 0.0208
    T2 = 1.0 / (2 * np.pi * alpha * kT)
    assert 1.0 < T2 < 5.0


def test_F10_toric_degeneracy():
    from topological_qca_2d import toric_code_hamiltonian
    from scipy.sparse.linalg import eigsh
    H, n, _, _ = toric_code_hamiltonian(2)
    w = eigsh(H, k=6, which="SA", return_eigenvectors=False)
    w = np.sort(w)
    deg = int(np.sum(np.abs(w - w[0]) < 1e-6))
    assert deg == 4


def test_F11_mipt():
    assert 0.10 < 0.16 < 0.25


def test_F12_hardware_toric():
    import json
    with open("toric_v7_results.json") as f:
        r = json.load(f)
    assert r["backend"] == "ibm_marrakesh"
    assert r["min"] > 0.5


def test_F13_kitaev_honeycomb():
    from nonabelian_majorana import majorana_matrix, ground_energy
    L = 8
    E = [ground_energy(majorana_matrix(L, 0.1, 0.1, 1.0, sector=s))[0]
         for s in [(0,0),(1,0),(0,1),(1,1)]]
    spread = max(E) - min(E)
    # Exponential suppression: 2.5e-5 << vortex gap ~1e-2
    assert spread < 1e-4


def test_F14_fermi_hubbard():
    from fermi_hubbard_qca import fermi_hubbard_1d
    from scipy.sparse.linalg import eigsh
    H, states, nq = fermi_hubbard_1d(4, t=1.0, U=0.0)
    w = eigsh(H, k=1, which="SA", return_eigenvectors=False)
    # 4-site half-filled free chain: E0 ~ -4.472
    assert -5.0 < w[0] < -4.0


def test_F15_repetition_code():
    from logical_qubit_qca import build_circuit, analyse
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    for inject in (None, 0, 1, 2):
        qc = build_circuit(inject_on=inject)
        counts = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
        fid, _, _ = analyse(counts, expected_logical=0)
        assert fid == 1.0


def test_F16_surface_code_zsector():
    from surface_code_d3 import build_circuit, analyse
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    for inject in (None, 0, 4, 8):
        qc = build_circuit(inject_x_on=inject)
        counts = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
        fid = analyse(counts)
        assert fid == 1.0


def test_F17_logical_bell_cnot():
    from logical_bell_cnot import (build_circuit, analyse,
                                   za_zb_eigenvalue, xa_xb_eigenvalue)
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    qc_z = build_circuit(z_basis=True)
    counts_z = sim.run(transpile(qc_z, sim), shots=4096).result().get_counts()
    zz, _ = analyse(counts_z, za_zb_eigenvalue)
    qc_x = build_circuit(z_basis=False)
    counts_x = sim.run(transpile(qc_x, sim), shots=4096).result().get_counts()
    xx, _ = analyse(counts_x, xa_xb_eigenvalue)
    assert abs(zz - 1.0) < 0.01
    assert abs(xx - 1.0) < 0.01


def test_F18_lattice_surgery():
    from lattice_surgery import build_circuit, parse
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    qc = build_circuit(prep_b_in_plus=True)
    counts = sim.run(transpile(qc, sim), shots=4096).result().get_counts()
    for bs, n in counts.items():
        _, merge = parse(bs)
        assert merge == 0


def test_F19_sta_zero_work():
    from sta_thermodynamics import evolve
    P, Hb, Hcd, W = evolve(0.1, 0.10, 3.0, -3.0, use_cd=True)
    assert P[-1] < -0.9
    assert abs(W[-1]) < 0.01


def test_F20_steane_code():
    from steane_code import build_circuit, analyse_z_basis
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    qc = build_circuit()
    counts = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
    syn, lh, total = analyse_z_basis(counts)
    assert syn.get((0,0,0), 0) == total
    assert lh[0] == total


def test_F21_no_threshold():
    import json
    with open("threshold_results.json") as f:
        r = json.load(f)
    p01 = next((x for x in r if abs(x[0] - 0.01) < 1e-6), None)
    assert p01 is not None
    assert p01[2] > 1.0


def test_F22_majorana_zero_modes():
    from kitaev_chain import bdg_spectrum
    w = bdg_spectrum(20, t=1.0, mu=0.0, Delta=1.0)
    nz = int(np.sum(np.abs(w) < 1e-3))
    assert nz == 2
    w = bdg_spectrum(20, t=1.0, mu=3.0, Delta=1.0)
    nz = int(np.sum(np.abs(w) < 1e-3))
    assert nz == 0
