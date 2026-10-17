"""Hardware regression tests for the ibm_fez validation run (v1.7.0).

These guard against future hardware runs producing worse results.
Skip if the JSON result files are absent (e.g., fresh clone).
"""
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _load(name):
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"{name} not present (run hardware first)")
    with open(p) as f:
        return json.load(f)


def test_hw_F12_toric_fez():
    """Toric-code stabilizers on ibm_fez: min > 0.65."""
    r = _load("toric_v7_results.json")
    assert r["backend"] == "ibm_fez"
    assert r["min"] > 0.65, f"min stab = {r['min']}"
    assert min(r["B_p"]) > 0.75, f"min B_p = {min(r['B_p'])}"
    assert min(r["A_v"]) > 0.65, f"min A_v = {min(r['A_v'])}"


def test_hw_F15_repetition_code():
    """Repetition-code scenarios: every scenario > 0.90."""
    r = _load("logical_qubit_results.json")
    assert r["backend"] == "ibm_fez"
    for label, data in r["results"].items():
        assert data["fidelity"] > 0.90, f"{label}: {data['fidelity']}"


def test_hw_F16_surface_code_zsector():
    """Surface-code Z-sector: every scenario > 0.70."""
    r = _load("surface_code_d3_results.json")
    assert r["backend"] == "ibm_fez"
    for label, fid in r["results"].items():
        assert fid > 0.70, f"{label}: {fid}"


def test_hw_F17_logical_bell():
    """Transversal logical CNOT: <Z> > 0.85, <X> > 0.75."""
    r = _load("logical_bell_results.json")
    assert r["zz"] > 0.85, f"<Z_A Z_B> = {r['zz']}"
    assert r["xx"] > 0.75, f"<X_A X_B> = {r['xx']}"
