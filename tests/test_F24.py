"""F24: STA clock saturates the Mandelstam-Tamm quantum speed limit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import numpy as np


def test_F24_qsl_saturation():
    from quantum_speed_limit import evolve_with_stats
    ts, P, DE = evolve_with_stats(0.1, 0.10, 3.0, -3.0, use_cd=True)
    # Flip achieved
    assert P[-1] < -0.9, f"P_final = {P[-1]}"
    # Saturates QSL: T * <Delta_E> close to pi/2
    product = 0.1 * np.mean(DE)
    assert 1.4 < product < 1.8, f"T*<Delta_E> = {product}"


def test_F24_qsl_baseline_not_saturated():
    from quantum_speed_limit import evolve_with_stats
    ts, P, DE = evolve_with_stats(0.1, 0.10, 3.0, -3.0, use_cd=False)
    # Baseline fails to flip
    assert P[-1] > 0.9, f"baseline P_final = {P[-1]}"
