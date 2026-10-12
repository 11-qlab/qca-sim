# Copyright 2026 11-qlab
# SPDX-License-Identifier: Apache-2.0

"""QCA simulator: Hubbard cell + exact diagonalization."""
from .cell import QCACell
from .gates import majority
from .cell_hubbard import (
    basis,
    hamiltonian,
    ground_state,
    polarization,
)

__all__ = [
    "QCACell",
    "majority",
    "basis",
    "hamiltonian",
    "ground_state",
    "polarization",
]
