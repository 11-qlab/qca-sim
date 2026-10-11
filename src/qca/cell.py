from dataclasses import dataclass


@dataclass
class QCACell:
    """A QCA cell with two electrons and polarization."""

    polarization: float = -1.0  # -1 or +1

    def switch(self, target: float, clock: float = 1.0) -> None:
        """Ideal clocked switching."""
        if clock > 0.5:
            self.polarization = 1.0 if target >= 0 else -1.0
