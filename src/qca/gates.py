def majority(a: int, b: int, c: int) -> int:
    """Majority gate: 1 if at least two inputs are 1."""
    return 1 if (a + b + c) >= 2 else 0
