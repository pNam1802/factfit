"""Small statistics helpers for reporting eval results honestly on small samples."""

from math import sqrt


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes out of n.

    Unlike k/n ± ..., it stays inside [0, 1] and is sensible at 0/n and n/n: catching
    34 of 34 cases supports "at least ~90%", not "100%".
    """
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
