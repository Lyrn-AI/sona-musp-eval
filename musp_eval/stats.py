"""Interval and paired-test helpers used in the report."""
from __future__ import annotations

import math
from math import comb


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k correct out of n, in percent."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test on the questions exactly one of two systems answered correctly."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
