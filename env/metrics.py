from __future__ import annotations
import numpy as np
from typing import Iterable


def p95(arr: Iterable[float]) -> float:
    a = np.asarray(list(arr), dtype=np.float64)
    return float(np.percentile(a, 95)) if a.size > 0 else 0.0


def jain_fairness(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0: return 0.0
    num = (x.sum()) ** 2
    den = x.size * (x ** 2).sum()
    return float(num / den) if den > 0 else 0.0
