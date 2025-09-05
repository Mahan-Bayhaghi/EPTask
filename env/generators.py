from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any

# ---------- helpers to coerce YAML values ----------
def _to_float_pair(val: Any) -> Tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    if isinstance(val, str):
        s = val.strip().strip("[]()")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    raise ValueError(f"Cannot parse float pair from: {val!r}")

def _to_int_pair(val: Any) -> Tuple[int, int]:
    f0, f1 = _to_float_pair(val)
    return int(round(f0)), int(round(f1))

def _to_int(val: Any) -> int:
    return int(float(val))

# ---------- data classes ----------
@dataclass
class Task:
    size_bits: float
    cycles: float
    deadline: int
    emax: float
    priority: int
    created_step: int
    holder: int
    done: bool = False
    completed_step: int | None = None
    energy_j: float = 0.0

@dataclass
class Vehicle:
    id: int
    x: float
    v: float
    battery: float
    queue: list  # indices of tasks

# ---------- sampling ----------
def sample_tasks(rng: np.random.Generator, n: int, cfg: dict, step: int, holder: int):
    # Coerce everything to numeric scalars/pairs
    size_lo, size_hi = _to_float_pair(cfg["task_size_bits"])
    cyc_lo, cyc_hi   = _to_float_pair(cfg["task_cycles"])
    ddl_lo, ddl_hi   = _to_int_pair(cfg["deadline_range"])
    emax_lo, emax_hi = _to_float_pair(cfg["emax_range"])
    priority_levels  = _to_int(cfg["priority_levels"])

    sizes = rng.uniform(size_lo, size_hi, size=n)
    cycles = rng.uniform(cyc_lo, cyc_hi, size=n)
    deadlines = rng.integers(ddl_lo, ddl_hi + 1, size=n).tolist()
    emaxs = rng.uniform(emax_lo, emax_hi, size=n)
    priorities = rng.integers(1, priority_levels + 1, size=n)

    tasks = []
    for i in range(n):
        tasks.append(Task(float(sizes[i]), float(cycles[i]), int(deadlines[i]),
                          float(emaxs[i]), int(priorities[i]), step, holder))
    return tasks

def init_vehicles(rng: np.random.Generator, max_vehicles: int):
    vehicles = []
    for i in range(max_vehicles):
        x = float(rng.uniform(0.0, 1000.0))
        v = float(rng.uniform(10.0, 30.0))
        vehicles.append(Vehicle(i, x, v, battery=1.0, queue=[]))
    return vehicles
