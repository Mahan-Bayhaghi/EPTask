from __future__ import annotations
from typing import Dict, Any
import random
import numpy as np
import yaml

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
