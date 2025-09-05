from __future__ import annotations
import numpy as np
import gymnasium as gym

def make_observation_space(max_vehicles: int, max_tasks: int, num_edges: int,
                           task_feat_dim: int = 8, target_feat_dim: int = 6, vehicle_feat_dim: int = 4):
    return gym.spaces.Dict(
        {
            "global":  gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
            "tasks":   gym.spaces.Box(-np.inf, np.inf, shape=(max_tasks, task_feat_dim), dtype=np.float32),
            "targets": gym.spaces.Box(-np.inf, np.inf, shape=(max_vehicles + num_edges + 2, target_feat_dim), dtype=np.float32),
            "vehicles":gym.spaces.Box(-np.inf, np.inf, shape=(max_vehicles, vehicle_feat_dim), dtype=np.float32),
            "masks":   gym.spaces.Box(0.0, 1.0, shape=(max_tasks, 3), dtype=np.float32),  # [offload, priority, power] per task
        }
    )

def make_action_space(max_vehicles: int, num_edges: int, priority_levels: int, power_bins: int, top_k: int):
    # offload choices: local (1) + vehicles (max_vehicles) + edges (num_edges) + cloud (1)
    offload_choices = max_vehicles + num_edges + 2
    nvec = []
    for _ in range(top_k):
        nvec.extend([offload_choices, priority_levels, power_bins])
    return gym.spaces.MultiDiscrete(nvec)
