from __future__ import annotations
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .spaces import make_observation_space, make_action_space
from .models import distance_to_snr, snr_to_rate_mbps, tx_time_seconds, compute_time_seconds, tx_energy_joules, compute_energy_joules
from .generators import sample_tasks, init_vehicles, Task

@dataclass
class EPTaskEnvConfig:
    horizon_steps: int = 600
    max_vehicles: int = 50
    num_edges: int = 8
    include_cloud: bool = True
    top_k: int = 4
    priority_levels: int = 4
    power_bins: int = 5
    power_rule: str = "bins"   # "bins" | "distance"
    max_tasks_per_vehicle: int = 10
    task_queue_capacity: int = 128

    v2v_bandwidth_mhz: float = 10.0
    v2i_bandwidth_mhz: float = 20.0
    noise_dbm: float = -90.0
    tx_power_dbm_bins: Tuple[float, ...] = (-5, 0, 5, 10, 15)
    vehicle_compute_mips: float = 50.0
    edge_compute_mips: float = 500.0
    cloud_compute_mips: float = 2000.0
    energy_coeff_tx: float = 1.0e-6
    energy_coeff_compute: float = 1.0e-9
    link_fair_share: bool = True

    task_arrival_rate: float = 0.8
    task_size_bits: Tuple[float, float] = (1.0e6, 4.0e6)
    task_cycles: Tuple[float, float] = (5.0e7, 2.0e8)
    deadline_range: Tuple[int, int] = (50, 200)
    emax_range: Tuple[float, float] = (0.05, 0.3)

    w_distance: float = 0.5
    w_workload: float = 0.5

    lambda_time: float = 1.0
    lambda_energy: float = 1.0
    lambda_deadline: float = 5.0

class EPTaskEnv(gym.Env):
    """
    Gymnasium environment for EPTask-style scheduling.
    Simplified dynamics, non-preemptive + EDF, factorized actions for top-K tasks.
    """
    metadata = {}

    def __init__(self, cfg: Dict[str, Any], seed: int = 42):
        super().__init__()
        self.cfg = EPTaskEnvConfig(**cfg["env"], **cfg.get("reward", {}))
        self._coerce_config_types()  # <-- NEW: make sure everything is numeric
        self.rng = np.random.default_rng(cfg.get("seed", seed))
        self.max_tasks = self.cfg.task_queue_capacity

        self.observation_space = make_observation_space(self.cfg.max_vehicles, self.max_tasks, self.cfg.num_edges)
        self.action_space = make_action_space(self.cfg.max_vehicles, self.cfg.num_edges,
                                              self.cfg.priority_levels, self.cfg.power_bins, self.cfg.top_k)
        self.reset(seed=seed)

    # --- NEW: normalize all numeric config fields (handles quoted YAML numbers) ---
    def _coerce_config_types(self):
        def _float(x): return float(x)

        def _int(x): return int(float(x))

        def _pair_float(x):
            a, b = (x if isinstance(x, (list, tuple)) else str(x).strip("[]()").split(","))
            return float(a), float(b)

        def _pair_int(x):
            a, b = _pair_float(x);
            return int(round(a)), int(round(b))

        self.cfg.v2v_bandwidth_mhz = _float(self.cfg.v2v_bandwidth_mhz)
        self.cfg.v2i_bandwidth_mhz = _float(self.cfg.v2i_bandwidth_mhz)
        self.cfg.noise_dbm = _float(self.cfg.noise_dbm)
        self.cfg.tx_power_dbm_bins = tuple(_float(v) for v in self.cfg.tx_power_dbm_bins)

        self.cfg.vehicle_compute_mips = _float(self.cfg.vehicle_compute_mips)
        self.cfg.edge_compute_mips = _float(self.cfg.edge_compute_mips)
        self.cfg.cloud_compute_mips = _float(self.cfg.cloud_compute_mips)
        self.cfg.energy_coeff_tx = _float(self.cfg.energy_coeff_tx)
        self.cfg.energy_coeff_compute = _float(self.cfg.energy_coeff_compute)

        self.cfg.task_arrival_rate = _float(self.cfg.task_arrival_rate)
        self.cfg.task_size_bits = _pair_float(self.cfg.task_size_bits)
        self.cfg.task_cycles = _pair_float(self.cfg.task_cycles)
        self.cfg.deadline_range = _pair_int(self.cfg.deadline_range)
        self.cfg.emax_range = _pair_float(self.cfg.emax_range)

        self.cfg.top_k = _int(self.cfg.top_k)
        self.cfg.priority_levels = _int(self.cfg.priority_levels)
        self.cfg.power_bins = _int(self.cfg.power_bins)
        self.cfg.max_tasks_per_vehicle = _int(self.cfg.max_tasks_per_vehicle)
        self.cfg.task_queue_capacity = _int(self.cfg.task_queue_capacity)
        self.cfg.num_edges = _int(self.cfg.num_edges)
        self.cfg.max_vehicles = _int(self.cfg.max_vehicles)

    # --------------- Gym API ---------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.vehicles = init_vehicles(self.rng, self.cfg.max_vehicles)
        self.edge_queues: List[list] = [[] for _ in range(self.cfg.num_edges)]
        self.cloud_queue: list = []
        self.tasks: List[Task] = []
        self.completed_indices: list = []
        self.deadline_misses: int = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        topk_idxs = self._select_topk_tasks_edf(self.cfg.top_k)

        # decode and apply decisions
        for i, task_idx in enumerate(topk_idxs):
            off = int(action[3*i + 0])
            prio_raw = int(action[3*i + 1])
            pbin = int(action[3*i + 2])

            # priority in [1..L]
            prio = int(np.clip(prio_raw + 1, 1, self.cfg.priority_levels))
            self.tasks[task_idx].priority = prio

            dest_kind, dest_id = self._decode_offload(off)
            self._enqueue_task(task_idx, dest_kind, dest_id)

            # choose power
            if self.cfg.power_rule == "bins":
                p_dbm = float(self.cfg.tx_power_dbm_bins[int(np.clip(pbin, 0, len(self.cfg.tx_power_dbm_bins)-1))])
            else:
                p_dbm = self._distance_based_power(task_idx, dest_kind, dest_id)
            setattr(self.tasks[task_idx], "_chosen_p_dbm", p_dbm)

        # simulate one step
        self._advance_time()

        # reward from tasks completed at this step
        reward = 0.0
        completed_now = [i for i in self.completed_indices if getattr(self, "_completed_flag_"+str(i), False)]
        for i in completed_now:
            task = self.tasks[i]
            completion_time = (task.completed_step - task.created_step)
            time_norm = completion_time / max(self.cfg.horizon_steps, 1)
            energy_norm = task.energy_j  # scale can be tuned later
            reward -= self.cfg.lambda_time * time_norm
            reward -= self.cfg.lambda_energy * energy_norm
            setattr(self, "_completed_flag_"+str(i), False)

        deadline_misses_now = sum(1 for i in range(len(self.tasks))
                                  if getattr(self, "_deadline_miss_flag_"+str(i), False))
        reward -= self.cfg.lambda_deadline * float(deadline_misses_now)
        for i in range(len(self.tasks)):
            if getattr(self, "_deadline_miss_flag_"+str(i), False):
                setattr(self, "_deadline_miss_flag_"+str(i), False)

        self.t += 1
        terminated = (self.t >= self.cfg.horizon_steps)
        truncated = False
        info = {"completed": len(completed_now), "deadline_misses": deadline_misses_now}
        return self._obs(), float(reward), terminated, truncated, info

    # --------------- Internals ---------------
    def _select_topk_tasks_edf(self, k: int):
        idxs = [i for i, t in enumerate(self.tasks) if not t.done]
        edf_sorted = sorted(
            idxs,
            key=lambda i: (self.tasks[i].deadline - (self.t - self.tasks[i].created_step),
                           self.tasks[i].created_step)
        )
        return edf_sorted[:k]

    def _decode_offload(self, off_index: int):
        V, E = self.cfg.max_vehicles, self.cfg.num_edges
        idx = int(np.clip(off_index, 0, V + E + 1))
        if idx == 0: return ("local", 0)
        if 1 <= idx <= V: return ("vehicle", idx - 1)
        if V < idx <= V + E: return ("edge", idx - 1 - V)
        return ("cloud", 0)

    def _remove_from_all_queues(self, task_idx: int):
        # remove if present anywhere
        for v in self.vehicles:
            if task_idx in v.queue:
                v.queue.remove(task_idx)
        for e in range(self.cfg.num_edges):
            if task_idx in self.edge_queues[e]:
                self.edge_queues[e].remove(task_idx)
        if task_idx in self.cloud_queue:
            self.cloud_queue.remove(task_idx)

    def _enqueue_task(self, task_idx: int, kind: str, dest_id: int):
        # move task to a new destination queue (non-preemptive EDF at service time)
        self._remove_from_all_queues(task_idx)
        if kind == "local":
            v = self.tasks[task_idx].holder
            self.vehicles[v].queue.append(task_idx)
        elif kind == "vehicle":
            self.tasks[task_idx].holder = dest_id
            self.vehicles[dest_id].queue.append(task_idx)
        elif kind == "edge":
            self.edge_queues[dest_id].append(task_idx)
        else:
            self.cloud_queue.append(task_idx)

    def _distance_based_power(self, task_idx: int, kind: str, dest_id: int) -> float:
        # choose power by distance thresholds for THIS task’s holder
        holder = self.tasks[task_idx].holder
        src_x = self.vehicles[holder].x
        if kind == "local":
            return float(self.cfg.tx_power_dbm_bins[0])
        if kind == "vehicle":
            d = abs(self.vehicles[dest_id].x - src_x)
        elif kind == "edge":
            anchor = (dest_id + 1) * 1000.0 / (self.cfg.num_edges + 1)
            d = abs(anchor - src_x)
        else:
            d = 1000.0
        bins = self.cfg.tx_power_dbm_bins
        if d < 100: return float(bins[0])
        if d < 300: return float(bins[min(1, len(bins)-1)])
        if d < 700: return float(bins[min(2, len(bins)-1)])
        return float(bins[-1])

    def _advance_time(self):
        # arrivals (Bernoulli approx to Poisson)
        for v in self.vehicles:
            if self.rng.random() < min(1.0, self.cfg.task_arrival_rate):
                if len(v.queue) < self.cfg.task_queue_capacity:
                    new_tasks = sample_tasks(self.rng, 1, {
                        "task_size_bits": self.cfg.task_size_bits,
                        "task_cycles": self.cfg.task_cycles,
                        "deadline_range": self.cfg.deadline_range,
                        "emax_range": self.cfg.emax_range,
                        "priority_levels": self.cfg.priority_levels
                    }, self.t, v.id)
                    base_idx = len(self.tasks)
                    self.tasks.extend(new_tasks)
                    v.queue.append(base_idx)

        # 1D movement
        for v in self.vehicles:
            v.x += v.v * 1.0

        # serve one task per executor per step with EDF
        for vi, veh in enumerate(self.vehicles):
            if veh.queue:
                t_idx = min(veh.queue, key=lambda i: (self.tasks[i].deadline - (self.t - self.tasks[i].created_step)))
                self._process_task_on_executor(t_idx, "vehicle", vi)
        for e_id in range(self.cfg.num_edges):
            if self.edge_queues[e_id]:
                t_idx = min(self.edge_queues[e_id], key=lambda i: (self.tasks[i].deadline - (self.t - self.tasks[i].created_step)))
                self._process_task_on_executor(t_idx, "edge", e_id)
        if self.cloud_queue:
            t_idx = min(self.cloud_queue, key=lambda i: (self.tasks[i].deadline - (self.t - self.tasks[i].created_step)))
            self._process_task_on_executor(t_idx, "cloud", 0)

        # deadlines
        for i, t in enumerate(self.tasks):
            if not t.done and (self.t - t.created_step) > t.deadline:
                setattr(self, "_deadline_miss_flag_"+str(i), True)
                self.deadline_misses += 1
                t.done = True
                t.completed_step = self.t

    def _process_task_on_executor(self, t_idx: int, exec_kind: str, exec_id: int):
        t = self.tasks[t_idx]
        if t.done: return
        holder = t.holder
        src_x = self.vehicles[holder].x
        if exec_kind == "vehicle":
            dst_x = self.vehicles[exec_id].x
            bw = self.cfg.v2v_bandwidth_mhz
            mips = self.cfg.vehicle_compute_mips
        elif exec_kind == "edge":
            dst_x = (exec_id + 1) * 1000.0 / (self.cfg.num_edges + 1)
            bw = self.cfg.v2i_bandwidth_mhz
            mips = self.cfg.edge_compute_mips
        else:
            dst_x = 1000.0
            bw = self.cfg.v2i_bandwidth_mhz
            mips = self.cfg.cloud_compute_mips

        d = abs(dst_x - src_x) + 1.0
        p_dbm = getattr(t, "_chosen_p_dbm", float(self.cfg.tx_power_dbm_bins[0]))
        snr = distance_to_snr(d, p_dbm, self.cfg.noise_dbm)
        rate_mbps = snr_to_rate_mbps(bw, snr)

        tx_t = tx_time_seconds(t.size_bits, rate_mbps)
        tx_e = tx_energy_joules(t.size_bits, p_dbm, tx_t, self.cfg.energy_coeff_tx)

        comp_t = compute_time_seconds(t.cycles, mips)
        comp_e = compute_energy_joules(t.cycles, comp_t, self.cfg.energy_coeff_compute)

        # complete within this step (simplified)
        t.energy_j += (tx_e + comp_e)
        t.done = True
        t.completed_step = self.t
        setattr(self, "_completed_flag_"+str(t_idx), True)

        # remove from queues
        if t_idx in self.vehicles[holder].queue: self.vehicles[holder].queue.remove(t_idx)
        if exec_kind == "vehicle" and t_idx in self.vehicles[exec_id].queue: self.vehicles[exec_id].queue.remove(t_idx)
        if exec_kind == "edge" and t_idx in self.edge_queues[exec_id]: self.edge_queues[exec_id].remove(t_idx)
        if exec_kind == "cloud" and t_idx in self.cloud_queue: self.cloud_queue.remove(t_idx)

        self.completed_indices.append(t_idx)

    # --------------- Obs builder ---------------
    def _obs(self):
        maxV, maxT, E = self.cfg.max_vehicles, self.max_tasks, self.cfg.num_edges

        global_vec = np.zeros(10, dtype=np.float32)
        global_vec[0] = float(self.t) / max(1, self.cfg.horizon_steps)
        global_vec[1] = float(self.cfg.v2v_bandwidth_mhz)
        global_vec[2] = float(self.cfg.v2i_bandwidth_mhz)
        global_vec[3] = float(self.cfg.noise_dbm)
        global_vec[4] = float(self.cfg.vehicle_compute_mips)
        global_vec[5] = float(self.cfg.edge_compute_mips)
        global_vec[6] = float(self.cfg.cloud_compute_mips)
        global_vec[7] = float(self.cfg.priority_levels)
        global_vec[8] = float(self.cfg.power_bins)
        global_vec[9] = float(self.cfg.top_k)

        tasks_mat = np.zeros((maxT, 8), dtype=np.float32)
        masks = np.zeros((maxT, 3), dtype=np.float32)
        for i, t in enumerate(self.tasks[:maxT]):
            age = self.t - t.created_step
            remain = t.deadline - age
            tasks_mat[i] = np.array([
                t.size_bits/1e6, t.cycles/1e6, float(age), float(remain),
                float(t.emax), float(t.priority), float(t.holder), 1.0 if not t.done else 0.0
            ], dtype=np.float32)
            masks[i] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        targets = np.zeros((maxV + E + 2, 6), dtype=np.float32)
        # local (0)
        targets[0] = np.array([1.0, 0.0, 0.0, 0.0, float(len(self.vehicles[0].queue) if self.vehicles else 0), 1.0], dtype=np.float32)
        # vehicles [1..V]
        for vi in range(maxV):
            qlen = len(self.vehicles[vi].queue) if vi < len(self.vehicles) else 0
            targets[1 + vi] = np.array([0.0, 1.0, 0.0, 0.0, float(qlen), 1.0], dtype=np.float32)
        # edges [V+1..V+E]
        for ei in range(E):
            qlen = len(self.edge_queues[ei]) if ei < len(self.edge_queues) else 0
            targets[1 + maxV + ei] = np.array([0.0, 0.0, 1.0, 0.0, float(qlen), 1.0], dtype=np.float32)
        # cloud [V+E+1]
        targets[maxV + E + 1] = np.array([0.0, 0.0, 0.0, 1.0, float(len(self.cloud_queue)), 1.0], dtype=np.float32)

        vehicles_mat = np.zeros((maxV, 4), dtype=np.float32)
        for vi in range(maxV):
            if vi < len(self.vehicles):
                v = self.vehicles[vi]
                vehicles_mat[vi] = np.array([v.x/1000.0, v.v/50.0,
                                             float(len(v.queue))/self.cfg.task_queue_capacity, v.battery], dtype=np.float32)

        return {
            "global": global_vec,
            "tasks": tasks_mat,
            "targets": targets,
            "vehicles": vehicles_mat,
            "masks": masks
        }
