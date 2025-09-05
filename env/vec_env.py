from __future__ import annotations
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .spaces import make_observation_space, make_action_space
from .models import distance_to_snr, snr_to_rate_mbps, tx_time_seconds, compute_time_seconds, tx_energy_joules, compute_energy_joules
from .generators import sample_tasks, init_vehicles, Task

# -------- running normalization (Welford) --------
class _RunningNorm:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / max(1, self.n - 1))) if self.n > 1 else 1.0
    def z(self, x: float, eps: float = 1e-6) -> float:
        return (x - self.mean) / (self.std + eps)

@dataclass
class EPTaskEnvConfig:
    horizon_steps: int = 600
    max_vehicles: int = 50
    num_edges: int = 8
    include_cloud: bool = True
    top_k: int = 4
    priority_levels: int = 4
    power_bins: int = 5
    power_rule: str = "distance"   # "distance" | "bins"
    allow_v2v: bool = True
    max_tasks_per_vehicle: int = 10
    task_queue_capacity: int = 128

    # Radio params (Table III-ish)
    v2v_bandwidth_mhz: float = 20.0
    v2i_bandwidth_mhz: float = 100.0
    noise_dbm: float = -95.0
    tx_power_dbm_bins: Tuple[float, ...] = (1.0,)   # used only if power_rule=="bins"
    # Optional gains / shadowing (tune as needed)
    tx_gain_db: float = 0.0
    rx_gain_db: float = 0.0
    shadow_atten_db: float = 0.0

    # Compute
    vehicle_compute_mips: float = 1000.0
    edge_compute_mips: float = 2000.0
    cloud_compute_mips: float = 5000.0

    # Energy coeffs
    energy_coeff_tx: float = 1.0e-9
    energy_coeff_compute: float = 2.0e-11

    link_fair_share: bool = True

    # Arrivals
    arrival_model: str = "poisson"   # "bernoulli" | "poisson"
    task_arrival_rate: float = 0.8   # used when bernoulli
    task_arrival_lambda: float = 1.3 # used when poisson
    task_burst_max: int = 5

    # Task distributions
    task_size_bits: Tuple[float, float] = (2.0e6, 20.0e6)
    task_cycles: Tuple[float, float] = (2.0e9, 20.0e9)
    deadline_range: Tuple[int, int] = (20, 80)
    emax_range: Tuple[float, float] = (0.05, 0.3)

    # Weights (for future selection scoring)
    w_distance: float = 0.5
    w_workload: float = 0.5

    # Reward weights
    lambda_time: float = 1.0
    lambda_energy: float = 1.0
    lambda_deadline: float = 5.0

class EPTaskEnv(gym.Env):
    """
    EPTask-style scheduling env.
    - Non-preemptive + EDF at executors
    - Conditional action space: (offload, priority) [+ power if bins]
    - Distance-based power default
    - Poisson arrivals (configurable)
    - Running reward normalization
    """
    metadata = {}

    def __init__(self, cfg: Dict[str, Any], seed: int = 42):
        super().__init__()
        self.cfg = EPTaskEnvConfig(**cfg["env"], **cfg.get("reward", {}))
        self._coerce_config_types()
        self.rng = np.random.default_rng(cfg.get("seed", seed))
        self.max_tasks = self.cfg.task_queue_capacity

        # whether the action includes power
        self._include_power = (self.cfg.power_rule.lower() == "bins")

        self.observation_space = make_observation_space(self.cfg.max_vehicles, self.max_tasks, self.cfg.num_edges)
        self.action_space = make_action_space(
            self.cfg.max_vehicles, self.cfg.num_edges, self.cfg.priority_levels,
            self.cfg.power_bins, self.cfg.top_k, include_power=self._include_power
        )

        # reward running norms
        self._rn_time = _RunningNorm()
        self._rn_energy = _RunningNorm()

        self.reset(seed=seed)

    # ---- type coercion (handles quoted YAML numbers) ----
    def _coerce_config_types(self):
        def _float(x): return float(x)
        def _int(x): return int(float(x))
        def _pair_float(x):
            if isinstance(x, (list, tuple)) and len(x) == 2: return float(x[0]), float(x[1])
            s = str(x).strip("[]()")
            a, b = [p.strip() for p in s.split(",")]
            return float(a), float(b)
        def _pair_int(x):
            a, b = _pair_float(x); return int(round(a)), int(round(b))

        self.cfg.v2v_bandwidth_mhz = _float(self.cfg.v2v_bandwidth_mhz)
        self.cfg.v2i_bandwidth_mhz = _float(self.cfg.v2i_bandwidth_mhz)
        self.cfg.noise_dbm = _float(self.cfg.noise_dbm)
        self.cfg.tx_power_dbm_bins = tuple(_float(v) for v in self.cfg.tx_power_dbm_bins)
        self.cfg.tx_gain_db = _float(getattr(self.cfg, "tx_gain_db", 0.0))
        self.cfg.rx_gain_db = _float(getattr(self.cfg, "rx_gain_db", 0.0))
        self.cfg.shadow_atten_db = _float(getattr(self.cfg, "shadow_atten_db", 0.0))

        self.cfg.vehicle_compute_mips = _float(self.cfg.vehicle_compute_mips)
        self.cfg.edge_compute_mips = _float(self.cfg.edge_compute_mips)
        self.cfg.cloud_compute_mips = _float(self.cfg.cloud_compute_mips)
        self.cfg.energy_coeff_tx = _float(self.cfg.energy_coeff_tx)
        self.cfg.energy_coeff_compute = _float(self.cfg.energy_coeff_compute)

        # arrivals
        self.cfg.arrival_model = str(getattr(self.cfg, "arrival_model", "poisson")).lower()
        self.cfg.task_arrival_rate = _float(getattr(self.cfg, "task_arrival_rate", 0.8))
        self.cfg.task_arrival_lambda = _float(getattr(self.cfg, "task_arrival_lambda", 1.3))
        self.cfg.task_burst_max = _int(getattr(self.cfg, "task_burst_max", 5))

        # tasks
        self.cfg.task_size_bits = _pair_float(self.cfg.task_size_bits)
        self.cfg.task_cycles = _pair_float(self.cfg.task_cycles)
        self.cfg.deadline_range = _pair_int(self.cfg.deadline_range)
        self.cfg.emax_range = _pair_float(self.cfg.emax_range)

        # misc
        self.cfg.top_k = _int(self.cfg.top_k)
        self.cfg.priority_levels = _int(self.cfg.priority_levels)
        self.cfg.power_bins = _int(self.cfg.power_bins)
        self.cfg.max_tasks_per_vehicle = _int(self.cfg.max_tasks_per_vehicle)
        self.cfg.task_queue_capacity = _int(self.cfg.task_queue_capacity)
        self.cfg.num_edges = _int(self.cfg.num_edges)
        self.cfg.max_vehicles = _int(self.cfg.max_vehicles)
        self.cfg.allow_v2v = bool(getattr(self.cfg, "allow_v2v", True))

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
        # reset running norms each episode (optionally keep across episodes if you prefer)
        self._rn_time = _RunningNorm()
        self._rn_energy = _RunningNorm()
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        topk_idxs = self._select_topk_tasks_edf(self.cfg.top_k)

        # parse (off, prio[, pbin]) per scheduled task
        stride = 3 if self._include_power else 2
        for i, task_idx in enumerate(topk_idxs):
            base = stride * i
            off = int(action[base + 0])
            prio_raw = int(action[base + 1])
            prio = int(np.clip(prio_raw + 1, 1, self.cfg.priority_levels))
            self.tasks[task_idx].priority = prio

            dest_kind, dest_id = self._decode_offload(off)
            self._enqueue_task(task_idx, dest_kind, dest_id)

            if self._include_power:
                pbin = int(action[base + 2])
                p_dbm = float(self.cfg.tx_power_dbm_bins[int(np.clip(pbin, 0, len(self.cfg.tx_power_dbm_bins)-1))])
            else:
                p_dbm = self._distance_based_power(task_idx, dest_kind, dest_id)
            setattr(self.tasks[task_idx], "_chosen_p_dbm", p_dbm)

        # simulate one step
        self._advance_time()

        # reward from tasks completed at this step (normalized)
        reward = 0.0
        completed_now = [i for i in self.completed_indices if getattr(self, "_completed_flag_"+str(i), False)]
        for i in completed_now:
            task = self.tasks[i]
            completion_time_steps = (task.completed_step - task.created_step)
            # update running stats
            self._rn_time.update(float(completion_time_steps))
            self._rn_energy.update(float(task.energy_j))
            # z-scores
            zt = self._rn_time.z(float(completion_time_steps))
            ze = self._rn_energy.z(float(task.energy_j))
            reward -= (self.cfg.lambda_time * zt + self.cfg.lambda_energy * ze)
            setattr(self, "_completed_flag_"+str(i), False)

        # deadline penalties
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
        if idx == 0:
            return ("local", 0)
        if 1 <= idx <= V:
            return ("vehicle", idx - 1) if self.cfg.allow_v2v else ("local", 0)
        if V < idx <= V + E:
            return ("edge", idx - 1 - V)
        return ("cloud", 0)

    def _remove_from_all_queues(self, task_idx: int):
        for v in self.vehicles:
            if task_idx in v.queue:
                v.queue.remove(task_idx)
        for e in range(self.cfg.num_edges):
            if task_idx in self.edge_queues[e]:
                self.edge_queues[e].remove(task_idx)
        if task_idx in self.cloud_queue:
            self.cloud_queue.remove(task_idx)

    def _enqueue_task(self, task_idx: int, kind: str, dest_id: int):
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
        """Choose TX power based on distance thresholds (coarse)"""
        holder = self.tasks[task_idx].holder
        src_x = self.vehicles[holder].x
        if kind == "local":
            return 0.0  # no TX
        if kind == "vehicle":
            d = abs(self.vehicles[dest_id].x - src_x)
        elif kind == "edge":
            anchor = (dest_id + 1) * 1000.0 / (self.cfg.num_edges + 1)
            d = abs(anchor - src_x)
        else:
            d = 1000.0  # far
        # simple tiers using bins if provided, else fixed 1 dBm
        bins = list(self.cfg.tx_power_dbm_bins) if len(self.cfg.tx_power_dbm_bins) > 0 else [1.0]
        if d < 100: return float(bins[0])
        if d < 300: return float(bins[min(1, len(bins)-1)])
        if d < 700: return float(bins[min(2, len(bins)-1)])
        return float(bins[-1])

    def _advance_time(self):
        # arrivals
        for v in self.vehicles:
            if self.cfg.arrival_model == "poisson":
                k = int(self.rng.poisson(self.cfg.task_arrival_lambda))
                k = min(k, int(self.cfg.task_burst_max))
            else:
                p = float(self.cfg.task_arrival_rate)
                k = 1 if self.rng.random() < min(1.0, p) else 0
            for _ in range(k):
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

        # move vehicles
        for v in self.vehicles:
            v.x += v.v * 1.0

        # serve one task per executor per step (EDF)
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

    def _link_rate_and_energy(self, t: Task, exec_kind: str, exec_id: int, holder_x: float) -> Tuple[float, float]:
        """Return (rate_mbps, tx_energy_j) for single-hop TX with simple link budget."""
        if exec_kind == "vehicle":
            dst_x = self.vehicles[exec_id].x
            bw = self.cfg.v2v_bandwidth_mhz
        else:
            dst_x = (exec_id + 1) * 1000.0 / (self.cfg.num_edges + 1) if exec_kind == "edge" else 1000.0
            bw = self.cfg.v2i_bandwidth_mhz

        d = abs(dst_x - holder_x) + 1.0
        p_dbm = getattr(t, "_chosen_p_dbm", float(self.cfg.tx_power_dbm_bins[0] if len(self.cfg.tx_power_dbm_bins)>0 else 1.0))
        # Link budget: add TX/RX gains, subtract shadowing
        p_eff_dbm = p_dbm + self.cfg.tx_gain_db + self.cfg.rx_gain_db - self.cfg.shadow_atten_db
        snr = distance_to_snr(d, p_eff_dbm, self.cfg.noise_dbm)
        rate_mbps = snr_to_rate_mbps(bw, snr)
        tx_t = tx_time_seconds(t.size_bits, rate_mbps)
        tx_e = tx_energy_joules(t.size_bits, p_dbm, tx_t, self.cfg.energy_coeff_tx)
        return rate_mbps, tx_e

    def _process_task_on_executor(self, t_idx: int, exec_kind: str, exec_id: int):
        t = self.tasks[t_idx]
        if t.done: return
        holder = t.holder
        holder_x = self.vehicles[holder].x

        # Link & TX (if not local)
        if exec_kind == "vehicle":
            rate_mbps, tx_e = self._link_rate_and_energy(t, exec_kind, exec_id, holder_x)
            mips = self.cfg.vehicle_compute_mips
        elif exec_kind == "edge":
            rate_mbps, tx_e = self._link_rate_and_energy(t, exec_kind, exec_id, holder_x)
            mips = self.cfg.edge_compute_mips
        elif exec_kind == "cloud":
            rate_mbps, tx_e = self._link_rate_and_energy(t, exec_kind, exec_id, holder_x)
            mips = self.cfg.cloud_compute_mips
        else:  # local
            rate_mbps, tx_e = (1e9, 0.0)  # no TX
            mips = self.cfg.vehicle_compute_mips

        comp_t = compute_time_seconds(t.cycles, mips)
        comp_e = compute_energy_joules(t.cycles, comp_t, self.cfg.energy_coeff_compute)

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
        masks = np.ones((maxT, 3), dtype=np.float32)
        for i, t in enumerate(self.tasks[:maxT]):
            age = self.t - t.created_step
            remain = t.deadline - age
            tasks_mat[i] = np.array([
                t.size_bits/1e6, t.cycles/1e6, float(age), float(remain),
                float(t.emax), float(t.priority), float(t.holder), 1.0 if not t.done else 0.0
            ], dtype=np.float32)

        targets = np.zeros((maxV + E + 2, 6), dtype=np.float32)
        targets[0] = np.array([1.0, 0.0, 0.0, 0.0, float(len(self.vehicles[0].queue) if self.vehicles else 0), 1.0], dtype=np.float32)
        for vi in range(maxV):
            qlen = len(self.vehicles[vi].queue) if vi < len(self.vehicles) else 0
            targets[1 + vi] = np.array([0.0, 1.0, 0.0, 0.0, float(qlen), 1.0], dtype=np.float32)
        for ei in range(E):
            qlen = len(self.edge_queues[ei]) if ei < len(self.edge_queues) else 0
            targets[1 + maxV + ei] = np.array([0.0, 0.0, 1.0, 0.0, float(qlen), 1.0], dtype=np.float32)
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
