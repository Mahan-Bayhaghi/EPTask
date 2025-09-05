from __future__ import annotations
import argparse, csv, os, numpy as np
from rl.utils import load_config
from env.vec_env import EPTaskEnv
from env.metrics import p95, jain_fairness
from env.models import tx_time_seconds, compute_time_seconds, distance_to_snr, snr_to_rate_mbps


def estimate_finish_time(env: EPTaskEnv, task_idx: int, dest_kind: str, dest_id: int):
    t = env.tasks[task_idx]
    holder = t.holder
    holder_x = env.vehicles[holder].x
    # link estimate
    if dest_kind == "local":
        tx_t = 0.0;
        mips = env.cfg.vehicle_compute_mips
    else:
        if dest_kind == "vehicle":
            dst_x = env.vehicles[dest_id].x;
            bw = env.cfg.v2v_bandwidth_mhz
        elif dest_kind == "edge":
            dst_x = (dest_id + 1) * 1000.0 / (env.cfg.num_edges + 1);
            bw = env.cfg.v2i_bandwidth_mhz
        else:
            dst_x = 1000.0;
            bw = env.cfg.v2i_bandwidth_mhz
        d = abs(dst_x - holder_x) + 1.0
        p_dbm = 1.0
        snr = distance_to_snr(d, p_dbm, env.cfg.noise_dbm)
        rate = snr_to_rate_mbps(bw, snr)
        tx_t = tx_time_seconds(t.size_bits, rate)
        mips = (env.cfg.vehicle_compute_mips if dest_kind == "vehicle" else
                env.cfg.edge_compute_mips if dest_kind == "edge" else env.cfg.cloud_compute_mips)
    comp_t = compute_time_seconds(t.cycles, mips)
    # crude queue wait proxy
    if dest_kind == "local":
        qlen = len(env.vehicles[holder].queue)
    elif dest_kind == "vehicle":
        qlen = len(env.vehicles[dest_id].queue)
    elif dest_kind == "edge":
        qlen = len(env.edge_queues[dest_id])
    else:
        qlen = len(env.cloud_queue)
    wait = qlen * comp_t
    return tx_t + wait + comp_t


def greedy_action(env: EPTaskEnv, obs):
    V, E = env.cfg.max_vehicles, env.cfg.num_edges

    def decode(off):
        if off == 0: return ("local", 0)
        if 1 <= off <= V: return ("vehicle", off - 1) if env.cfg.allow_v2v else ("local", 0)
        if V < off <= V + E: return ("edge", off - 1 - V)
        return ("cloud", 0)

    stride = 2 if env.cfg.power_rule == "distance" else 3
    action = []
    topk_idxs = env._select_topk_tasks_edf(env.cfg.top_k)
    for idx in topk_idxs:
        best_off = 0;
        best_eta = float("inf")
        for off in range(V + E + 2):
            kind, did = decode(off)
            eta = estimate_finish_time(env, idx, kind, did)
            if eta < best_eta:
                best_eta = eta;
                best_off = off
        prio_raw = env.cfg.priority_levels - 1
        if stride == 2:
            action.extend([best_off, prio_raw])
        else:
            pbin = max(0, min(env.cfg.power_bins - 1, env.cfg.power_bins // 2))
            action.extend([best_off, prio_raw, pbin])
    while len(action) < stride * env.cfg.top_k:
        action.extend([0] * stride)
    return np.array(action, dtype=np.int64)


def action_local(env: EPTaskEnv):
    return np.array([0, env.cfg.priority_levels - 1] * env.cfg.top_k, dtype=np.int64)


def action_offload(env: EPTaskEnv):
    V, E = env.cfg.max_vehicles, env.cfg.num_edges

    def best_nonlocal(idx):
        best_off = V + E + 1;
        best_eta = float("inf")
        for off in range(1, V + E + 2):
            kind, did = (("vehicle", off - 1) if 1 <= off <= V else
                         ("edge", off - 1 - V) if off <= V + E else ("cloud", 0))
            eta = estimate_finish_time(env, idx, kind, did)
            if eta < best_eta:
                best_eta, best_off = eta, off
        return best_off

    parts = []
    for idx in env._select_topk_tasks_edf(env.cfg.top_k):
        parts.extend([best_nonlocal(idx), env.cfg.priority_levels - 1])
    while len(parts) < 2 * env.cfg.top_k:
        parts.extend([0, env.cfg.priority_levels - 1])
    return np.array(parts, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--mode", choices=["heuristic", "local", "offload", "random"], default="random")
    p.add_argument("--out", default="../runs/random_eval.csv")  # per-episode
    p.add_argument("--metrics_out", default="../runs/random_metrics.csv")  # aggregate metrics
    args = p.parse_args()

    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # per-episode log
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["episode", "return", "steps", "completed_total", "deadline_miss_total"])

        # metric accumulators
        all_completion_times = []
        all_energies = []
        misses = 0
        per_vehicle_completed = np.zeros(env.cfg.max_vehicles, dtype=np.int64)

        for ep in range(args.episodes):
            obs, info = env.reset()
            done, ret, steps, comp_tot, miss_tot = False, 0.0, 0, 0, 0
            while not done:
                if args.mode == "heuristic":
                    action = greedy_action(env, obs)
                elif args.mode == "local":
                    action = action_local(env)
                elif args.mode == "offload":
                    action = action_offload(env)
                else:
                    action = env.action_space.sample()

                obs, r, term, trunc, inf = env.step(action)
                ret += float(r);
                steps += 1
                comp_tot += int(inf.get("completed", 0))
                miss_tot += int(inf.get("deadline_misses", 0))
                misses += int(inf.get("deadline_misses", 0))
                done = term or trunc

                # collect completion info
                for idx in list(env.completed_indices):
                    task = env.tasks[idx]
                    flag = "_metrics_collected_" + str(idx)
                    if getattr(env, flag, False): continue
                    if task.completed_step is not None:
                        all_completion_times.append(task.completed_step - task.created_step)
                        all_energies.append(task.energy_j)
                        holder = int(task.holder) if 0 <= int(task.holder) < env.cfg.max_vehicles else 0
                        per_vehicle_completed[holder] += 1
                        setattr(env, flag, True)

            w.writerow([ep, f"{ret:.4f}", steps, comp_tot, miss_tot])
            print(f"[{args.mode}] ep={ep} return={ret:.3f} steps={steps} completed={comp_tot} misses={miss_tot}")

    # write aggregate metrics compatible with plot_metrics.py
    with open(args.metrics_out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "value"])
        wr.writerow(["completion_time_mean", np.mean(all_completion_times) if all_completion_times else 0.0])
        wr.writerow(["completion_time_p95", p95(all_completion_times)])
        wr.writerow(["energy_mean", np.mean(all_energies) if all_energies else 0.0])
        tot = misses + len(all_completion_times)
        wr.writerow(["miss_rate", misses / max(1, tot)])
        wr.writerow(["fairness", jain_fairness(per_vehicle_completed)])
        wr.writerow(["throughput_tasks", len(all_completion_times)])

    env.close()
    print("Wrote", args.out, "and", args.metrics_out)


if __name__ == "__main__":
    main()
