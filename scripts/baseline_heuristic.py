from __future__ import annotations
import argparse, csv, os, numpy as np
from rl.utils import load_config
from env.vec_env import EPTaskEnv, tx_time_seconds, compute_time_seconds, distance_to_snr, snr_to_rate_mbps

def estimate_finish_time(env: EPTaskEnv, task_idx: int, dest_kind: str, dest_id: int):
    t = env.tasks[task_idx]
    holder = t.holder
    holder_x = env.vehicles[holder].x
    # link rate estimate
    if dest_kind == "local":
        tx_t = 0.0; mips = env.cfg.vehicle_compute_mips
    else:
        if dest_kind == "vehicle":
            dst_x = env.vehicles[dest_id].x; bw = env.cfg.v2v_bandwidth_mhz
        elif dest_kind == "edge":
            dst_x = (dest_id + 1) * 1000.0 / (env.cfg.num_edges + 1); bw = env.cfg.v2i_bandwidth_mhz
        else:
            dst_x = 1000.0; bw = env.cfg.v2i_bandwidth_mhz
        d = abs(dst_x - holder_x) + 1.0
        p_dbm = 1.0
        snr = distance_to_snr(d, p_dbm, env.cfg.noise_dbm)
        rate = snr_to_rate_mbps(bw, snr)
        tx_t = tx_time_seconds(t.size_bits, rate)
        mips = (env.cfg.vehicle_compute_mips if dest_kind == "vehicle" else
                env.cfg.edge_compute_mips if dest_kind == "edge" else env.cfg.cloud_compute_mips)
    comp_t = compute_time_seconds(t.cycles, mips)
    # crude queue wait: queue_len / service_rate ≈ qlen * comp_t (since we do 1 task/step)
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
    # try each destination for each of top_k tasks and pick the argmin ETA
    V, E = env.cfg.max_vehicles, env.cfg.num_edges
    # offload indices: 0=local, 1..V=vehicles, V+1..V+E=edges, V+E+1=cloud
    def decode(off):
        if off == 0: return ("local", 0)
        if 1 <= off <= V: return ("vehicle", off-1) if env.cfg.allow_v2v else ("local", 0)
        if V < off <= V+E: return ("edge", off-1-V)
        return ("cloud", 0)

    stride = 2 if env.cfg.power_rule == "distance" else 3
    action = []
    topk_idxs = env._select_topk_tasks_edf(env.cfg.top_k)
    for _i, idx in enumerate(topk_idxs):
        # scan choices
        best_off = 0; best_eta = float("inf")
        for off in range(V + E + 2):
            kind, did = decode(off)
            eta = estimate_finish_time(env, idx, kind, did)
            if eta < best_eta:
                best_eta = eta; best_off = off
        # choose highest priority raw (L-1) to reduce waiting; power mid-bin if used
        prio_raw = env.cfg.priority_levels - 1
        if stride == 2:
            action.extend([best_off, prio_raw])
        else:
            pbin = max(0, min(env.cfg.power_bins-1, env.cfg.power_bins//2))
            action.extend([best_off, prio_raw, pbin])

    # pad if fewer than K tasks
    while len(action) < stride*env.cfg.top_k:
        action.extend([0]*stride)
    return np.array(action, dtype=np.int64)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="../runs/m1_baseline.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["episode","return","steps","completed_total","deadline_miss_total"])
        for ep in range(args.episodes):
            obs, info = env.reset()
            done, ret, steps, comp_tot, miss_tot = False, 0.0, 0, 0, 0
            while not done:
                action = greedy_action(env, obs)
                obs, r, term, trunc, inf = env.step(action)
                ret += float(r); steps += 1
                comp_tot += int(inf.get("completed",0))
                miss_tot += int(inf.get("deadline_misses",0))
                done = term or trunc
            w.writerow([ep, f"{ret:.4f}", steps, comp_tot, miss_tot])
            print(f"[baseline] ep={ep} return={ret:.3f} steps={steps} completed={comp_tot} misses={miss_tot}")
    env.close()
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
