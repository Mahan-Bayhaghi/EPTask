from __future__ import annotations
import argparse, csv, os, numpy as np
from rl.utils import load_config
from env.vec_env import EPTaskEnv

def make_greedy_action(env, obs):
    """
    Very simple baseline:
    - Offload: choose 'local' for all K (index 0 in offload head)
    - Priority: highest level (raw = L-1, env will +1 to [1..L])
    - Power: mid bin (index power_bins//2)
    """
    cfg = env.cfg
    off_idx = 0
    prio_raw = cfg.priority_levels - 1
    pbin = max(0, min(cfg.power_bins - 1, cfg.power_bins // 2))
    triplet = [off_idx, prio_raw, pbin]
    action = []
    for _ in range(cfg.top_k):
        action.extend(triplet)
    return np.array(action, dtype=np.int64)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/small.yaml")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="../runs/baseline_eval.csv")
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
                action = make_greedy_action(env, obs)
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
