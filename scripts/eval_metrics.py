from __future__ import annotations
import argparse, csv, os, numpy as np
from stable_baselines3 import PPO
from rl.utils import load_config
from env.vec_env import EPTaskEnv
from env.metrics import p95, jain_fairness

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--model", default="../runs/m1_rl/latest_model.zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="../runs/m1_rl_metrics.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))
    model = PPO.load(args.model, device="cpu")

    # collections
    all_completion_times = []
    all_energies = []
    misses = 0
    per_vehicle_completed = np.zeros(env.cfg.max_vehicles, dtype=np.int64)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        t0 = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, inf = env.step(action)
            done = term or trunc
            misses += int(inf.get("deadline_misses", 0))
            # collect completion info via flags set during processing
            for idx in list(env.completed_indices):
                task = env.tasks[idx]
                if getattr(env, "_metrics_collected_"+str(idx), False):  # avoid double count
                    continue
                if task.completed_step is not None:
                    all_completion_times.append(task.completed_step - task.created_step)
                    all_energies.append(task.energy_j)
                    holder = int(task.holder) if 0 <= int(task.holder) < env.cfg.max_vehicles else 0
                    per_vehicle_completed[holder] += 1
                    setattr(env, "_metrics_collected_"+str(idx), True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["completion_time_mean", np.mean(all_completion_times) if all_completion_times else 0.0])
        w.writerow(["completion_time_p95", p95(all_completion_times)])
        w.writerow(["energy_mean", np.mean(all_energies) if all_energies else 0.0])
        w.writerow(["miss_rate", misses / max(1, (misses + len(all_completion_times)))])
        w.writerow(["fairness", jain_fairness(per_vehicle_completed)])
        w.writerow(["throughput_tasks", len(all_completion_times)])

    print(f"Wrote metrics to {args.out}")

if __name__ == "__main__":
    main()
