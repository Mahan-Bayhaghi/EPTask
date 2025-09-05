from __future__ import annotations
import argparse, csv, os, numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from rl.utils import load_config
from env.vec_env import EPTaskEnv
from env.metrics import p95, jain_fairness
from env.wrappers import MultiDiscreteToBoxWrapper

ALGO_TO_LOADER = {
    "ppo": PPO.load,
    "sac": SAC.load,
    "ddpg": DDPG.load,
}


def unwrap_env(env):
    base = env
    # unwrap nested wrappers (if any)
    while hasattr(base, "env"):
        base = base.env
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--model", default="../runs/ddpg/latest_model.zip")
    p.add_argument("--algo", choices=["ppo", "sac", "ddpg"], default="ddpg")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out", default="../runs/ddpg_metrics.csv")
    args = p.parse_args()

    cfg = load_config(args.config)

    # build env (wrapped for SAC/DDPG)
    if args.algo in ("sac", "ddpg"):
        env = MultiDiscreteToBoxWrapper(EPTaskEnv(cfg, seed=cfg.get("seed", 42)))
    else:
        env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))

    base_env = unwrap_env(env)

    loader = ALGO_TO_LOADER[args.algo]
    model = loader(args.model, device="cpu")

    # collections
    all_completion_times = []
    all_energies = []
    misses = 0
    per_vehicle_completed = np.zeros(base_env.cfg.max_vehicles, dtype=np.int64)

    for ep in range(args.episodes):
        obs, info = env.reset()
        # clear any per-episode flags on base env
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, inf = env.step(action)
            done = term or trunc
            misses += int(inf.get("deadline_misses", 0))
            # collect completion info
            for idx in list(base_env.completed_indices):
                task = base_env.tasks[idx]
                flag_name = "_metrics_collected_" + str(idx)
                if getattr(base_env, flag_name, False):
                    continue
                if task.completed_step is not None:
                    all_completion_times.append(task.completed_step - task.created_step)
                    all_energies.append(task.energy_j)
                    holder = int(task.holder) if 0 <= int(task.holder) < base_env.cfg.max_vehicles else 0
                    per_vehicle_completed[holder] += 1
                    setattr(base_env, flag_name, True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["completion_time_mean", np.mean(all_completion_times) if all_completion_times else 0.0])
        w.writerow(["completion_time_p95", p95(all_completion_times)])
        w.writerow(["energy_mean", np.mean(all_energies) if all_energies else 0.0])
        total = misses + len(all_completion_times)
        w.writerow(["miss_rate", misses / max(1, total)])
        w.writerow(["fairness", jain_fairness(per_vehicle_completed)])
        w.writerow(["throughput_tasks", len(all_completion_times)])

    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
