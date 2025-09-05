from __future__ import annotations
import argparse
from rl.utils import load_config
from env.vec_env import EPTaskEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))

    total_miss, total_completed, total_tasks = 0, 0, 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # random policy; we just want feasibility pressure
            obs, r, term, trunc, inf = env.step(action)
            total_miss += int(inf.get("deadline_misses", 0))
            total_completed += int(inf.get("completed", 0))
            total_tasks += int(inf.get("completed", 0)) + int(inf.get("deadline_misses", 0))
            done = term or trunc

    env.close()
    print(f"episodes={args.episodes} completed={total_completed} misses={total_miss} "
          f"miss_rate={(total_miss/max(1,total_tasks)):.3f}")

if __name__ == "__main__":
    main()
