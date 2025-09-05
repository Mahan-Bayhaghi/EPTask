from __future__ import annotations
import argparse, csv, os, numpy as np
from stable_baselines3 import PPO
from rl.utils import load_config
from env.vec_env import EPTaskEnv

def active_tasks(obs) -> int:
    # last task feature is "active flag" (1 if not done)
    return int(np.asarray(obs["tasks"][:, -1]).sum())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/small.yaml")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="../runs/rollouts_random.csv")
    p.add_argument("--policy", choices=["random","model"], default="random")
    p.add_argument("--model", help="path to saved model.zip if --policy model")
    args = p.parse_args()

    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))

    model = None
    if args.policy == "model":
        if not args.model:
            raise ValueError("Provide --model when --policy model")
        model = PPO.load(args.model, device="cpu")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(
            ["episode","step","reward","completed","deadline_misses","active_tasks"]
        )
        for ep in range(args.episodes):
            obs, info = env.reset()
            done, step = False, 0
            while not done:
                if args.policy == "random":
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, inf = env.step(action)
                w.writerow([ep, step, float(r), inf.get("completed",0),
                            inf.get("deadline_misses",0), active_tasks(obs)])
                done = term or trunc; step += 1
    env.close()
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
