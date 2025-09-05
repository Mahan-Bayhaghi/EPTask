from __future__ import annotations
import argparse
from stable_baselines3 import PPO
from rl.utils import load_config
from env.vec_env import EPTaskEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/small.yaml")
    p.add_argument("--model", default="../runs/latest_model.zip")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))
    model = PPO.load(args.model, device="cpu")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done, ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += float(r)
            done = term or trunc
        print(f"Episode {ep} return: {ret:.3f}")

if __name__ == "__main__":
    main()

