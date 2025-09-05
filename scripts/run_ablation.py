from __future__ import annotations
import argparse, shutil, os
from stable_baselines3 import PPO
from rl.utils import load_config
from env.vec_env import EPTaskEnv
from eval_metrics import main as eval_metrics_main  # reuse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--mode", choices=["baseline","rl"], required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out", default="runs/m1_metrics.csv")
    p.add_argument("--logdir", default="runs/m1_rl")
    args = p.parse_args()

    if args.mode == "baseline":
        # quick proxy: run baseline_heuristic, then compute metrics on its rollouts (or just reuse heuristic CSV)
        print("Use baseline_heuristic.py to generate baseline CSV, and eval_metrics.py for RL.")
        print("This helper is a placeholder to keep CLI consistent.")
        return

    # mode == rl
    cfg = load_config(args.config)
    env = EPTaskEnv(cfg, seed=cfg.get("seed", 42))
    model = PPO("MultiInputPolicy", env, verbose=0, device="cpu")
    os.makedirs(args.logdir, exist_ok=True)
    model.learn(total_timesteps=100000)
    model.save(os.path.join(args.logdir, "latest_model"))
    env.close()

    # evaluate
    import sys
    sys.argv = ["eval_metrics.py", "--config", args.config, "--model", os.path.join(args.logdir, "latest_model.zip"),
                "--episodes", str(args.episodes), "--out", args.out]
    eval_metrics_main()

if __name__ == "__main__":
    main()
