from __future__ import annotations
import argparse, os, yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from rl.utils import load_config, set_global_seeds
from env.vec_env import EPTaskEnv
from env.wrappers import MultiDiscreteToBoxWrapper

def make_env(config_path: str, seed: int):
    cfg = load_config(config_path)
    def _init():
        base = EPTaskEnv(cfg, seed=seed)
        return MultiDiscreteToBoxWrapper(base)
    return _init

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--logdir", default="../runs/sac")
    args = p.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "used_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    set_global_seeds(cfg.get("seed", 42))
    env = DummyVecEnv([make_env(args.config, seed=cfg.get("seed", 42))])

    model = SAC("MultiInputPolicy", env,
                learning_rate=3e-4, batch_size=256,
                buffer_size=200_000, train_freq=1, gradient_steps=1,
                tau=0.005, gamma=0.99, ent_coef="auto",
                verbose=1, device="cpu")

    logger = configure(args.logdir, ["stdout", "csv"])
    model.set_logger(logger)

    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.logdir, "latest_model"))
    env.close()
    print(f"Saved SAC model to {args.logdir}/latest_model.zip")

if __name__ == "__main__":
    main()
