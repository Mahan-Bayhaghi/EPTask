from __future__ import annotations
import argparse, os, yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from rl.utils import load_config, set_global_seeds
from env.vec_env import EPTaskEnv

def make_env(config_path: str, seed: int):
    cfg = load_config(config_path)
    def _init():
        return EPTaskEnv(cfg, seed=seed)
    return _init

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="../configs/default.yaml")
    p.add_argument("--timesteps", type=int, default=100000)
    p.add_argument("--logdir", type=str, default="../runs/ppo")
    args = p.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.logdir, exist_ok=True)
    # snapshot config
    with open(os.path.join(args.logdir, "used_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    set_global_seeds(cfg.get("seed", 42))
    env = DummyVecEnv([make_env(args.config, seed=cfg.get("seed", 42))])

    model = PPO("MultiInputPolicy", env,
                learning_rate=cfg["ppo"]["learning_rate"],
                n_steps=cfg["ppo"]["n_steps"],
                batch_size=cfg["ppo"]["batch_size"],
                n_epochs=cfg["ppo"]["n_epochs"],
                gamma=cfg["ppo"]["gamma"],
                gae_lambda=cfg["ppo"]["gae_lambda"],
                clip_range=cfg["ppo"]["clip_range"],
                ent_coef=cfg["ppo"]["ent_coef"],
                vf_coef=cfg["ppo"]["vf_coef"],
                max_grad_norm=cfg["ppo"]["max_grad_norm"],
                verbose=1, device="cpu")

    logger = configure(args.logdir, ["stdout", "csv"])
    model.set_logger(logger)

    total_timesteps = args.timesteps or cfg["ppo"]["total_timesteps"]
    model.learn(total_timesteps=total_timesteps)

    save_path = os.path.join(args.logdir, "latest_model")
    model.save(save_path)
    print(f"Saved model to {save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
