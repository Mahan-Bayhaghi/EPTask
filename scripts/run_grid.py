from __future__ import annotations
import argparse, os, subprocess, sys, yaml, shutil, tempfile

METHODS_LEARN = ["ppo", "sac", "ddpg"]
METHODS_FIXED = ["heuristic", "local", "offload", "random"]


def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def write_temp_config(base_cfg_path: str, vehicles: int, out_dir: str) -> str:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("env", {})["max_vehicles"] = vehicles
    # Optionally tune arrivals with scale
    lam = cfg["env"].get("task_arrival_lambda", 1.3)
    cfg["env"]["task_arrival_lambda"] = lam
    tmp_path = os.path.join(out_dir, f"config_V{vehicles}.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return tmp_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--vehicles", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--outdir", default="../runs/grid")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tmpcfg_dir = os.path.join(args.outdir, "_configs")
    os.makedirs(tmpcfg_dir, exist_ok=True)

    for V in args.vehicles:
        cfg_path = write_temp_config(args.config, V, tmpcfg_dir)

        for seed in args.seeds:
            # Learning methods
            for m in METHODS_LEARN:
                logdir = os.path.join(args.outdir, f"{m}_V{V}_seed{seed}")
                os.makedirs(logdir, exist_ok=True)
                if m == "ppo":
                    run([sys.executable, "./train.py", "--config", cfg_path, "--timesteps", str(args.timesteps),
                         "--logdir", logdir])
                    model = os.path.join(logdir, "latest_model.zip")
                    run([sys.executable, "./eval_metrics.py", "--config", cfg_path, "--algo", "ppo",
                         "--model", model, "--episodes", str(args.episodes),
                         "--out", os.path.join(args.outdir, f"{m}_V{V}_seed{seed}.csv")])
                elif m == "sac":
                    run([sys.executable, "./train_sac.py", "--config", cfg_path, "--timesteps",
                         str(args.timesteps), "--logdir", logdir])
                    model = os.path.join(logdir, "latest_model.zip")
                    run([sys.executable, "./eval_metrics.py", "--config", cfg_path, "--algo", "sac",
                         "--model", model, "--episodes", str(args.episodes),
                         "--out", os.path.join(args.outdir, f"{m}_V{V}_seed{seed}.csv")])
                elif m == "ddpg":
                    run([sys.executable, "./train_ddpg.py", "--config", cfg_path, "--timesteps",
                         str(args.timesteps), "--logdir", logdir])
                    model = os.path.join(logdir, "latest_model.zip")
                    run([sys.executable, "./eval_metrics.py", "--config", cfg_path, "--algo", "ddpg",
                         "--model", model, "--episodes", str(args.episodes),
                         "--out", os.path.join(args.outdir, f"{m}_V{V}_seed{seed}.csv")])

            # Fixed baselines
            for mode in METHODS_FIXED:
                run([sys.executable, "./baseline_heuristic.py", "--config", cfg_path,
                     "--episodes", str(args.episodes), "--mode", mode,
                     "--out", os.path.join(args.outdir, f"{mode}_eval_V{V}_seed{seed}.csv"),
                     "--metrics_out", os.path.join(args.outdir, f"{mode}_V{V}_seed{seed}.csv"),
                     ])

    print("Grid complete. Metric CSVs are under", args.outdir)


if __name__ == "__main__":
    main()
