from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("completion_time_mean", "Completion Time (Mean)"),
    ("completion_time_p95", "Completion Time (P95)"),
    ("energy_mean", "Energy per Task (Mean)"),
    ("miss_rate", "Deadline Miss Rate"),
    ("fairness", "Jain Fairness"),
    ("throughput_tasks", "Throughput (Tasks)"),
]

METHOD_LABELS = {
    "ppo": "PPO",
    "sac": "SAC",
    "ddpg": "DDPG",
    "heuristic": "Heuristic",
    "local": "Local-only",
    "offload": "Offloading-only",
    "random": "Random",
}


def read_metric_csv(path: str) -> dict:
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f);
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    k, v = row[0], row[1]
                    try:
                        out[k] = float(v)
                    except:
                        pass
    except Exception as e:
        print("[plot_vs_vehicles] skip", path, "->", e)
    return out


def gather(pattern_dir: str):
    """
    Returns:
      data[method][vehicles] = list of dicts (one per seed)
    """
    data = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(pattern_dir, "*.csv")):
        base = os.path.basename(path)
        # filename pattern: {method}_V{vehicles}_seed{n}.csv
        # also accept *_eval_* but ignore those (we want metrics files)
        if "_eval_" in base:
            continue
        parts = base.split("_")
        if len(parts) < 2:
            continue
        method = parts[0]
        if method not in METHOD_LABELS:
            continue
        try:
            # find V{num} in parts
            V = None
            for p in parts:
                if p.startswith("V") and p[1:].split(".")[0].isdigit():
                    V = int(p[1:].split(".")[0])
                    break
            if V is None:
                continue
        except:
            continue
        d = read_metric_csv(path)
        if d:
            data[method][V].append(d)
    return data


def agg_mean_std(values: list[float]):
    if not values: return (0.0, 0.0)
    a = np.asarray(values, dtype=np.float64)
    return (float(a.mean()), float(a.std(ddof=0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default="../runs/grid", help="directory with *_V{vehicles}_seed{seed}.csv")
    p.add_argument("--outdir", default="../runs/grid_plots")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = gather(args.indir)
    if not data:
        print("[plot_vs_vehicles] no data found in", args.indir);
        return

    # Determine vehicles list from data
    vehicles_all = sorted({V for m in data for V in data[m].keys()})

    # 1) One plot per metric
    per_metric_paths = []
    for key, title in METRICS:
        plt.figure(figsize=(7, 4))
        for method in METHOD_LABELS:
            if method not in data:
                continue
            means, stds = [], []
            for V in vehicles_all:
                vals = [d.get(key, 0.0) for d in data[method].get(V, [])]
                m, s = agg_mean_std(vals)
                means.append(m);
                stds.append(s)
            if any(v != 0.0 for v in means):
                plt.plot(vehicles_all, means, marker="o", label=METHOD_LABELS[method])
        plt.xlabel("#vehicles")
        plt.ylabel("value")
        plt.title(title)
        plt.legend()
        outp = os.path.join(args.outdir, f"{key}_vs_vehicles.png")
        plt.tight_layout()
        plt.savefig(outp, dpi=150, bbox_inches="tight")
        print("Saved", outp)
        per_metric_paths.append(outp)

    # 2) Combined 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, (key, title) in enumerate(METRICS):
        ax = axes[i]
        for method in METHOD_LABELS:
            if method not in data:
                continue
            means = []
            for V in vehicles_all:
                vals = [d.get(key, 0.0) for d in data[method].get(V, [])]
                m, _ = agg_mean_std(vals)
                means.append(m)
            if any(v != 0.0 for v in means):
                ax.plot(vehicles_all, means, marker="o", label=METHOD_LABELS[method])
        ax.set_xlabel("#vehicles")
        ax.set_ylabel("value")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0.0, 0.02, 1, 0.92])
    combo_out = os.path.join(args.outdir, "final_vs_vehicles.png")
    fig.savefig(combo_out, dpi=150, bbox_inches="tight")
    print("Saved", combo_out)


if __name__ == "__main__":
    main()
