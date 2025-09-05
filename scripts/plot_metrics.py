from __future__ import annotations
import argparse, csv, os
import matplotlib.pyplot as plt

# Hardcoded list of metric files to compare
HARD_CODED_FILES = [
    "../runs/ppo_metrics.csv",
    "../runs/sac_metrics.csv",
    "../runs/ddpg_metrics.csv",
    "../runs/heuristic_metrics.csv",
    "../runs/local_metrics.csv",
    "../runs/offload_metrics.csv",
    "../runs/random_metrics.csv",
]

# Pretty labels for legend (fallback to filename stem if not matched)
PRETTY_LABELS = {
    "ppo_metrics": "PPO",
    "sac_metrics": "SAC",
    "ddpg_metrics": "DDPG",
    "heuristic_metrics": "Heuristic",
    "local_metrics": "Local-only",
    "offload_metrics": "Offloading-only",
    "random_metrics": "Random",
}

# Metrics to plot
METRIC_ORDER = [
    "completion_time_mean",
    "completion_time_p95",
    "energy_mean",
    "miss_rate",
    "fairness",
    "throughput_tasks",
]
METRIC_SHORT = ["ct_mean", "ct_p95", "energy", "miss_rate", "fairness", "throughput"]
METRIC_TITLE = {
    "completion_time_mean": "Completion Time (Mean)",
    "completion_time_p95": "Completion Time (P95)",
    "energy_mean": "Energy per Task (Mean)",
    "miss_rate": "Deadline Miss Rate",
    "fairness": "Jain Fairness",
    "throughput_tasks": "Throughput (Tasks)",
}


def load_metrics_csv(path: str) -> dict:
    d = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            _ = next(r, None)  # header
            for row in r:
                if len(row) >= 2:
                    k, v = row[0], row[1]
                    try:
                        d[k] = float(v)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[plot_metrics] skip {path}: {e}")
    return d


def pretty_label_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    for key, lab in PRETTY_LABELS.items():
        if key in stem:
            return lab
    return stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../runs/final_compare.png",
                   help="path for the combined 2x3 figure (individual PNGs will be saved alongside)")
    args = p.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load all available files
    series_by_method = {}  # method_label -> {metric_name: value}
    methods_in_order = []
    print("[plot_metrics] Looking for metric files:")
    for path in HARD_CODED_FILES:
        print("  -", path)
        if not os.path.isfile(path):
            print("    -> not found, skipping.")
            continue
        d = load_metrics_csv(path)
        if not d:
            print("    -> unreadable/empty, skipping.")
            continue
        label = pretty_label_from_path(path)
        series_by_method[label] = d
        methods_in_order.append(label)
        print("    -> loaded as", label)

    if not series_by_method:
        print("[plot_metrics] No plottable metric files found. Ensure CSVs exist in 'runs/'.")
        return

    # 1) Individual plots per metric
    for metric, short in zip(METRIC_ORDER, METRIC_SHORT):
        vals = [series_by_method[m].get(metric, 0.0) for m in methods_in_order]

        plt.figure(figsize=(7, 4))
        plt.bar(range(len(methods_in_order)), vals)
        plt.xticks(range(len(methods_in_order)), methods_in_order, rotation=30, ha="right")
        plt.ylabel("value")
        plt.title(METRIC_TITLE.get(metric, metric))
        out_path = os.path.join(out_dir, f"{short}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved", out_path)

    # 2) Combined 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, (metric, short) in enumerate(zip(METRIC_ORDER, METRIC_SHORT)):
        ax = axes[i]
        vals = [series_by_method[m].get(metric, 0.0) for m in methods_in_order]
        ax.bar(range(len(methods_in_order)), vals)
        ax.set_xticks(range(len(methods_in_order)))
        ax.set_xticklabels(methods_in_order, rotation=30, ha="right")
        ax.set_ylabel("value")
        ax.set_title(METRIC_TITLE.get(metric, metric))
    fig.suptitle("Method Comparison (per metric)", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
