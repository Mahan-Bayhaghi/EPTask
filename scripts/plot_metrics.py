from __future__ import annotations
import argparse, csv, os
import matplotlib.pyplot as plt

def load_metrics(path: str):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if len(row) >= 2:
                try:
                    d[row[0]] = float(row[1])
                except:
                    pass
    return d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default="../runs/m1_rl_metrics.csv ../runs/m1_baseline.csv")
    p.add_argument("--out", default="../runs/m1_compare.png")
    args = p.parse_args()

    series = []
    labels = []
    for path in args.inputs:
        d = load_metrics(path)
        series.append([
            d.get("completion_time_mean", 0.0),
            d.get("completion_time_p95", 0.0),
            d.get("energy_mean", 0.0),
            d.get("miss_rate", 0.0),
            d.get("fairness", 0.0),
            d.get("throughput_tasks", 0.0),
        ])
        labels.append(os.path.splitext(os.path.basename(path))[0])

    metrics = ["ct_mean","ct_p95","energy","miss_rate","fairness","throughput"]
    x = range(len(metrics))

    plt.figure(figsize=(10,5))
    for s, label in zip(series, labels):
        plt.plot(x, s, marker="o", label=label)
    plt.xticks(list(x), metrics, rotation=0)
    plt.ylabel("value")
    plt.legend()
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)

if __name__ == "__main__":
    main()
