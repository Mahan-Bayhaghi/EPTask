from __future__ import annotations
import argparse, csv, glob, os
from statistics import mean, pstdev


def load_metric_csv(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f);
        next(r, None)
        for k, v in r:
            try:
                d[k] = float(v)
            except:
                pass
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True, help="glob, e.g., runs/*_metrics_seed*.csv")
    p.add_argument("--out", default="../runs/summary.csv")
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print("No files match", args.pattern);
        return

    rows = [load_metric_csv(f) for f in files]
    keys = sorted({k for r in rows for k in r.keys()})
    agg = {}
    for k in keys:
        vals = [r.get(k) for r in rows if k in r]
        agg[k] = (mean(vals), pstdev(vals)) if vals else (0.0, 0.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n"])
        for k in keys:
            m, s = agg[k]
            w.writerow([k, f"{m:.6f}", f"{s:.6f}", len(files)])
    print("Saved summary to", args.out)


if __name__ == "__main__":
    main()
