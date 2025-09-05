from __future__ import annotations
import argparse, glob, os, csv
import matplotlib.pyplot as plt

def try_float(x):
    try:
        return float(x)
    except Exception:
        return None

def plot_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        xs, ys = [], []

        # --- Mode 1: SB3 progress.csv ---
        if ("time/total_timesteps" in headers) and ("rollout/ep_rew_mean" in headers):
            for row in reader:
                x = try_float(row["time/total_timesteps"])
                y = try_float(row["rollout/ep_rew_mean"])
                if x is not None and y is not None:
                    xs.append(x); ys.append(y)
            xlabel, ylabel = "timesteps", "rollout/ep_rew_mean"

        # --- Mode 2: our per-step logs (rollout_data) ---
        elif ("step" in headers) and ("reward" in headers):
            step_cum = 0
            for row in reader:
                y = try_float(row["reward"])
                if y is None:
                    continue
                xs.append(step_cum); ys.append(y)
                step_cum += 1
            xlabel, ylabel = "steps (cumulative over rows)", "reward"

        # --- Mode 3: our per-episode logs (baseline/eval) ---
        elif ("episode" in headers) and ("return" in headers):
            ep_idx = 0
            for row in reader:
                y = try_float(row["return"])
                if y is None:
                    continue
                xs.append(ep_idx); ys.append(y)
                ep_idx += 1
            xlabel, ylabel = "episode", "return"

        # --- Fallback: last numeric column per row ---
        else:
            for row in reader:
                nums = [try_float(row.get(h, "")) for h in headers]
                nums = [v for v in nums if v is not None]
                if not nums:
                    continue
                xs.append(len(xs)); ys.append(nums[-1])
            xlabel, ylabel = "row", "value"

    if not ys:
        print(f"[plot] No plottable data in {os.path.basename(path)}")
        return

    plt.figure()
    plt.plot(xs, ys)
    plt.title(os.path.basename(path))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out = os.path.splitext(path)[0] + ".png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="../runs")
    args = p.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.logdir, "*.csv")))
    if not csvs:
        print("No CSV logs found in", args.logdir)
        return

    for path in csvs:
        plot_csv(path)

if __name__ == "__main__":
    main()
