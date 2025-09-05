# EPTask (CPU-friendly scaffold)

This is a Windows/CPU-friendly scaffold for implementing the RL approach from:
**“EPtask: Deep Reinforcement Learning Based Energy-Efficient and Priority-Aware Task Scheduling for Dynamic Vehicular Edge Computing.”**

## Quick start

1) **Python**: 3.10+ recommended

2) **Create venv**
   - **Windows (PowerShell)**
     ```powershell
     py -3.10 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3) **Install deps (CPU)**
    ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
    ```
If PyTorch fails on Windows, install the CPU wheel from pytorch.org, then re-run.

4) **Train (small config)**
    ```bash
    python scripts/train.py --config configs/small.yaml --timesteps 30000
    ```

5) **Evaluate & plot**
    ```bash
    python scripts/eval.py --config configs/small.yaml --model runs/latest_model.zip
    python scripts/plot.py --logdir runs
    ```
   
**What’s inside**
```
eptask/
  env/                # Gymnasium environment (EDF queues, link/compute/energy)
  rl/                 # Utilities (config loading, seeding)
  configs/            # Scenario configs (placeholders -> swap with Table III later)
  scripts/            # train/eval/plot entrypoints
  tests/              # minimal sanity test
  runs/               # logs & saved model (created at runtime)
```

**Defaults we set (change anytime)**
- Queues: non-preemptive + EDF at all executors
- Action: per selected task — (offload target, priority, power bin)
- Top-K tasks scheduled/step: 4 (configurable)
- Episode horizon: 600 steps (configurable)
- Priority levels: 4 (configurable)
- Power: discrete bins by default; switch to a distance-based rule with power_rule: "distance"

This scaffold aims for speed & clarity. It’s faithful to the paper’s idea but simplified. You can later replace placeholders with exact values from Table III by editing configs/default.yaml.


---

### `eptask/requirements.txt`
```txt
numpy>=1.24
gymnasium>=0.29.1
stable-baselines3>=2.3.0
torch>=2.1.0
matplotlib>=3.8
pyyaml>=6.0.1
tqdm>=4.66
```
