# RL-courses-and-distillation

Reinforcement learning curricula for autonomous highway on-ramp merging in Eclipse SUMO.
Use the existing `pytorch` environment for training and evaluation.

Experiment metadata and outputs live under the ignored `runs/` directory.
Version experiment configurations and scripts. Keep model weights and raw outputs local.

## Run on Windows

From the repository root, use the existing PyTorch environment and put the
SUMO binaries (`sumo` and `netconvert`) on PATH. The tested setup uses
Python 3.10, PyTorch 2.5.1, and SUMO 1.22.0. TraCI communicates over localhost.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
& $python -B -m unittest discover -s tests -v
& $python -m rl_courses train --config configs/smoke.json --seed 42 --output runs/smoke
& $python -m rl_courses evaluate --run runs/smoke --split validation --output runs/smoke-eval
```

Use a new output directory for each run. `configs/stage1.json` is a starting
point for validation runs on CUDA. Save the final protocol before evaluating
on the `test` and `stress` splits. Keep each run's manifest and scene catalog
with its checkpoints, validation history, and episode records.
