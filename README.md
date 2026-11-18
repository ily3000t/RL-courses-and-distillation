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
& $python -m rl_courses train --config configs/smoke.json --method fixed --seed 42 --output runs/smoke
& $python -m rl_courses evaluate --run runs/smoke --split validation --output runs/smoke-eval
```

Use a new output directory for each run. `configs/stage1.json` is a starting
point for validation runs on CUDA. Save the final protocol before evaluating
on the `test` and `stress` splits. Keep each run's manifest and scene catalog
with its checkpoints, validation history, and episode records.

## Stage-one comparisons

The network has a 60 m merge lane. The three mainline demands are 450, 1100,
and 2000 vehicles/hour. They share one geometry, observation, action mapping
and reward. SUMO enforces safe speeds and checks the gaps for lane-change
requests. The stress catalog raises Hard demand and uses separate SUMO seeds.

| Method | Training scene selection |
| --- | --- |
| `hard` | Sample only Hard scenes. |
| `uniform` | Sample a difficulty uniformly, then a training scene in that difficulty. |
| `fixed` | Allocate equal decision-step budgets to Easy, Medium, and Hard in that order. |
| `shuffled` | Shuffle the same blocks with a shared final Hard block. |
| `plr` | Discover training scenes and replay ranked scenes according to SAC TD error and staleness. |

Every group uses the same number of SAC decision steps and the same
validation and test catalogs. An episode still active at a common block
boundary is truncated and recorded as `schedule_cut`. The evaluation
environment runs each scene to its natural task endpoint. Check
`schedule.json` and `training_summary.json` for planned and realized budgets.

PLR here is an [adaptation of Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html).
Each level is a scenario ID. Score completed rollouts using the mean absolute
residual of the two SAC critics against the soft TD target. The early rollout
log distinguishes unseen, warm-up, and scored replay selections. Inspect
`plr_episodes.jsonl` to check visitation and scoring.
