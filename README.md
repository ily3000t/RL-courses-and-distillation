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

## Compare all methods across shared seeds

```powershell
.\scripts\run_stage1.ps1 -Output runs/stage1-validation
```

That command trains and evaluates all five methods with seeds 42–46 on
validation scenes. The report includes Hard learning curves, scene metrics
for each seed, across-seed means and standard deviations, and the first
validation checkpoint to meet both thresholds. It counts runs that never
meet the thresholds separately. The defaults are provisional. Set
`-SuccessThreshold` and `-CollisionThreshold` before a formal comparison.
Use `-Config configs/smoke.json -Seeds 42` for a short end-to-end check.

After freezing the protocol, use the `evaluate` command on the saved
checkpoints with `--split test` or `--split stress`. The checkpoint is selected
on Hard validation, in order of highest success rate, lowest collision
rate, lowest timeout rate, and highest mean return.

## Diagnose merge failures

```powershell
& $python -B -m rl_courses diagnose --run runs/stage1-baselines-v1/plr/seed-42/train --run runs/stage1-baselines-v1/hard/seed-42/train --split stress --scene stress-hard-000 --scene stress-hard-001 --output runs/stress-replay
```

The replay uses saved policies and constant-target-speed approaches under
the same SUMO safety settings. It writes local episode summaries and
per-tick traces with acceleration, merge intent, lane position relative
to the merge end, front/rear gaps and observed request blocking.
The gap proxy compares SUMO neighbor clearances against secure gaps
while ego is on the merge lane with more than 3 m remaining. It measures
an available opportunity under the current trajectory.

## Profile training opportunities

```powershell
& $python -B -m rl_courses profile --config configs/stage1.json --output runs/stage1-opportunities
```

Profiles run three fixed approach speeds (12, 19 and 26 m/s) on the
training catalog. The opportunity group within each density is based on
how many reference approaches SUMO actually allowed to merge. The
diagnostic clearance proxy missed observed accepted requests, so it is
kept as a trace field. A zero count means that these probes found no
accepted merge. The bundle stores the setup, scene seeds and profiling
cost separately from SAC interaction steps.

```powershell
& $python -B -m rl_courses train --config configs/stage1.json --method opportunity-progress --profiles runs/stage1-opportunities/opportunities.json --seed 42 --output runs/opportunity-42
```

`opportunity-progress` and `density-progress` use the same scheduler.
They start from a uniform density and uniform scene within density.
After warmup, the scheduler favors groups with improving success across
two windows of completed episodes and reserves a fraction of sampling
for the original distribution. Scheduler cutoffs are excluded from
feedback. Selection probabilities and observed outcomes are recorded
in `curriculum_episodes.jsonl`. The density ablation groups by demand.
The opportunity method subdivides each density using its probe results.

## Compare the opportunity curriculum with PLR and the density ablation

```powershell
.\scripts\run_stage1.ps1 -Comparison curriculum -Profiles runs/stage1-opportunities/opportunities.json -Output runs/stage1-curriculum-validation
```

Use the same locked stress seed across methods after the protocol is
finalized. To evaluate already trained checkpoints on that separate
catalog, use `evaluate --split stress --stress-seed <integer>` with each
saved `--run` and its own new output directory. The runner accepts
`-Split stress -StressSeed <integer>` for the same mode. Every holdout is
disjoint from the original catalog and is saved in the evaluation manifest.
The comparison reports Hard success, pre-merge timeout and missed merge
rates. Select and freeze the holdout seed without inspecting its results.
