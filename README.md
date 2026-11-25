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

## Compare the density backoff ablation

```powershell
.\scripts\run_stage1.ps1 -Comparison backoff -Profiles runs/stage1-opportunities/opportunities.json -Output runs/stage1-backoff-validation
```

The runner trains `opportunity-progress` and `opportunity-progress-backoff`
with the same seeds, profile bundle, and interaction budget. The backoff
variant tracks the latest completed episodes in each opportunity subgroup
and its parent density. With `w` episodes per full window and `n` observed
episodes in the subgroup, it compares two local windows of length
`k = min(w, floor(n/2))`. Its priority score blends the resulting local
progress with density progress using local weight `k/w`. With fewer than
two subgroup episodes, the parent provides the full score. Once the
subgroup has two full windows, its score is entirely local. The curriculum
log records parent progress, subgroup counts, local scores, and weights.

## Compare the reference-based merge curriculum

| Method | Within-density priority |
| --- | --- |
| `merge-reference-failure` | Reference evidence times recent pre-merge failure estimate |
| `merge-failure` | Recent pre-merge failure estimate |
| `merge-reference` | Reference evidence |
| `plr` | The existing SAC TD-error and staleness baseline |

Reference evidence is one if any fixed-speed probe on the same training
scene completed a SUMO-accepted merge and the probe episode had no
collision. An accepted merge followed by a timeout counts as evidence for
the merge itself. A scene without reference evidence remains eligible for
uniform sampling. Probes never enter the SAC replay buffer.

Pre-merge failures include timeouts before merging and missed merges.
Every completed episode contributes a binary observation to this rate.
Schedule cutoffs are excluded. With \(f\) failures in the last \(n\) valid
episodes on the scene, the smoothed estimate is `(f + 1) / (n + 2)`.
`configs/stage1.json` uses windows of at most eight episodes and 10,000
uniform warm-up decision steps. Each of the three merge methods samples
traffic density uniformly with a separate seed stream. Within the chosen
density, the sampler uses 20% uniform sampling after warmup and otherwise
samples by priority. If every priority is zero, it samples uniformly.
The density stream matches across the three methods by episode index.
Actual decision counts per density depend on episode length and are
recorded separately.

From the repository root, run the comparison on the existing development
stress catalog with seed `20261120`:

```powershell
.\scripts\run_stage1.ps1 -Comparison merge-curriculum -Profiles runs/stage1-opportunities-v1/opportunities.json -Split stress -StressSeed 20261120 -Output runs/stage1-merge-curriculum-dev-v1
```

This runs seeds 42–46 for 120,000 decision steps each. Checkpoints are
selected on the Hard validation set and evaluated on the specified
pressure catalog. For the final comparison, freeze a new stress seed and
scenario count before viewing its results. The runner accepts
`-StressCount` with `-StressSeed` to set that count. Report full-episode
success, collision, and combined pre-merge failure rates. The two
reference methods must use the same profile bundle, and its collection
cost is recorded separately from SAC interactions.

`merge_reference_evidence.json` records the scene labels. Each row in
`merge_curriculum_episodes.jsonl` records the conditional sampling
probability and the statistics available when the scene was selected,
followed by the observed outcome and its eligibility for rate updates.

## Compare reference weighting with hard gating

`merge-reference-weighted` uses the same recent failure estimate as
`merge-failure`. It multiplies that estimate by two for scenes with
positive reference evidence and by one for all other scenes. Every
scene can enter priority sampling. The density stream, warmup, failure
estimator, and uniform sampling fraction follow the original comparison.
The episode log records the selected scene's `reference_weight`.

```powershell
.\scripts\run_stage1.ps1 -Comparison merge-weighting -Profiles runs/stage1-opportunities-v1/opportunities.json -Split stress -StressSeed 20261120 -Output runs/stage1-merge-weighting-dev-v1
```

This comparison uses the weighted method, hard gating
(`merge-reference-failure`), failure-only sampling (`merge-failure`), and PLR.
The original `merge-curriculum` comparison still uses its original four
methods. Use the development set to compare success and pre-merge failure
and the episode logs to compare coverage of Hard scenes without positive
reference evidence. Reserve a new stress seed for the final comparison.

## Distill a SAC teacher

Choose a completed teacher training run and its validation-selected
`best.zip`. Start with PLR, then repeat with the weighted-reference teacher.
The runner collects stochastic SAC rollouts from the training catalog only,
using exactly 64 decisions per scene by default. It saves deterministic
teacher actions and the pre-tanh diagonal Gaussian parameters for the same
observations. Short episodes restart within their scene quota. Collection
cuts and actual sampling cost are recorded separately from student training.

```powershell
.\scripts\run_stage2.ps1 -TeacherRun runs/stage1-merge-curriculum-dev-v1/plr/seed-42/train -Output runs/stage2-plr-validation-v1
.\scripts\run_stage2.ps1 -TeacherRun runs/stage1-merge-weighted-only-dev-v1/merge-reference-weighted/seed-42/train -ReuseScratch runs/stage2-plr-validation-v1 -Output runs/stage2-weighted-validation-v1
```

Each student uses the same `[64, 64]` SAC actor architecture. `scratch`
trains a small SAC policy with uniform scene sampling under the stage-one
interaction budget. `bc` minimizes MSE against the teacher's deterministic
action. `soft` minimizes teacher-to-student KL in pre-tanh Gaussian space.
BC and soft share the same dataset, seed, initialization, optimizer settings
and minibatch sequence. Neither uses a critic loss or online RL fine-tuning.
The second run reuses the same scratch checkpoints and evaluates them under
its recorded evaluation protocol. A single runner invocation holds the
teacher fixed across the student training seeds.

Use `-Split test` or `-Split stress -StressSeed <seed> -StressCount <count>`
for a frozen comparison. Teachers and students use deterministic closed-loop
evaluation and select checkpoints using Hard validation. The report contains
per-seed BC-versus-soft metrics, student-versus-teacher success gaps and
retention, and separate teacher training, collection and student budgets.
Each evaluation exports and measures the deterministic CPU actor (FP32
TorchScript, batch size one, one thread). Those latencies cover actor
execution alone. The training archive is used for reproducing experiments.

For an end-to-end check, use a teacher trained with `configs/smoke.json`
and pass `-Config configs/smoke.json -StudentConfig configs/distillation-smoke.json
-Seeds 42 -StepsPerScene 8 -Device cpu`.

### Compare full-episode and truncated teacher collection

Both modes keep 64 training rows per scene by default. In `full-episode`
mode the collector lets the last episode reach its normal end, including
environment timeouts and failures. It then samples uniformly without
replacement from all collected decisions in that scene. The first 64
rollout decisions use the same action noise as the truncated baseline.
The continuation and row selection use separate random streams derived
from the collection seed.

`decision_steps` counts the rows the student can train on.
`rollout_decision_steps` counts the simulator decisions used to collect them.
`retained_tail_steps_by_difficulty` counts sampled decisions beyond the
original quota. In `episodes.json`, `start` and `stop` index the retained
dataset while `rollout_start` and `rollout_stop` index all collected
decisions. Full-episode datasets also record each retained row's raw
position as `rollout_indices`.

From the repository root, run both modes with the same weighted-reference
teacher, student seeds and frozen evaluation scenes. The truncated suite
below rebuilds the original dataset and reuses the trained BC and soft
students after verifying the dataset hash. Both suites reuse the same
scratch checkpoints, and each suite evaluates its checkpoints on all
requested splits without repeating training.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$common = @{
    Python = $python
    TeacherRun = 'runs/stage1-merge-weighted-only-dev-v1/merge-reference-weighted/seed-42/train'
    ReuseScratch = 'runs/stage2-plr-validation-v1'
    Split = @('validation', 'test', 'stress')
    StressSeed = 20261129
    StressCount = 96
}
$root = 'runs/stage2-weighted-collection-ablation-v1'
.\scripts\run_stage2.ps1 @common -CollectionMode truncate -ReuseDistilled runs/stage2-weighted-validation-v1 -Output "$root/truncate"
.\scripts\run_stage2.ps1 @common -CollectionMode full-episode -Output "$root/full-episode"
foreach ($split in $common.Split) {
    & $python -B -m rl_courses report-collection-ablation `
        --truncate-suite "$root/truncate" --full-episode-suite "$root/full-episode" `
        --split $split --output "$root/paired-$split"
    if ($LASTEXITCODE -ne 0) { throw "Paired report failed for $split." }
}
```

Use the pressure set as the primary comparison. The paired report tracks
Hard success, missed merges, collisions and timeouts for BC and soft
distillation. It reports percentage-point changes and exploratory intervals
from joint resampling of student seeds and scenarios with the teacher
checkpoint held fixed. When measuring deployment latency, run comparisons
serially on the same machine.

### Sample around the merge decision

Use one full-episode rollout pool for both the uniform and merge-phase
datasets. With a budget of 64 rows per scene, merge-phase sampling reserves
up to 16 rows from the last 50 metres of the approach and up to 16 from
within the merge zone before entering the target lane. It fills the
remaining quota uniformly from all unselected rows in that scene. Sampling
is without replacement and depends on the recorded state, independent of
the episode outcome.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$teacher = 'runs/stage1-merge-weighted-only-dev-v1/merge-reference-weighted/seed-42/train'
$root = 'runs/stage2-weighted-phase-sampling'
& $python -B -m rl_courses collect-teacher --run $teacher --seed 104730 --steps-per-scene 64 `
    --collection-mode full-episode --save-rollout-pool --output "$root/uniform/collection"
if ($LASTEXITCODE -ne 0) { throw 'Teacher collection failed.' }
& $python -B -m rl_courses sample-teacher --collection "$root/uniform/collection" --output "$root/merge-phase/collection"
if ($LASTEXITCODE -ne 0) { throw 'Merge-phase sampling failed.' }
```

Point `train-student --data` to each collection in turn. Hold the student
configuration, seed list, checkpoint selection, and evaluation scenes fixed.
Use the saved pool hash and the dataset hashes to verify which examples
each student received. Both groups share the rollout cost. Only the
merge-phase group incurs the additional offline sampling cost.

### Relabel states visited by a student

Use the uniform full-episode dataset as the control. Choose a completed
student trained on that dataset and replay its validation-selected
checkpoint once per training scene with deterministic actions. For each
scene, replace up to 16 observations in each of the two priority phases
with observations from that student's matching scene and phase. The
original teacher supplies the new action and Gaussian targets. The
student supplies the rollout actions used to visit the states.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$baseline = 'runs/stage2-weighted-phase-sampling-v1/uniform/collection'
$pilot = 'runs/stage2-weighted-collection-ablation-v1/full-episode/bc/seed-42/train'
$output = 'runs/stage2-weighted-student-relabeling/student-relabeling/collection'
& $python -B -m rl_courses relabel-student --collection $baseline --student-run $pilot --output $output
if ($LASTEXITCODE -ne 0) { throw 'Student relabeling failed.' }
```

Train both methods with `train-student --data` and the original seed list
and optimizer budget. The number of rows per scene and phase stays fixed.
The summary records additional student rollout decisions and teacher
queries. In the output dataset, `student_rollout_indices` points to
`student_rollouts.npz` for substituted rows and is `-1` for original rows.

### Replay short corrections from the same student state

Run the fixed BC student on the Hard training catalog. At its first decision
within 50 metres of the merge entrance and its first unmerged decision on
the merge lane, replay the recorded action prefix. Each replay checks that
the anchor matches the original observation and active traffic state.
Branches use the student, replace only acceleration, replace only the
merge request, or replace both with teacher outputs. A replacement lasts
up to four decision steps or until the vehicle merges. The fixed student
continues to the original episode end in all four branches.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$teacher = 'runs/stage1-merge-weighted-only-dev-v1/merge-reference-weighted/seed-42/train'
$student = 'runs/stage2-weighted-collection-ablation-v1/full-episode/bc/seed-42/train'
& $python -B -m rl_courses replay-corrections --teacher-run $teacher --student-run $student `
    --intervention-steps 4 --output runs/stage2-weighted-merge-correction-replay-v1
if ($LASTEXITCODE -ne 0) { throw 'Correction replay failed.' }
```

Use repeated `--scene` flags for a small subset. The summary counts rescues
from missed merges and pre-merge timeouts, regressions from successes,
and the separate simulation and teacher-query costs. For a rescued anchor,
the diagnostics prefer a single changed component. When both single
changes succeed, the faster merge is selected, with acceleration breaking
an exact tie. The JSONL records retain the original prefix, the matched
traffic snapshot, and actions during the intervention.

### Distill the verified acceleration corrections

Build one correction target bundle from the five completed BC replays.
This uses approach anchors where the student failed before merging and
the acceleration-only intervention changed the terminal result to success.
Each record retains both teacher action outputs for matched supervision.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$base = 'runs/stage2-weighted-phase-sampling-v1/uniform/collection'
$output = 'runs/stage2-weighted-correction-component-v1'
$replayArgs = @('--replay', 'runs/stage2-weighted-merge-correction-replay-v1')
foreach ($seed in 43..46) {
    $replayArgs += @('--replay', "runs/stage2-weighted-merge-correction-replay-seed-$seed-v1")
}
& $python -B -m rl_courses build-correction-targets --collection $base @replayArgs --output "$output/corrections"
if ($LASTEXITCODE -ne 0) { throw 'Correction target build failed.' }

foreach ($seed in 42..46) {
    foreach ($supervision in @('both', 'acceleration')) {
        & $python -B -m rl_courses train-student --config configs/stage1.json `
            --student-config configs/distillation.json --method bc --seed $seed --data $base `
            --corrections "$output/corrections" --correction-batch-size 16 --correction-supervision $supervision `
            --output "$output/$supervision/seed-$seed/train"
        if ($LASTEXITCODE -ne 0) { throw "Correction training failed for seed $seed." }
    }
}
```

Both arms draw 240 ordinary and 16 correction rows per 256-row batch.
Ordinary rows supervise both actions. Correction rows in the acceleration
arm supervise acceleration only. The loss averages active dimensions
within each row, then averages equally across rows. Both arms use the
same sampling sequence for the same seed, and export the existing actor.

Use `--correction-supervision acceleration-equal` for the matched-weight
control. For each correction row, `both` weights acceleration and the
merge request by 0.5 each, `acceleration` weights acceleration by 1.0,
and `acceleration-equal` weights acceleration by 0.5. The latter two
modes give the merge request zero direct weight on correction rows.
Ordinary rows and the batch denominator are identical across modes.
Reuse the correction bundle and seeds, and keep 16 correction rows per
batch. Compare with `both` to test removal of the merge-request loss at
fixed acceleration weight, and with `acceleration` to test the
acceleration weight at fixed action supervision.

### Diagnose waiting in the paired pressure replays

Use the archived fresh pressure comparison of `both` and
`acceleration-equal`. The diagnostic selects the paired scenes in which
`both` missed the merge and the equal-weight policy either succeeded or
timed out before merging. Each policy must reproduce its saved evaluation.
The trace records low-speed time, gap proxies, and actual lane-change
requests and SUMO responses.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
& $python -B -m rl_courses diagnose-waiting --suite runs/stage2-weighted-correction-equal-v1 `
    --intervention-steps 4 --output runs/stage2-pressure-waiting-diagnostics-v1
if ($LASTEXITCODE -ne 0) { throw 'Pressure waiting diagnostic failed.' }
```

On the timed-out equal-weight trajectories, the acceleration probe starts
after one second at or below 1 m/s near the merge. The request probe starts
at the first decision on the merge lane covering a gap-proxy tick with no
request. Each probe verifies the replayed traffic state and hands control
back to the student after four decisions or upon merging. The JSON case
table and JSONL traces are for diagnosis. They use a previously evaluated
stress split.

### Probe acceleration before sustained waiting

Run the equal-weight student's saved policy on the Hard training catalog.
The two anchors are the first unmerged decision on the merge lane above
1 m/s and the first decision there at a speed in (1, 5] m/s. Both precede
the first tick at or below 1 m/s on that lane. At each anchor, replay the
control and replace only acceleration with the reference-weighted
teacher for up to four decisions. Use the original episode deadline and
require the student to continue after the intervention.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$teacher = 'runs/stage1-merge-weighted-only-dev-v1/merge-reference-weighted/seed-42/train'
$root = 'runs/stage2-weighted-pre-wait-probe-v1'
foreach ($seed in 42..46) {
    $student = "runs/stage2-weighted-correction-equal-v1/acceleration-equal/seed-$seed/train"
    & $python -B -m rl_courses replay-pre-wait --teacher-run $teacher --student-run $student `
        --intervention-steps 4 --output "$root/seed-$seed"
    if ($LASTEXITCODE -ne 0) { throw "Pre-wait replay failed for seed $seed." }
}
```

The primary cohort consists of baseline timeouts without merging after
spending at least one continuous second at or below 1 m/s on the merge
lane. The summary records rescues, handback timing, regressions and
other outcomes. The case table records both verified branches at
each available anchor.

### Test teacher takeover through the end of the episode

Use the frozen pre-wait suite with its `protocol.json` and completed
`seed-*/cases.json` runs. Select the sustained waiting timeouts from the
Hard training catalog. At the recorded merge-lane entry decision,
verify the student control and replay a matched branch in which the same
teacher controls acceleration and merge requests through termination.
The original episode deadline applies even after merging. Run the teacher
from the start once per distinct scenario to measure what it achieves
when it also controls the approach.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
& $python -B -m rl_courses diagnose-takeover --suite runs/stage2-weighted-pre-wait-probe-v1 `
    --output runs/stage2-weighted-teacher-takeover-v1
```

Use `cases.json` for the paired outcomes and `teacher_from_start.json`
for the per-scenario outcomes and entry speeds. Keep the two denominators
separate when reporting results.

### Compare small SAC warm starts at matched online budgets

Use five matched student seeds. All arms use the uniform training sampler
and the same small SAC architecture and online settings. Initialize the
ordinary and correction BC arms from their fixed final offline checkpoints,
which share a teacher, demonstration dataset and optimizer budget. Transfer
the actor's mean and hidden layers. Initialize the log-standard-deviation
head, critic, optimizer and replay buffer with the ordinary SAC procedure
in all arms. Save the policy at 0, 15,000 and 30,000 online decisions.

```powershell
$python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe'
$suite = 'runs/stage2-warm-start-budget-v1'
& $python -B -m rl_courses freeze-warm-start --config configs/stage1.json `
    --student-config configs/distillation.json `
    --bc-root runs/stage2-weighted-collection-ablation-v1/full-episode/bc `
    --correction-root runs/stage2-weighted-correction-equal-v1/acceleration-equal `
    --output $suite
& $python -B -m rl_courses run-warm-start --suite $suite --workers 6
& $python -B -m rl_courses report-warm-start --suite $suite --output "$suite/report-dev"
```

The default development pressure catalog uses seed 20261203. The final
pressure catalog uses seed 20261205 and stays unused during development.
After freezing the comparison, run the saved checkpoints on this second
catalog using `--split final-stress` with the run and report commands.
The runner reuses completed training and evaluation attempts. The report
includes paired differences, separate archived preparation and online
interaction costs, and the first measured budget with at least 60%
success and zero observed collisions. This threshold is a simulator
comparison criterion.
