"""Compare completed runs and summarize variation across training seeds."""

from collections import defaultdict
import json
from statistics import mean, stdev

from .records import read_json
from .sampling import BASELINE_METHODS


METRICS = ("success_rate", "collision_rate", "timeout_rate", "missed_merge_rate")


def _completed(path):
    if read_json(path / "status.json")["state"] != "completed":
        raise ValueError(f"Run is incomplete: {path}")


def _moments(values):
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else None}


def summarize_suite(root, split, success, collision):
    if not (0 <= success <= 1 and 0 <= collision <= 1):
        raise ValueError("Pass rate thresholds in [0, 1].")
    methods = defaultdict(dict)
    reference = None
    for run in sorted(root.glob("*/seed-*/train")):
        evaluation = run.parent / split
        _completed(run)
        _completed(evaluation)
        metadata = read_json(run / "manifest.json")
        eval_metadata = read_json(evaluation / "manifest.json")
        config = metadata["config"].copy()
        method = config.pop("method")
        seed = metadata["seed"]
        if seed in methods[method]:
            raise ValueError(f"Duplicate run for {method}, seed {seed}.")
        if (eval_metadata["split"] != split or eval_metadata["seed"] != seed
                or eval_metadata["config"] != metadata["config"]
                or eval_metadata["training_git_sha"] != metadata["git_sha"]
                or eval_metadata["git_sha"] != metadata["git_sha"]):
            raise ValueError(f"Evaluation provenance differs from its training run: {evaluation}")
        protocol = (config, metadata["git_sha"], metadata["versions"],
                    eval_metadata["versions"], read_json(run / "scenes.json"))
        if reference is None:
            reference = protocol
        elif reference != protocol:
            raise ValueError(f"Experiment protocol differs across runs: {run}")
        rows = [
            json.loads(line) for line in
            (run / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        history = [{"step": row["step"], **row["summary"]["hard"]} for row in rows]
        first_hit = next((
            row["step"] for row in history
            if row["success_rate"] >= success and row["collision_rate"] <= collision
        ), None)
        methods[method][seed] = {
            "seed": seed,
            "evaluation": read_json(evaluation / "summary.json")["hard"],
            "validation_steps_to_target": first_hit,
            "hard_validation": history,
            "training": read_json(run / "training_summary.json"),
        }

    if not methods:
        raise ValueError(f"No completed experiment directories found at {root}.")
    if set(methods) != set(BASELINE_METHODS):
        raise ValueError(f"Expected completed runs for all five methods at {root}.")
    expected = set(next(iter(methods.values())))
    if any(set(runs) != expected for runs in methods.values()):
        raise ValueError("Each method must include the same training seeds.")
    summaries = {}
    for method, runs in methods.items():
        trials = [runs[seed] for seed in sorted(runs)]
        hits = [row["validation_steps_to_target"] for row in trials
                if row["validation_steps_to_target"] is not None]
        summaries[method] = {
            "hard_evaluation": {
                metric: _moments([row["evaluation"][metric] for row in trials])
                for metric in METRICS
            },
            "success_and_collision_targets_met": len(hits),
            "steps_to_target_when_met": _moments(hits) if hits else None,
            "runs": trials,
        }
    return {"git_sha": reference[1], "split": split, "seeds": sorted(expected),
            "success_threshold": success, "collision_threshold": collision,
            "methods": summaries}
