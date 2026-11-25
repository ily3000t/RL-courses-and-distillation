"""Verify and report paired estimates for the warm-start budget experiment."""

import numpy as np

from .distillation_report import _hard_indicators
from .evaluation import summarize
from .records import read_json
from .warm_start import ARMS, _completed_attempt, file_hash


COMPARISONS = (
    ("bc", "scratch"), ("acceleration-equal", "bc"), ("acceleration-equal", "scratch"),
)
BOOTSTRAP_SEED = 104731
BOOTSTRAP_REPETITIONS = 10000


def _load_run(suite, arm, seed, split, protocol, protocol_hash):
    design = protocol["config"]
    job = suite / arm / f"seed-{seed}"
    train = _completed_attempt(job / "train")
    evaluation = _completed_attempt(job / split)
    if train is None or evaluation is None:
        raise ValueError(f"Missing completed {split} result for {arm}, seed {seed}.")
    identity = {"arm": arm, "seed": seed, "protocol_sha256": protocol_hash, "git_sha": protocol["git_sha"],
                "versions": protocol["versions"]}
    for folder, expected in ((train, identity), (evaluation, {**identity, "split": split, "training": str(train)})):
        meta = read_json(folder / "manifest.json")
        if any(meta[key] != value for key, value in expected.items()):
            raise ValueError(f"Result provenance differs from the frozen suite: {folder}.")

    spec = read_json(train / "training_summary.json")
    if (spec["online_decisions"] != design["budgets"][-1]
            or sum(spec["steps_by_difficulty"].values()) != spec["online_decisions"]):
        raise ValueError(f"The saved online decision counts disagree: {train}.")
    entries = read_json(train / "checkpoints.json")
    if [row["step"] for row in entries] != design["budgets"]:
        raise ValueError(f"Training snapshots do not match the frozen budget: {train}.")
    snapshots = {}
    expected_scenes = [(row["scene_id"], row["seed"], row["difficulty"]) for row in design["evaluation_catalogs"][split]]
    for checkpoint in entries:
        step = checkpoint["step"]
        result = read_json(evaluation / f"step-{step}.json")
        observed = [(row["scene_id"], row["scene_seed"], row["difficulty"]) for row in result["episodes"]]
        updates = max(0, step - design["online"]["sac"]["learning_starts"])
        if (observed != expected_scenes or checkpoint["sac_updates"] != updates
                or any(result[key] != value for key, value in checkpoint.items())
                or checkpoint["sha256"] != file_hash(train / checkpoint["checkpoint"])):
            raise ValueError(f"Checkpoint, decisions or evaluation catalog changed at {evaluation}, step {step}.")
        snapshots[step] = result
    return spec, snapshots


def report_warm_start(suite, split):
    protocol_path = suite / "protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = file_hash(protocol_path)
    design = protocol["config"]
    seeds = design["student_seeds"]
    runs = {
        arm: {seed: _load_run(suite, arm, seed, split, protocol, protocol_hash) for seed in seeds}
        for arm in ARMS
    }
    sizes = {run[0]["actor_parameters"] for by_seed in runs.values() for run in by_seed.values()}
    if len(sizes) != 1:
        raise ValueError("Actor deployment sizes differ among the matched groups.")
    count = len(design["evaluation_catalogs"][split])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices_seed = rng.integers(len(seeds), size=(BOOTSTRAP_REPETITIONS, len(seeds), 1))
    indices_scene = rng.integers(count, size=(BOOTSTRAP_REPETITIONS, 1, count))
    threshold = design["threshold"]
    first_observed = dict.fromkeys(ARMS)
    points = []
    for step in design["budgets"]:
        episodes = {arm: {seed: runs[arm][seed][1][step]["episodes"] for seed in seeds} for arm in ARMS}
        groups = {}
        indicators = {}
        for arm, by_seed in episodes.items():
            episode_rows = [row for rows in by_seed.values() for row in rows]
            aggregated = summarize(episode_rows)
            indicators[arm] = _hard_indicators(by_seed, seeds)
            groups[arm] = {
                **aggregated, "successful_episodes": int(indicators[arm]["success_rate"].sum()),
                "collision_episodes": int(indicators[arm]["collision_rate"].sum()),
                "evaluation_decisions": sum(row["decisions"] for row in episode_rows),
                "per_seed": {seed: summarize(rows) for seed, rows in by_seed.items()},
            }
            if (first_observed[arm] is None and aggregated["success_rate"] >= threshold["success_rate_at_least"]
                    and aggregated["collision_rate"] <= threshold["collision_rate_at_most"]):
                first_observed[arm] = step
        paired = {}
        for candidate, control in COMPARISONS:
            comparison = {}
            for metric in ("success_rate", "collision_rate", "timeout_rate", "missed_merge_rate"):
                diff = indicators[candidate][metric].astype(np.int8) - indicators[control][metric]
                interval = np.quantile(diff[indices_seed, indices_scene].mean(axis=(1, 2)), (0.025, 0.975))
                comparison[metric] = {
                    "difference_pp": float(diff.mean() * 100),
                    "candidate_only_episodes": int((diff > 0).sum()),
                    "control_only_episodes": int((diff < 0).sum()),
                    "exploratory_95_interval_pp": (interval * 100).tolist(),
                }
            paired[f"{candidate}_minus_{control}"] = comparison
        points.append({"online_decisions_per_seed": step, "groups": groups, "paired": paired})
    return {
        "protocol_sha256": protocol_hash, "training_git_sha": protocol["git_sha"], "split": split,
        "stress_seed": design["stress_seeds"][split], "student_seeds": seeds, "unique_scenarios": count,
        "actor_parameters": sizes.pop(), "threshold": threshold, "first_observed_budget_meeting_target": first_observed,
        "points": points, "archived_costs": design["archived_costs"],
        "training": {arm: {seed: by_seed[seed][0] for seed in seeds} for arm, by_seed in runs.items()},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "axes": ["student_seed", "scene"]},
    }
