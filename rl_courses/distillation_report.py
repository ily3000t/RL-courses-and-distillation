"""Report matched student and teacher closed-loop results and deployment costs."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .distillation import STUDENT_METHODS
from .evaluation import summarize
from .records import read_json
from .reporting import METRICS


def _read_completed(directory):
    if read_json(directory / "status.json")["state"] != "completed":
        raise ValueError(f"Run is incomplete: {directory}")
    return read_json(directory / "manifest.json")


def _read_evaluation(directory, split):
    meta = _read_completed(directory)
    checkpoint = Path(meta["checkpoint"])
    train = _read_completed(checkpoint.parent)
    if (meta["split"] != split or checkpoint.name != "best.zip" or meta["checkpoint_kind"] != "best"
            or sha256(checkpoint.read_bytes()).hexdigest() != meta["checkpoint_sha256"]
            or meta["training_git_sha"] != train["git_sha"] or meta["seed"] != train["seed"]
            or meta["config"] != train["config"]):
        raise ValueError(f"Evaluation provenance does not match its training run: {directory}")
    rows = read_json(directory / "episodes.json")
    expected = [(row["scene_id"], row["seed"], row["difficulty"]) for row in meta["evaluation_catalog"]]
    observed = [(row["scene_id"], row["scene_seed"], row["difficulty"]) for row in rows]
    if observed != expected:
        raise ValueError(f"Evaluation rows do not match the scenario catalog: {directory}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["difficulty"]].append(row)
    deployment = read_json(directory / "deployment.json")
    artifact = directory / "actor.pt"
    if (artifact.stat().st_size != deployment["file_bytes"]
            or sha256(artifact.read_bytes()).hexdigest() != deployment["actor_sha256"]):
        raise ValueError(f"Exported actor differs from the deployment record: {directory}")
    return meta, train, {
        "all": summarize(rows), **{difficulty: summarize(group) for difficulty, group in groups.items()}
    }, deployment, rows


def summarize_distillation(root, split):
    result, _, _, _ = _summarize_distillation(root, split)
    return result


def _summarize_distillation(root, split):
    data = root / "collection"
    collection = _read_completed(data)
    collection_summary = read_json(data / "collection_summary.json")
    teacher = collection["teacher"]
    tm, _, metrics, deployment, _ = _read_evaluation(root / "teacher" / split, split)
    if (tm["checkpoint_sha256"] != teacher["checkpoint_sha256"]
            or tm["training_git_sha"] != teacher["git_sha"] or tm["seed"] != teacher["seed"]):
        raise ValueError("The evaluated teacher differs from the collection teacher.")
    teacher_result = {"checkpoint": teacher, "evaluation": metrics, "deployment": deployment}
    eval_protocol = (tm["evaluation_catalog"], tm["versions"], tm["git_sha"], tm["stress_holdout_seed"],
                     tm["config"]["environment"], tm["observation_fields"])
    deployment_fields = ("format", "device", "cpu_model", "cpu_threads", "batch_size", "timed_calls")
    benchmark = tuple(deployment[key] for key in deployment_fields)
    catalog = read_json(data / "scenes.json")
    student_protocol = None
    students, hard_episodes = {}, {}
    for method in STUDENT_METHODS:
        runs = {}
        hard_episodes[method] = {}
        for path in sorted((root / method).glob(f"seed-*/{split}")):
            meta, train, evaluation, actor, rows = _read_evaluation(path, split)
            config = train["config"].copy()
            if config.pop("method") != method or read_json(Path(meta["checkpoint"]).parent / "scenes.json") != catalog:
                raise ValueError(f"Student method or training catalog differs: {path}")
            protocol = (config, train["versions"])
            if student_protocol is None:
                student_protocol = protocol
            if protocol != student_protocol:
                raise ValueError(f"Student settings differ across methods or seeds: {path}")
            if eval_protocol != (meta["evaluation_catalog"], meta["versions"], meta["git_sha"],
                                 meta["stress_holdout_seed"], meta["config"]["environment"],
                                 meta["observation_fields"]):
                raise ValueError(f"Student and teacher evaluation protocols differ: {path}")
            if tuple(actor[key] for key in deployment_fields) != benchmark:
                raise ValueError(f"Student and teacher actor benchmarks differ: {path}")
            if method != "scratch":
                source = train["teacher_data"]
                if (source["dataset_sha256"] != collection_summary["dataset_sha256"]
                        or source["teacher"] != teacher or source["collection_seed"] != collection["seed"]):
                    raise ValueError(f"Student was trained from a different teacher dataset: {path}")
            seed = meta["seed"]
            if seed in runs or path.parent.name != f"seed-{seed}":
                raise ValueError(f"Duplicate or misplaced seed: {path}")
            hard_episodes[method][seed] = [row for row in rows if row["difficulty"] == "hard"]
            runs[seed] = {"seed": seed, "training_git_sha": train["git_sha"],
                          "checkpoint": meta["checkpoint"], "checkpoint_sha256": meta["checkpoint_sha256"],
                          "evaluation": evaluation,
                          "deployment": actor, "training": read_json(Path(meta["checkpoint"]).parent / "training_summary.json")}
        students[method] = runs
    seeds = sorted(students["scratch"])
    if not seeds or any(set(runs) != set(seeds) for runs in students.values()):
        raise ValueError("All three student methods must have the same nonempty seed set.")
    if len({run["deployment"]["actor_parameters"] for runs in students.values() for run in runs.values()}) != 1:
        raise ValueError("Student deployment actor sizes differ.")

    aggregates = {}
    for method, runs in students.items():
        hard_metrics = {}
        for metric in METRICS:
            values = [runs[seed]["evaluation"]["hard"][metric] for seed in seeds]
            hard_metrics[metric] = {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else None}
        teacher_success = metrics["hard"]["success_rate"]
        success = hard_metrics["success_rate"]["mean"]
        aggregates[method] = {
            "hard_evaluation": hard_metrics, "teacher_hard_success_delta": success - teacher_success,
            "teacher_hard_success_retention": success / teacher_success if teacher_success else None,
            "teacher_to_student_parameter_ratio": deployment["actor_parameters"] / runs[seeds[0]]["deployment"]["actor_parameters"],
            "runs": [runs[seed] for seed in seeds],
        }
    result = {
        "split": split, "stress_holdout_seed": tm["stress_holdout_seed"],
        "evaluation_git_sha": tm["git_sha"], "evaluation_catalog": tm["evaluation_catalog"],
        "student_seeds": seeds, "teacher": teacher_result, "collection": collection_summary,
        "student_config": student_protocol[0], "students": aggregates,
        "soft_minus_bc_hard": {
            metric: [students["soft"][seed]["evaluation"]["hard"][metric]
                     - students["bc"][seed]["evaluation"]["hard"][metric] for seed in seeds]
            for metric in METRICS
        },
    }
    return result, student_protocol, eval_protocol, hard_episodes


def _hard_indicators(episodes, seeds):
    outcomes = np.array([[row["outcome"] for row in episodes[seed]] for seed in seeds])
    merged = np.array([[row["merged"] for row in episodes[seed]] for seed in seeds])
    timeout = outcomes == "timeout"
    before_merge = timeout & ~merged
    missed = outcomes == "missed_merge"
    return {
        "success_rate": outcomes == "success", "collision_rate": outcomes == "collision",
        "timeout_rate": timeout, "pre_merge_timeout_rate": before_merge,
        "missed_merge_rate": missed, "pre_merge_failure_rate": before_merge | missed,
    }


def summarize_collection_ablation(truncated_root, complete_root, split, bootstrap_seed):
    baseline_data = truncated_root / "collection"
    candidate_data = complete_root / "collection"
    collection_protocol = None
    for mode, data in (("truncate", baseline_data), ("full-episode", candidate_data)):
        meta = _read_completed(data)
        summary = read_json(data / "collection_summary.json")
        config = meta["config"].copy()
        catalog = read_json(data / "scenes.json")
        if config.pop("collection_mode", "truncate") != mode or summary.get("collection_mode", "truncate") != mode:
            raise ValueError(f"Expected {mode} collection in {data}.")
        protocol = (config, catalog, meta["seed"], meta["teacher"], meta["versions"], meta["observation_fields"])
        if collection_protocol is None:
            collection_protocol = protocol
        if protocol != collection_protocol:
            raise ValueError("Both collection modes must share the teacher, seed, environment, catalog and quota.")
        quota = config["steps_per_scene"] * len(catalog["train"])
        cost = summary.get("rollout_decision_steps", summary["decision_steps"])
        if (summary["decision_steps"] != quota or cost < quota
                or (mode == "truncate" and cost != quota)
                or (mode == "full-episode" and (summary["outcomes"].get("collection_cut", 0)
                                               or "rollout_decision_steps" not in summary))
                or sha256((data / "trajectories.npz").read_bytes()).hexdigest() != summary["dataset_sha256"]):
            raise ValueError(f"Collection data, retained quota or raw interaction budget is invalid in {data}.")

    baseline, base_train, base_eval, base_episodes = _summarize_distillation(truncated_root, split)
    candidate, new_train, new_eval, new_episodes = _summarize_distillation(complete_root, split)
    seeds = baseline["student_seeds"]
    if base_train != new_train or base_eval != new_eval or seeds != candidate["student_seeds"]:
        raise ValueError("Both suites must share the student settings, seeds and evaluation protocol.")
    for left, right in zip(baseline["students"]["scratch"]["runs"], candidate["students"]["scratch"]["runs"], strict=True):
        if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
            raise ValueError("Both suites must use the same from-scratch checkpoints.")
    if baseline["teacher"]["evaluation"] != candidate["teacher"]["evaluation"] or base_episodes["scratch"] != new_episodes["scratch"]:
        raise ValueError("Teacher and from-scratch evaluations must reproduce across both suites.")

    # Resample seed rows and scenario columns while keeping the two policies paired.
    repetitions = 10000
    rng = np.random.default_rng(bootstrap_seed)
    scene_count = len(base_episodes["bc"][seeds[0]])
    seed_indices = rng.integers(len(seeds), size=(repetitions, len(seeds), 1))
    scene_indices = rng.integers(scene_count, size=(repetitions, 1, scene_count))
    comparison = {}
    for method in ("bc", "soft"):
        left = _hard_indicators(base_episodes[method], seeds)
        right = _hard_indicators(new_episodes[method], seeds)
        comparison[method] = {}
        for metric in METRICS:
            difference = right[metric].astype(np.int8) - left[metric]
            samples = difference[seed_indices, scene_indices].mean(axis=(1, 2))
            comparison[method][metric] = {
                "truncate": float(left[metric].mean()), "full_episode": float(right[metric].mean()),
                "full_minus_truncate_pp": float(100 * difference.mean()),
                "per_seed_difference_pp": (100 * difference.mean(axis=1)).tolist(),
                "exploratory_95_interval_pp": (100 * np.quantile(samples, (0.025, 0.975))).tolist(),
            }
    return {
        "split": split, "evaluation_git_sha": baseline["evaluation_git_sha"],
        "stress_holdout_seed": baseline["stress_holdout_seed"],
        "evaluation_catalog": baseline["evaluation_catalog"], "hard_scene_count": scene_count, "student_seeds": seeds,
        "teacher": baseline["teacher"]["checkpoint"], "student_config": baseline["student_config"],
        "collection_seed": collection_protocol[2], "collection_config": collection_protocol[0],
        "collections": {"truncate": baseline["collection"], "full_episode": candidate["collection"]},
        "bootstrap": {"seed": bootstrap_seed, "repetitions": repetitions, "paired_axes": ["student_seed", "scene"]},
        "hard_full_minus_truncate": comparison,
    }
