"""Freeze and run the three-arm, 120k-decision small SAC curriculum comparison."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl_courses.catalog import make_catalog, make_stress_holdout, serialize_catalog
from rl_courses.evaluation import is_pre_merge_failure, summarize
from rl_courses.opportunities import opportunity_groups
from rl_courses.records import ROOT, manifest, new_output, read_json, write_json
from rl_courses.scenarios import DEMAND, Scene


METHODS = ("uniform", "merge-failure", "merge-reference-weighted")
PRIMARY = ("merge-reference-weighted", "merge-failure")
STRESS_SEED = 20261208
STRESS_COUNT = 96


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def completed(directory):
    path = directory / "status.json"
    return path.is_file() and read_json(path)["state"] == "completed"


def freeze(args):
    config = read_json(args.config)
    config["sac"]["policy_kwargs"]["net_arch"] = [64, 64]
    budget = len(DEMAND) * config["budget"]["block_steps"] * config["budget"]["blocks_per_difficulty"]
    if budget != 120000:
        raise ValueError("Use the stated 120,000-decision budget.")
    catalog = serialize_catalog(make_catalog(**config["scenarios"]))
    profiles = args.profiles.resolve(strict=True)
    opportunity_groups(read_json(profiles), [Scene(**row) for row in catalog["train"]], config["environment"])
    scenes = serialize_catalog({
        "stress": make_stress_holdout(catalog, args.stress_seed, args.stress_count)
    })["stress"]
    seeds = sorted(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct student seeds.")
    design = {
        "student_config": config, "scene_catalog": catalog, "stress_catalog": scenes,
        "methods": list(METHODS), "training_seeds": seeds, "checkpoint_kind": "best",
        "online_decisions_per_seed": budget, "stress_seed": args.stress_seed, "stress_count": args.stress_count,
        "profiles": str(profiles), "profiles_sha256": digest(profiles), "primary_comparison": list(PRIMARY),
        "archived_uniform": {}, "bootstrap_seed": 104731, "bootstrap_repetitions": 10000,
    }
    protocol = manifest(design, seeds, config["sac"]["device"], False)
    for seed in seeds:
        train = (args.uniform_root / f"seed-{seed}" / "train").resolve(strict=True)
        if not completed(train):
            raise ValueError(f"Archived uniform training is incomplete: {train}.")
        metadata = read_json(train / "manifest.json")
        archived = metadata["config"].copy()
        if (archived.pop("method") != "scratch" or metadata["scratch_scene_sampling"] != "uniform"
                or not archived.pop("distillation") or archived != config
                or read_json(train / "scenes.json") != catalog or metadata["seed"] != seed
                or metadata["versions"] != protocol["versions"] or metadata["git_status"]
                or read_json(train / "training_summary.json")["steps"] != budget):
            raise ValueError(f"Archived uniform controls are incompatible: {train}.")
        design["archived_uniform"][str(seed)] = {
            "directory": str(train), "training_git_sha": metadata["git_sha"],
            "manifest_sha256": digest(train / "manifest.json"), "best_sha256": digest(train / "best.zip"),
        }
    suite = new_output(args.suite)
    write_json(suite / "config.json", config)
    write_json(suite / "protocol.json", protocol)
    print(suite / "protocol.json", flush=True)


def target_dir(parent, stage):
    previous = [path for path in sorted(parent.glob(f"{stage}-*")) if path.is_dir()]
    completed_runs = [path for path in previous if completed(path)]
    if len(completed_runs) == 1:
        return completed_runs[0]
    if completed_runs or any(not (path / "status.json").is_file()
                             or read_json(path / "status.json")["state"] == "running" for path in previous):
        raise ValueError(f"Resolve incomplete or duplicate attempts in {parent} before restarting.")
    return parent / f"{stage}-{len(previous) + 1:03d}"


def launch(args, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-B", "-m", "rl_courses", *map(str, args), "--output", str(destination)]
    with destination.with_suffix(".log").open("x", encoding="utf-8") as log:
        subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def job(suite, protocol, method, seed):
    design = protocol["config"]
    directory = suite / method / f"seed-{seed}"
    if method == "uniform":
        archived = design["archived_uniform"][str(seed)]
        train = Path(archived["directory"])
        if (not completed(train) or digest(train / "manifest.json") != archived["manifest_sha256"]
                or digest(train / "best.zip") != archived["best_sha256"]):
            raise ValueError(f"The frozen uniform checkpoint changed: {train}.")
    else:
        train = target_dir(directory, "train")
        if not completed(train):
            args = ["train", "--config", suite / "config.json", "--method", method, "--seed", seed]
            if method == "merge-reference-weighted":
                args += ["--profiles", design["profiles"]]
            launch(args, train)
        training = read_json(train / "manifest.json")
        if (training["git_sha"] != protocol["git_sha"] or training["versions"] != protocol["versions"]
                or training["seed"] != seed or training["config"] != {**design["student_config"], "method": method}
                or (method == "merge-reference-weighted"
                    and training["opportunity_profiles"]["sha256"] != design["profiles_sha256"])):
            raise ValueError(f"New training differs from the frozen protocol: {train}.")
    if read_json(train / "training_summary.json")["steps"] != design["online_decisions_per_seed"]:
        raise ValueError(f"The run did not consume the exact 120,000-decision budget: {train}.")
    evaluation = target_dir(directory, "stress")
    if not completed(evaluation):
        launch(["evaluate", "--run", train, "--split", "stress", "--checkpoint", design["checkpoint_kind"],
                "--stress-seed", design["stress_seed"], "--stress-count", design["stress_count"]], evaluation)
    meta = read_json(evaluation / "manifest.json")
    rows = read_json(evaluation / "episodes.json")
    expected = [(row["scene_id"], row["seed"], row["difficulty"]) for row in design["stress_catalog"]]
    if (meta["evaluation_catalog"] != design["stress_catalog"] or meta["git_sha"] != protocol["git_sha"]
            or meta["seed"] != seed or meta["checkpoint_kind"] != design["checkpoint_kind"]
            or meta["checkpoint_sha256"] != digest(train / "best.zip")
            or [(row["scene_id"], row["scene_seed"], row["difficulty"]) for row in rows] != expected):
        raise ValueError(f"Evaluation differs from the frozen paired protocol: {evaluation}.")
    return method, seed, rows, {
        "training": str(train), "evaluation": str(evaluation), "training_git_sha": meta["training_git_sha"],
        "selected_step": read_json(train / "best.json")["step"], "checkpoint_sha256": meta["checkpoint_sha256"],
    }


def paired_report(design, episodes):
    seeds = design["training_seeds"]
    scenes = design["stress_count"]
    repeats = design["bootstrap_repetitions"]
    rng = np.random.default_rng(design["bootstrap_seed"])
    sample_seed = rng.integers(len(seeds), size=(repeats, len(seeds), 1))
    sample_scene = rng.integers(scenes, size=(repeats, 1, scenes))
    rules = {
        "success_rate": lambda row: row["outcome"] == "success",
        "pre_merge_failure_rate": is_pre_merge_failure,
        "collision_rate": lambda row: row["outcome"] == "collision",
    }
    metrics = {method: summarize([row for seed in seeds for row in episodes[method][seed]]) for method in METHODS}
    differences = {}
    for candidate, baseline in (PRIMARY, ("merge-failure", "uniform"), ("merge-reference-weighted", "uniform")):
        comparison = {}
        for name, condition in rules.items():
            diff = np.array([[int(condition(left)) - int(condition(right))
                              for left, right in zip(episodes[candidate][seed], episodes[baseline][seed], strict=True)]
                             for seed in seeds])
            interval = np.quantile(diff[sample_seed, sample_scene].mean(axis=(1, 2)), (0.025, 0.975))
            comparison[name] = {
                "candidate_minus_baseline_pp": float(100 * diff.mean()),
                "exploratory_95_interval_pp": (100 * interval).tolist(),
                "per_seed_difference_pp": (100 * diff.mean(axis=1)).tolist(),
                "candidate_only": int((diff > 0).sum()), "baseline_only": int((diff < 0).sum()),
            }
        differences[f"{candidate}_minus_{baseline}"] = comparison
    return {"metrics": metrics, "paired": differences}


def run(args):
    suite = args.suite.resolve(strict=True)
    path = suite / "protocol.json"
    protocol = read_json(path)
    design = protocol["config"]
    actual = manifest({"suite": str(suite), "workers": args.workers}, design["training_seeds"],
                      design["student_config"]["sac"]["device"], False)
    if any(actual[key] != protocol[key] for key in ("git_sha", "versions")):
        raise ValueError("Use the same clean commit and environment that froze this suite.")
    if (read_json(suite / "config.json") != design["student_config"]
            or digest(Path(design["profiles"])) != design["profiles_sha256"]):
        raise ValueError("The frozen configuration or profiles changed.")
    if not 1 <= args.workers <= len(METHODS) * len(design["training_seeds"]):
        raise ValueError("Worker count must fit the number of jobs.")
    episodes = {method: {} for method in METHODS}
    sources = {method: {} for method in METHODS}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(job, suite, protocol, method, seed)
            for method in METHODS for seed in design["training_seeds"]
        ]
        for pending in as_completed(jobs):
            method, seed, rows, source = pending.result()
            episodes[method][seed] = rows
            sources[method][seed] = source
            print(f"{method} seed {seed}: complete.", flush=True)
    result = paired_report(design, episodes)
    write_json(suite / "report.json", {
        **result, "runs": sources, "manifest": actual, "protocol_sha256": digest(path),
    })
    print(suite / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs" / "stage1.json")
    prepare.add_argument("--profiles", type=Path, required=True)
    prepare.add_argument("--uniform-root", type=Path, required=True)
    prepare.add_argument("--suite", type=Path, required=True)
    prepare.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    prepare.add_argument("--stress-seed", type=int, default=STRESS_SEED)
    prepare.add_argument("--stress-count", type=int, default=STRESS_COUNT)
    prepare.set_defaults(command=freeze)
    perform = commands.add_parser("run")
    perform.add_argument("--suite", type=Path, required=True)
    perform.add_argument("--workers", type=int, default=6)
    perform.set_defaults(command=run)
    args = parser.parse_args()
    args.command(args)


if __name__ == "__main__":
    main()
