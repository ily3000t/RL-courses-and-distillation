"""Command-line entry points for training and evaluating the stage-one policies."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from stable_baselines3 import SAC

from .catalog import make_catalog, make_stress_holdout, serialize_catalog
from .diagnostics import comparison_inputs, run_diagnostics
from .environment import EnvConfig, MergeEnv
from .evaluation import evaluate
from .opportunities import DEFAULT_REFERENCE_SPEEDS, build_profiles, validate_speeds
from .records import manifest, new_output, read_json, write_json
from .reporting import summarize_suite
from .sampling import COMPARISONS, METHODS, OPPORTUNITY_METHODS
from .scenarios import Scene
from .training import train


def train_command(args):
    config = read_json(args.config)
    config["method"] = args.method or config.get("method", "hard")
    if args.device:
        config["sac"]["device"] = args.device
    if not 0 <= args.seed < 2**31:
        raise ValueError("Use a nonnegative signed 31-bit seed.")
    catalog = make_catalog(**config["scenarios"])
    profiles = None
    if config["method"] in OPPORTUNITY_METHODS and args.profiles is None:
        raise ValueError("Pass --profiles with the output of the profile command.")
    if args.profiles is not None and config["method"] not in OPPORTUNITY_METHODS:
        raise ValueError("--profiles requires an opportunity-progress method.")
    meta = manifest(config, args.seed, config["sac"]["device"], args.allow_dirty)
    if args.profiles is not None:
        profile_bytes = args.profiles.read_bytes()
        profiles = json.loads(profile_bytes)
        meta["opportunity_profiles"] = {
            "path": str(args.profiles.resolve()), "sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "profile_source_sha256": profiles["profile_source_sha256"],
            "reference_speeds_mps": profiles["reference_speeds_mps"],
            "probe_episodes": profiles["probe_episodes"],
            "probe_simulated_seconds_after_spawn": profiles["probe_simulated_seconds_after_spawn"],
        }
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", serialize_catalog(catalog))
    run_with_status(output, lambda: train(config, catalog, args.seed, output, profiles))


def evaluate_command(args):
    training = read_json(args.run / "manifest.json")
    config = training["config"]
    catalog = read_json(args.run / "scenes.json")
    if (args.stress_seed is not None or args.stress_count is not None) and args.split != "stress":
        raise ValueError("A new stress holdout requires --split stress.")
    if args.stress_count is not None and args.stress_seed is None:
        raise ValueError("--stress-count requires --stress-seed.")
    if args.stress_seed is None:
        scenes = [Scene(**row) for row in catalog[args.split]]
    else:
        count = args.stress_count if args.stress_count is not None else len(catalog["stress"])
        scenes = make_stress_holdout(catalog, args.stress_seed, count)
    checkpoint = (args.run / f"{args.checkpoint}.zip").resolve(strict=True)
    meta = manifest(config, training["seed"], args.device, args.allow_dirty)
    meta.update({
        "training_git_sha": training["git_sha"], "split": args.split,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_scenes": [scene.scene_id for scene in scenes],
        "evaluation_catalog": [asdict(scene) for scene in scenes],
        "stress_holdout_seed": args.stress_seed,
        "checkpoint_kind": args.checkpoint,
    })
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)

    def predict():
        model = SAC.load(checkpoint, device=args.device)
        env = MergeEnv(output, EnvConfig(**config["environment"]))
        try:
            summary, episodes = evaluate(model, env, scenes)
            write_json(output / "summary.json", summary)
            write_json(output / "episodes.json", episodes)
            print(summary)
        finally:
            env.close()

    run_with_status(output, predict)


def run_with_status(output, operation):
    path = output / "status.json"
    write_json(path, {"state": "running"})
    try:
        operation()
    except BaseException as error:
        write_json(path, {"state": "failed", "error": str(error), "type": type(error).__name__})
        raise
    write_json(path, {"state": "completed"})


def report_command(args):
    result = summarize_suite(args.suite, args.split, args.success_threshold,
                             args.collision_threshold, args.comparison)
    config = {"split": args.split, "success_threshold": args.success_threshold,
              "collision_threshold": args.collision_threshold, "suite": str(args.suite.resolve()),
              "comparison": args.comparison}
    meta = manifest(config, result["seeds"], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "report.json", result)
    print(output / "report.json")


def diagnose_command(args):
    speeds = args.reference_speed or DEFAULT_REFERENCE_SPEEDS
    validate_speeds(speeds)
    inputs, scenes = comparison_inputs(args.run, args.split, args.scene)
    config = {
        "split": args.split, "scene_ids": [scene.scene_id for scene in scenes],
        "reference_speeds_mps": speeds, "environment": inputs[0]["environment"],
        "checkpoints": [{key: value for key, value in row.items() if key != "scenes"}
                        for row in inputs],
    }
    meta = manifest(config, [row["seed"] for row in inputs], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    run_with_status(output, lambda: run_diagnostics(inputs, scenes, speeds, output))
    print(output / "episodes.json")


def profile_command(args):
    speeds = args.reference_speed or DEFAULT_REFERENCE_SPEEDS
    validate_speeds(speeds)
    config = read_json(args.config)
    scenes = make_catalog(**config["scenarios"])["train"]
    metadata = manifest(
        {"environment": config["environment"], "scenarios": config["scenarios"],
         "reference_speeds_mps": speeds},
        config["scenarios"]["seed"], "cpu", args.allow_dirty,
    )
    output = new_output(args.output)
    write_json(output / "manifest.json", metadata)
    run_with_status(output, lambda: build_profiles(scenes, config["environment"], speeds, output))
    print(output / "opportunities.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--method", choices=METHODS)
    train_parser.add_argument("--profiles", type=Path)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--device")
    train_parser.set_defaults(command=train_command)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--run", type=Path, required=True)
    eval_parser.add_argument("--split", choices=("validation", "test", "stress"), default="test")
    eval_parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    eval_parser.add_argument("--stress-seed", type=int)
    eval_parser.add_argument("--stress-count", type=int)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.set_defaults(command=evaluate_command)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--suite", type=Path, required=True)
    report_parser.add_argument("--split", choices=("validation", "test", "stress"), default="validation")
    report_parser.add_argument("--comparison", choices=COMPARISONS, default="baselines")
    report_parser.add_argument("--success-threshold", type=float, default=0.90)
    report_parser.add_argument("--collision-threshold", type=float, default=0.05)
    report_parser.set_defaults(command=report_command)
    diagnose_parser = commands.add_parser("diagnose")
    diagnose_parser.add_argument("--run", type=Path, action="append", required=True)
    diagnose_parser.add_argument("--split", choices=("validation", "test", "stress"), default="stress")
    diagnose_parser.add_argument("--scene", action="append")
    diagnose_parser.add_argument("--reference-speed", action="append", type=float)
    diagnose_parser.set_defaults(command=diagnose_command)
    profile_parser = commands.add_parser("profile")
    profile_parser.add_argument("--config", type=Path, required=True)
    profile_parser.add_argument("--reference-speed", action="append", type=float)
    profile_parser.set_defaults(command=profile_command)
    for command in (train_parser, eval_parser, report_parser, diagnose_parser, profile_parser):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    args.command(args)


if __name__ == "__main__":
    main()
