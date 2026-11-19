"""Command-line entry points for training and evaluating the stage-one policies."""

import argparse
import hashlib
import math
from pathlib import Path

from stable_baselines3 import SAC

from .catalog import make_catalog, serialize_catalog
from .diagnostics import comparison_inputs, run_diagnostics
from .environment import EnvConfig, MergeEnv
from .evaluation import evaluate
from .records import manifest, new_output, read_json, write_json
from .reporting import summarize_suite
from .sampling import METHODS
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
    meta = manifest(config, args.seed, config["sac"]["device"], args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", serialize_catalog(catalog))
    run_with_status(output, lambda: train(config, catalog, args.seed, output))


def evaluate_command(args):
    training = read_json(args.run / "manifest.json")
    config = training["config"]
    scenes = [Scene(**row) for row in read_json(args.run / "scenes.json")[args.split]]
    checkpoint = (args.run / f"{args.checkpoint}.zip").resolve(strict=True)
    meta = manifest(config, training["seed"], args.device, args.allow_dirty)
    meta.update({
        "training_git_sha": training["git_sha"], "split": args.split,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_scenes": [scene.scene_id for scene in scenes],
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
    result = summarize_suite(args.suite, args.split, args.success_threshold, args.collision_threshold)
    config = {"split": args.split, "success_threshold": args.success_threshold,
              "collision_threshold": args.collision_threshold, "suite": str(args.suite.resolve())}
    meta = manifest(config, result["seeds"], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "report.json", result)
    print(output / "report.json")


def diagnose_command(args):
    speeds = args.reference_speed or [12, 19, 26]
    if len(set(speeds)) != len(speeds) or not all(math.isfinite(speed) and 0 < speed <= 31 for speed in speeds):
        raise ValueError("Reference speeds must be distinct and in (0, 31] m/s.")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--method", choices=METHODS)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--device")
    train_parser.set_defaults(command=train_command)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--run", type=Path, required=True)
    eval_parser.add_argument("--split", choices=("validation", "test", "stress"), default="test")
    eval_parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.set_defaults(command=evaluate_command)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--suite", type=Path, required=True)
    report_parser.add_argument("--split", choices=("validation", "test", "stress"), default="validation")
    report_parser.add_argument("--success-threshold", type=float, default=0.90)
    report_parser.add_argument("--collision-threshold", type=float, default=0.05)
    report_parser.set_defaults(command=report_command)
    diagnose_parser = commands.add_parser("diagnose")
    diagnose_parser.add_argument("--run", type=Path, action="append", required=True)
    diagnose_parser.add_argument("--split", choices=("validation", "test", "stress"), default="stress")
    diagnose_parser.add_argument("--scene", action="append")
    diagnose_parser.add_argument("--reference-speed", action="append", type=float)
    diagnose_parser.set_defaults(command=diagnose_command)
    for command in (train_parser, eval_parser, report_parser, diagnose_parser):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    args.command(args)


if __name__ == "__main__":
    main()
