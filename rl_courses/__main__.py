"""Command-line entry points for training and evaluating the stage-one policies."""

import argparse
import hashlib
from pathlib import Path

from stable_baselines3 import SAC

from .catalog import make_catalog, serialize_catalog
from .environment import EnvConfig, MergeEnv
from .evaluation import evaluate
from .records import manifest, new_output, read_json, write_json
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
    for command in (train_parser, eval_parser):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    args.command(args)


if __name__ == "__main__":
    main()
