"""Train, evaluate, and distill SAC policies for SUMO ramp merging."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from stable_baselines3 import SAC

from .actor_export import export_actor
from .catalog import make_catalog, make_stress_holdout, serialize_catalog
from .correction_replay import BRANCHES, run_correction_replay
from .correction_targets import correction_replay_targets, write_correction_targets
from .diagnostics import comparison_inputs, run_diagnostics
from .distillation import CORRECTION_SUPERVISION, STUDENT_METHODS, student_config, train_offline
from .distillation_report import summarize_collection_ablation, summarize_distillation
from .environment import OBSERVATION_FIELDS, EnvConfig, MergeEnv
from .evaluation import evaluate
from .opportunities import DEFAULT_REFERENCE_SPEEDS, build_profiles, validate_speeds
from .phase_sampling import WINDOW_METERS, sample_merge_phases
from .pre_wait_replay import ANCHORS, SLOWING_SPEED_MPS, run_pre_wait_replay
from .records import manifest, new_output, read_json, write_json
from .reporting import summarize_suite
from .sampling import COMPARISONS, METHODS, PROFILE_METHODS
from .scenarios import Scene
from .student_relabeling import relabel_student_rollouts
from .teacher_data import COLLECTION_MODES, collect_trajectories
from .training import train
from .waiting_diagnostics import WAIT_SECONDS, WAIT_SPEED_MPS, run_waiting_diagnostics, waiting_replay_inputs


def train_command(args):
    config = read_json(args.config)
    config["method"] = args.method or config.get("method", "hard")
    if args.device:
        config["sac"]["device"] = args.device
    if not 0 <= args.seed < 2**31:
        raise ValueError("Use a nonnegative signed 31-bit seed.")
    catalog = make_catalog(**config["scenarios"])
    profiles = None
    if config["method"] in PROFILE_METHODS and args.profiles is None:
        raise ValueError(f"{config['method']} requires --profiles from the profile command.")
    if args.profiles is not None and config["method"] not in PROFILE_METHODS:
        raise ValueError(f"{config['method']} does not use --profiles.")
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
        if args.benchmark_actor:
            export_actor(model, output)

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


def _hard_replay_inputs(args):
    runs = (args.teacher_run, args.student_run)
    for run in runs:
        if read_json(run / "status.json")["state"] != "completed":
            raise ValueError(f"Use a completed policy run: {run}")
        if read_json(run / "manifest.json")["observation_fields"] != list(OBSERVATION_FIELDS):
            raise ValueError(f"Observation fields differ from the current SUMO environment: {run}")
    inputs, selected = comparison_inputs(runs, "train", args.scene)
    scenes = [scene for scene in selected if scene.difficulty == "hard"]
    if not scenes or (args.scene and len(scenes) != len(selected)):
        raise ValueError("Correction replay uses Hard training scenes only.")
    if not 1 <= args.intervention_steps <= inputs[0]["environment"]["max_decisions"]:
        raise ValueError("Intervention steps must be positive and fit the original episode horizon.")
    return inputs, scenes


def replay_corrections_command(args):
    inputs, scenes = _hard_replay_inputs(args)
    config = {
        "split": "train", "evaluation_catalog": [asdict(scene) for scene in scenes],
        "environment": inputs[0]["environment"], "approach_window_m": WINDOW_METERS,
        "intervention_steps": args.intervention_steps, "stop_on_merge": True,
        "continuation_policy": "student", "policy_actions": "deterministic_sac",
        "branch_teacher_channels": BRANCHES,
        **{name: {key: item for key, item in row.items() if key not in ("scenes", "environment")}
           for name, row in zip(("teacher", "student"), inputs, strict=True)},
    }
    meta = manifest(config, [row["seed"] for row in inputs], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    run_with_status(output, lambda: run_correction_replay(inputs, scenes, args.intervention_steps, output))
    print(output / "summary.json")


def replay_pre_wait_command(args):
    inputs, scenes = _hard_replay_inputs(args)
    config = {
        "split": "train", "evaluation_catalog": [asdict(scene) for scene in scenes],
        "environment": inputs[0]["environment"], "anchors": list(ANCHORS),
        "slowing_speed_mps": SLOWING_SPEED_MPS, "wait_speed_mps": WAIT_SPEED_MPS, "wait_seconds": WAIT_SECONDS,
        "intervention_steps": args.intervention_steps, "teacher_channels": BRANCHES["acceleration"],
        "stop_on_merge": True, "continuation_policy": "student", "policy_actions": "deterministic_sac",
        **{name: {key: item for key, item in row.items() if key not in ("scenes", "environment")}
           for name, row in zip(("teacher", "student"), inputs, strict=True)},
    }
    meta = manifest(config, [row["seed"] for row in inputs], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    run_with_status(output, lambda: run_pre_wait_replay(inputs, scenes, args.intervention_steps, output))
    print(output / "summary.json")


def diagnose_waiting_command(args):
    if args.limit_per_transition is not None and args.limit_per_transition < 1:
        raise ValueError("Use a positive case limit.")
    config = waiting_replay_inputs(args.suite.resolve(), args.limit_per_transition)
    if not 1 <= args.intervention_steps <= config["environment"]["max_decisions"]:
        raise ValueError("Intervention steps must be positive and fit the original episode horizon.")
    config.update(intervention_steps=args.intervention_steps, limit_per_transition=args.limit_per_transition)
    meta = manifest(config, sorted(config["students"]), "cpu", args.allow_dirty)
    if meta["versions"] != config["source_evaluation_versions"]:
        raise ValueError("Replay requires the same runtime versions as the source evaluation.")
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    run_with_status(output, lambda: run_waiting_diagnostics(config, args.intervention_steps, output))
    print(output / "summary.json")


def build_correction_targets_command(args):
    source = args.collection.resolve()
    collection = read_json(source / "manifest.json")
    catalog = read_json(source / "scenes.json")
    teacher_summary = read_json(source / "collection_summary.json")
    dataset_hash = hashlib.sha256((source / "trajectories.npz").read_bytes()).hexdigest()
    if (read_json(source / "status.json")["state"] != "completed"
            or collection["config"]["split"] != "train"
            or collection["observation_fields"] != list(OBSERVATION_FIELDS)
            or dataset_hash != teacher_summary["dataset_sha256"]):
        raise ValueError("Use the completed teacher dataset used by the replay pilots.")
    samples, sources = correction_replay_targets(args.replay, collection, dataset_hash, catalog["train"])
    settings = {
        "phase": "approach_within_50m", "selected_branch": "acceleration",
        "environment": collection["config"]["environment"], "base_dataset_sha256": dataset_hash,
        "teacher_checkpoint_sha256": collection["teacher"]["checkpoint_sha256"],
        "replays": sources, "base_collection": str(source),
    }
    meta = manifest(settings, sorted({seed for seed, _, _ in samples}), "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", catalog)
    run_with_status(output, lambda: write_correction_targets(output, samples, meta["observation_fields"]))
    print(output / "summary.json")


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


def collect_teacher_command(args):
    if args.steps_per_scene < 1 or not 0 <= args.seed < 2**31:
        raise ValueError("Use positive steps per scene and a nonnegative signed 31-bit seed.")
    if args.save_rollout_pool and args.collection_mode != "full-episode":
        raise ValueError("--save-rollout-pool requires --collection-mode full-episode.")
    if read_json(args.run / "status.json")["state"] != "completed":
        raise ValueError("The teacher training run must be complete.")
    training = read_json(args.run / "manifest.json")
    catalog = read_json(args.run / "scenes.json")
    scenes = [Scene(**row) for row in catalog["train"]]
    checkpoint = (args.run / "best.zip").resolve(strict=True)
    environment = training["config"]["environment"]
    config = {"environment": environment, "steps_per_scene": args.steps_per_scene,
              "rollout_policy": "stochastic_sac", "split": "train", "collection_mode": args.collection_mode}
    meta = manifest(config, args.seed, args.device, args.allow_dirty)
    if training["observation_fields"] != meta["observation_fields"]:
        raise ValueError("Teacher observation fields differ from the current environment.")
    meta["teacher"] = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "git_sha": training["git_sha"], "seed": training["seed"], "method": training["config"]["method"],
        "training": read_json(args.run / "training_summary.json"),
        "opportunity_profiles": training.get("opportunity_profiles"),
    }
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", catalog)

    def collect():
        model = SAC.load(checkpoint, device=args.device)
        env = MergeEnv(output, EnvConfig(**environment))
        try:
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError("Teacher spaces differ from the collection environment.")
            collect_trajectories(model, env, scenes, args.seed, args.steps_per_scene, output,
                                 args.collection_mode, save_rollout_pool=args.save_rollout_pool)
        finally:
            env.close()

    run_with_status(output, collect)


def sample_teacher_command(args):
    source = args.collection.resolve()
    if read_json(source / "status.json")["state"] != "completed":
        raise ValueError("The uniform source collection must be complete.")
    collection = read_json(source / "manifest.json")
    config = collection["config"]
    if (config["collection_mode"] != "full-episode" or config.get("sampling_mode", "uniform") != "uniform"
            or config["split"] != "train" or collection["observation_fields"] != list(OBSERVATION_FIELDS)):
        raise ValueError("Sampling requires a uniform, full-episode training collection with matching observation fields.")
    quota, seed = config["steps_per_scene"], collection["seed"]
    catalog = read_json(source / "scenes.json")
    settings = {**config, "sampling_mode": "merge-phase", "priority_window_m": WINDOW_METERS,
                "minimum_per_priority_phase": quota // 4}
    meta = manifest(settings, seed, "cpu", args.allow_dirty)
    meta["teacher"] = collection["teacher"]
    meta["source_collection"] = {"path": str(source), "git_sha": collection["git_sha"]}
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", catalog)
    run_with_status(output, lambda: sample_merge_phases(source, output, catalog, seed, quota))


def relabel_student_command(args):
    source, behavior = args.collection.resolve(), args.student_run.resolve()
    for path in (source, behavior):
        if read_json(path / "status.json")["state"] != "completed":
            raise ValueError(f"Use a completed source run: {path}")
    collection = read_json(source / "manifest.json")
    pilot = read_json(behavior / "manifest.json")
    config = collection["config"]
    catalog = read_json(source / "scenes.json")
    quota = config["steps_per_scene"]
    summary = read_json(source / "collection_summary.json")
    dataset = source / "trajectories.npz"
    teacher_data = pilot.get("teacher_data", {})
    if (config.get("collection_mode") != "full-episode" or config.get("sampling_mode", "uniform") != "uniform"
            or config["split"] != "train" or quota < 4 or config["environment"] != pilot["config"]["environment"]
            or collection["observation_fields"] != list(OBSERVATION_FIELDS)
            or collection["observation_fields"] != pilot["observation_fields"]
            or catalog != read_json(behavior / "scenes.json")
            or summary["decision_steps"] != quota * len(catalog["train"])
            or teacher_data.get("teacher") != collection["teacher"]
            or teacher_data.get("dataset_sha256") != summary["dataset_sha256"]
            or hashlib.sha256(dataset.read_bytes()).hexdigest() != summary["dataset_sha256"]):
        raise ValueError("Use a student trained on the same uniform full-episode data and matching environment.")
    teacher_checkpoint = Path(collection["teacher"]["checkpoint"])
    if hashlib.sha256(teacher_checkpoint.read_bytes()).hexdigest() != collection["teacher"]["checkpoint_sha256"]:
        raise ValueError("The original teacher checkpoint has changed.")
    student_checkpoint = behavior / "best.zip"
    seed = collection["seed"]
    settings = {**config, "sampling_mode": "student-relabeling", "priority_window_m": WINDOW_METERS,
                "maximum_replacements_per_priority_phase": quota // 4,
                "student_rollouts_per_scene": 1, "student_rollout_policy": "deterministic"}
    meta = manifest(settings, seed, args.device, args.allow_dirty)
    meta["teacher"] = collection["teacher"]
    meta["source_collection"] = {"path": str(source), "git_sha": collection["git_sha"]}
    meta["behavior_student"] = {
        "checkpoint": str(student_checkpoint), "checkpoint_sha256": hashlib.sha256(student_checkpoint.read_bytes()).hexdigest(),
        "git_sha": pilot["git_sha"], "seed": pilot["seed"], "method": pilot["config"]["method"],
    }
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", catalog)

    def relabel():
        teacher = SAC.load(teacher_checkpoint, device=args.device)
        student = SAC.load(student_checkpoint, device=args.device)
        env = MergeEnv(output, EnvConfig(**config["environment"]))
        try:
            for model in (teacher, student):
                if model.observation_space != env.observation_space or model.action_space != env.action_space:
                    raise ValueError("Policy spaces do not match the relabeling environment.")
            scenes = [Scene(**row) for row in catalog["train"]]
            relabel_student_rollouts(teacher, student, env, scenes, seed, quota, dataset, summary, output)
        finally:
            env.close()

    run_with_status(output, relabel)


def train_student_command(args):
    if not 0 <= args.seed < 2**31:
        raise ValueError("Use a nonnegative signed 31-bit seed.")
    scratch = args.method == "scratch"
    if scratch == (args.data is not None):
        raise ValueError("BC and soft distillation require --data. The scratch student does not use it.")
    config = student_config(read_json(args.config), read_json(args.student_config), args.method, args.device)
    if (args.corrections is None) != (args.correction_supervision is None):
        raise ValueError("Use --corrections and --correction-supervision together.")
    if args.corrections is not None:
        correction_batch = 16 if args.correction_batch_size is None else args.correction_batch_size
        if args.method != "bc" or not 1 <= correction_batch < config["distillation"]["batch_size"]:
            raise ValueError("Correction supervision requires BC and a correction batch smaller than the total batch.")
        config["correction"] = {"supervision": args.correction_supervision, "batch_size": correction_batch}
    elif args.correction_batch_size is not None:
        raise ValueError("--correction-batch-size requires --corrections.")
    catalog = make_catalog(**config["scenarios"])
    serialized = serialize_catalog(catalog)
    meta = manifest(config, args.seed, config["sac"]["device"], args.allow_dirty)
    if not scratch:
        if read_json(args.data / "status.json")["state"] != "completed":
            raise ValueError("The teacher collection must be complete.")
        collection = read_json(args.data / "manifest.json")
        summary = read_json(args.data / "collection_summary.json")
        steps = collection["config"]["steps_per_scene"]
        dataset = (args.data / "trajectories.npz").resolve(strict=True)
        dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
        if (collection["config"]["split"] != "train" or steps < 1
                or collection["config"]["environment"] != config["environment"]
                or collection["observation_fields"] != meta["observation_fields"]
                or read_json(args.data / "scenes.json") != serialized
                or summary["decision_steps"] != steps * len(catalog["train"])
                or dataset_hash != summary["dataset_sha256"]):
            raise ValueError("Teacher data do not match the student environment, catalog or collection manifest.")
        meta["teacher_data"] = {
            "collection": str(args.data.resolve()), "dataset_sha256": dataset_hash,
            "collection_git_sha": collection["git_sha"], "collection_seed": collection["seed"],
            "teacher": collection["teacher"],
        }
        if args.corrections is not None:
            bundle = args.corrections.resolve()
            correction = read_json(bundle / "manifest.json")
            targets = read_json(bundle / "summary.json")
            if (read_json(bundle / "status.json")["state"] != "completed"
                    or correction["config"]["base_dataset_sha256"] != dataset_hash
                    or correction["config"]["teacher_checkpoint_sha256"] != collection["teacher"]["checkpoint_sha256"]
                    or correction["config"]["environment"] != config["environment"]
                    or correction["observation_fields"] != meta["observation_fields"]
                    or read_json(bundle / "scenes.json") != serialized
                    or hashlib.sha256((bundle / "targets.npz").read_bytes()).hexdigest() != targets["targets_sha256"]):
                raise ValueError("The correction targets do not match the base dataset and environment.")
            meta["correction_data"] = {
                "bundle": str(bundle), "building_git_sha": correction["git_sha"],
                "targets_sha256": targets["targets_sha256"], "rows": targets["rows"],
            }
    else:
        meta["scratch_scene_sampling"] = "uniform"
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "scenes.json", serialized)

    def fit():
        if not scratch:
            train_offline(config, catalog, args.seed, output, dataset, steps, args.corrections)
            return
        start = perf_counter()
        validation_steps = train({**config, "method": "uniform"}, catalog, args.seed, output)
        summary = read_json(output / "training_summary.json")
        summary.update(wall_seconds=perf_counter() - start, validation_decision_steps=validation_steps)
        write_json(output / "training_summary.json", summary)

    run_with_status(output, fit)


def distillation_report_command(args):
    result = summarize_distillation(args.suite, args.split)
    meta = manifest({"suite": str(args.suite.resolve()), "split": args.split},
                    result["student_seeds"], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "report.json", result)
    print(output / "report.json")


def collection_ablation_report_command(args):
    result = summarize_collection_ablation(args.truncate_suite, args.full_episode_suite,
                                           args.split, args.bootstrap_seed)
    config = {
        "split": args.split, "truncate_suite": str(args.truncate_suite.resolve()),
        "full_episode_suite": str(args.full_episode_suite.resolve()), "bootstrap_seed": args.bootstrap_seed,
    }
    meta = manifest(config, result["student_seeds"], "cpu", args.allow_dirty)
    output = new_output(args.output)
    write_json(output / "manifest.json", meta)
    write_json(output / "report.json", result)
    print(output / "report.json")


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
    eval_parser.add_argument("--benchmark-actor", action="store_true")
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
    correction_parser = commands.add_parser("replay-corrections")
    correction_parser.add_argument("--teacher-run", type=Path, required=True)
    correction_parser.add_argument("--student-run", type=Path, required=True)
    correction_parser.add_argument("--scene", action="append")
    correction_parser.add_argument("--intervention-steps", type=int, default=4)
    correction_parser.set_defaults(command=replay_corrections_command)
    pre_wait_parser = commands.add_parser("replay-pre-wait")
    pre_wait_parser.add_argument("--teacher-run", type=Path, required=True)
    pre_wait_parser.add_argument("--student-run", type=Path, required=True)
    pre_wait_parser.add_argument("--scene", action="append")
    pre_wait_parser.add_argument("--intervention-steps", type=int, default=4)
    pre_wait_parser.set_defaults(command=replay_pre_wait_command)
    waiting_parser = commands.add_parser("diagnose-waiting")
    waiting_parser.add_argument("--suite", type=Path, required=True)
    waiting_parser.add_argument("--limit-per-transition", type=int)
    waiting_parser.add_argument("--intervention-steps", type=int, default=4)
    waiting_parser.set_defaults(command=diagnose_waiting_command)
    build_correction_parser = commands.add_parser("build-correction-targets")
    build_correction_parser.add_argument("--collection", type=Path, required=True)
    build_correction_parser.add_argument("--replay", type=Path, action="append", required=True)
    build_correction_parser.set_defaults(command=build_correction_targets_command)
    profile_parser = commands.add_parser("profile")
    profile_parser.add_argument("--config", type=Path, required=True)
    profile_parser.add_argument("--reference-speed", action="append", type=float)
    profile_parser.set_defaults(command=profile_command)
    collect_parser = commands.add_parser("collect-teacher")
    collect_parser.add_argument("--run", type=Path, required=True)
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--steps-per-scene", type=int, default=64)
    collect_parser.add_argument("--collection-mode", choices=COLLECTION_MODES, default="truncate")
    collect_parser.add_argument("--save-rollout-pool", action="store_true")
    collect_parser.add_argument("--device", default="cpu")
    collect_parser.set_defaults(command=collect_teacher_command)
    sample_parser = commands.add_parser("sample-teacher")
    sample_parser.add_argument("--collection", type=Path, required=True)
    sample_parser.set_defaults(command=sample_teacher_command)
    relabel_parser = commands.add_parser("relabel-student")
    relabel_parser.add_argument("--collection", type=Path, required=True)
    relabel_parser.add_argument("--student-run", type=Path, required=True)
    relabel_parser.add_argument("--device", default="cpu")
    relabel_parser.set_defaults(command=relabel_student_command)
    student_parser = commands.add_parser("train-student")
    student_parser.add_argument("--config", type=Path, required=True)
    student_parser.add_argument("--student-config", type=Path, required=True)
    student_parser.add_argument("--method", choices=STUDENT_METHODS, required=True)
    student_parser.add_argument("--data", type=Path)
    student_parser.add_argument("--corrections", type=Path)
    student_parser.add_argument("--correction-supervision", choices=CORRECTION_SUPERVISION)
    student_parser.add_argument("--correction-batch-size", type=int)
    student_parser.add_argument("--seed", type=int, required=True)
    student_parser.add_argument("--device")
    student_parser.set_defaults(command=train_student_command)
    distillation_parser = commands.add_parser("report-distillation")
    distillation_parser.add_argument("--suite", type=Path, required=True)
    distillation_parser.add_argument("--split", choices=("validation", "test", "stress"), default="validation")
    distillation_parser.set_defaults(command=distillation_report_command)
    ablation_parser = commands.add_parser("report-collection-ablation")
    ablation_parser.add_argument("--truncate-suite", type=Path, required=True)
    ablation_parser.add_argument("--full-episode-suite", type=Path, required=True)
    ablation_parser.add_argument("--split", choices=("validation", "test", "stress"), default="stress")
    ablation_parser.add_argument("--bootstrap-seed", type=int, default=104731)
    ablation_parser.set_defaults(command=collection_ablation_report_command)
    for command in (train_parser, eval_parser, report_parser, diagnose_parser, correction_parser, pre_wait_parser, waiting_parser,
                    build_correction_parser, profile_parser,
                    collect_parser, sample_parser, relabel_parser, student_parser, distillation_parser, ablation_parser):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    args.command(args)


if __name__ == "__main__":
    main()
