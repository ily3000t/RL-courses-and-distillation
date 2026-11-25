"""Run small SAC students with matched online budgets and different actor initializations."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .catalog import make_catalog, make_stress_holdout, serialize_catalog
from .distillation import student_config
from .environment import OBSERVATION_FIELDS, EnvConfig, MergeEnv
from .evaluation import evaluate
from .records import new_output, read_json, run_with_status, write_json
from .sampling import BaselineSampler
from .scenarios import Scene
from .training import TrainingEnv


ARMS = ("scratch", "bc", "acceleration-equal")
SPLITS = ("dev-stress", "final-stress")


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _pretrained_source(path, arm, seed, config, catalog):
    meta = read_json(path / "manifest.json")
    previous = meta["config"]
    correction = previous.get("correction")
    expected = None if arm == "bc" else {"supervision": arm, "batch_size": 16}
    trained = read_json(path / "training_summary.json")
    if (read_json(path / "status.json")["state"] != "completed" or meta["git_status"]
            or meta["seed"] != seed or meta["observation_fields"] != list(OBSERVATION_FIELDS)
            or previous["method"] != "bc" or correction != expected
            or previous["environment"] != config["environment"]
            or previous["distillation"] != config["distillation"]
            or previous["sac"]["policy_kwargs"] != config["sac"]["policy_kwargs"]
            or previous["sac"]["learning_rate"] != config["sac"]["learning_rate"]
            or read_json(path / "scenes.json") != catalog
            or trained["steps"] != 0 or trained["optimizer_updates"] != config["distillation"]["updates"]):
        raise ValueError(f"Expected a comparable completed offline {arm} run for seed {seed}: {path}.")
    checkpoint = (path / "final.zip").resolve(strict=True)
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "training_git_sha": meta["git_sha"], "manifest_sha256": file_hash(path / "manifest.json"),
        "teacher_data": meta["teacher_data"], "correction_data": meta.get("correction_data"),
        "offline_optimizer_updates": trained["optimizer_updates"],
        "offline_validation_decision_steps": trained["validation_decision_steps"],
    }


def _preparation_costs(sources):
    collection_rows = {}
    corrections = {}
    for per_seed in sources.values():
        for source in per_seed.values():
            data = source["teacher_data"]
            path = Path(data["collection"])
            if path not in collection_rows:
                summary = read_json(path / "collection_summary.json")
                if summary["dataset_sha256"] != data["dataset_sha256"]:
                    raise ValueError(f"Teacher collection changed: {path}.")
                collection_rows[path] = summary["rollout_decision_steps"]
        correction = per_seed["acceleration-equal"]["correction_data"]
        if correction["bundle"] not in corrections:
            path = Path(correction["bundle"])
            replays = read_json(path / "manifest.json")["config"]["replays"]
            corrections[correction["bundle"]] = sum(
                row["student_baseline_decision_steps"] + row["branch_replay_decision_steps"] for row in replays
            )
    if len(set(collection_rows.values())) != 1 or len(corrections) != 1:
        raise ValueError("Use one common demonstration dataset and one correction bundle.")
    teacher = next(iter(sources.values()))["bc"]["teacher_data"]["teacher"]
    return {
        "teacher_training_decisions": teacher["training"]["steps"],
        "teacher_reference_profile_episodes": teacher["opportunity_profiles"]["probe_episodes"],
        "teacher_reference_profile_simulated_seconds": teacher["opportunity_profiles"]["probe_simulated_seconds_after_spawn"],
        "shared_demonstration_collection_decisions": next(iter(collection_rows.values())),
        "correction_rollout_and_replay_decisions": next(iter(corrections.values())),
        "cost_scope": "Archived preparation counted once per suite. Online and evaluation decisions are separate.",
    }


def warm_start_design(base, student, bc_root, correction_root, seeds, budgets,
                      stress_seeds, scene_count, success_target, device):
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("Student seeds must be distinct nonnegative signed 31-bit integers.")
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] != 0 or budgets[-1] <= 0:
        raise ValueError("Budget checkpoints must be increasing, distinct, and start at zero.")
    if len(set(stress_seeds)) != len(SPLITS) or not 0 < success_target <= 1:
        raise ValueError("Provide distinct stress seeds and a success-rate target in (0, 1].")
    config = student_config(base, student, "scratch", device)
    if config["sac"]["train_freq"] != 1 or config["sac"]["gradient_steps"] != 1:
        raise ValueError("This experiment uses one SAC update per decision once learning starts.")
    catalog = serialize_catalog(make_catalog(**config["scenarios"]))
    sources = {
        str(seed): {
            arm: _pretrained_source(root / f"seed-{seed}" / "train", arm, seed, config, catalog)
            for arm, root in (("bc", bc_root), ("acceleration-equal", correction_root))
        }
        for seed in seeds
    }
    teacher_data = [source["teacher_data"] for pair in sources.values() for source in pair.values()]
    if (len({row["dataset_sha256"] for row in teacher_data}) != 1
            or len({row["teacher"]["checkpoint_sha256"] for row in teacher_data}) != 1
            or len({pair["acceleration-equal"]["correction_data"]["targets_sha256"] for pair in sources.values()}) != 1):
        raise ValueError("All initializations must share one teacher, demonstration dataset and correction bundle.")
    stress = serialize_catalog({
        split: make_stress_holdout(catalog, seed, scene_count)
        for split, seed in zip(SPLITS, stress_seeds, strict=True)
    })
    if {row["seed"] for row in stress[SPLITS[0]]} & {row["seed"] for row in stress[SPLITS[1]]}:
        raise ValueError("The development and final stress catalogs overlap.")
    return {
        "online": config, "scene_sampling": "uniform", "torch_cpu_threads_per_job": 1,
        "arms": list(ARMS), "student_seeds": sorted(seeds), "budgets": budgets, "catalog": catalog,
        "evaluation_catalogs": stress, "stress_seeds": dict(zip(SPLITS, stress_seeds, strict=True)),
        "threshold": {"success_rate_at_least": success_target, "collision_rate_at_most": 0.0},
        "checkpoint_selection": "fixed_online_steps_and_fixed_final_offline_initialization",
        "transferred_parameters": ["actor.latent_pi", "actor.mu"],
        "offline_sources": sources, "archived_costs": _preparation_costs(sources),
    }


def initialize_actor(model, source):
    """Transfer the deterministic actor trained by BC into a fresh SAC learner."""
    path = Path(source["checkpoint"])
    if file_hash(path) != source["checkpoint_sha256"]:
        raise ValueError(f"Frozen offline checkpoint has changed: {path}.")
    # Loading an SB3 archive initializes a temporary SAC model and changes RNG state.
    # Save the fresh learner's Python, NumPy, and Torch streams across that load.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[model.device.index or 0] if model.device.type == "cuda" else []):
        try:
            previous = SAC.load(path, device="cpu")
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    if (model.observation_space != previous.observation_space or model.action_space != previous.action_space
            or previous.actor.features_extractor.state_dict() or model.actor.features_extractor.state_dict()
            or previous.actor.use_sde or model.actor.use_sde):
        raise ValueError("Warm-start requires matching SAC spaces, a parameter-free encoder and diagonal Gaussian actors.")
    model.actor.latent_pi.load_state_dict(previous.actor.latent_pi.state_dict(), strict=True)
    model.actor.mu.load_state_dict(previous.actor.mu.state_dict(), strict=True)


class BudgetCheckpoints(BaseCallback):
    def __init__(self, output, steps, original_env):
        super().__init__()
        self.output = output
        self.steps = set(steps)
        self.original_env = original_env
        self.checkpoints = []

    def save_due(self):
        step = self.model.num_timesteps
        if step not in self.steps:
            return
        if step != self.original_env.decisions:
            raise RuntimeError("The SAC step count differs from the SUMO decision budget.")
        path = self.output / f"step-{step}.zip"
        self.model.save(path)
        self.checkpoints.append({"step": step, "checkpoint": path.name, "sha256": file_hash(path),
                                 "sac_updates": self.model._n_updates})
        self.steps.remove(step)
        write_json(self.output / "checkpoints.json", self.checkpoints)

    def _on_rollout_start(self):
        # The prior decision has already triggered its update. At step zero neither has occurred.
        self.save_due()
        if self.model.num_timesteps % 5000 == 0:
            write_json(self.output / "progress.json", {"online_decisions": self.model.num_timesteps})

    def _on_step(self):
        return True

    def _on_training_end(self):
        self.save_due()
        if self.steps:
            raise RuntimeError("Training skipped a frozen budget checkpoint.")
        write_json(self.output / "progress.json", {"online_decisions": self.model.num_timesteps})


def train_warm_start(design, arm, seed, output):
    start = perf_counter()
    config = design["online"]
    sampler = BaselineSampler("uniform", [Scene(**row) for row in design["catalog"]["train"]],
                              seed, **config["budget"])
    train_env = TrainingEnv(MergeEnv(output, EnvConfig(**config["environment"])),
                            sampler.pick, config["budget"]["block_steps"], output)
    try:
        model = SAC("MlpPolicy", train_env, seed=seed, **config["sac"])
        if arm != "scratch":
            initialize_actor(model, design["offline_sources"][str(seed)][arm])
        callback = BudgetCheckpoints(output, design["budgets"], train_env)
        model.learn(total_timesteps=design["budgets"][-1], callback=callback)
        if model.num_timesteps != design["budgets"][-1]:
            raise RuntimeError("Training exceeded the frozen decision budget.")
        # A deterministic exported actor omits the stochastic log-standard-deviation head.
        actor = model.actor
        actor_parameters = sum(p.numel() for layer in (actor.latent_pi, actor.mu) for p in layer.parameters())
        write_json(output / "training_summary.json", {
            "online_decisions": model.num_timesteps, "online_optimizer_updates": model._n_updates,
            "steps_by_difficulty": train_env.steps_by_difficulty, "actor_parameters": actor_parameters,
            "wall_seconds": perf_counter() - start,
        })
    finally:
        train_env.close()


def _completed_attempt(parent):
    completed = [
        path for path in sorted(parent.glob("attempt-*"))
        if (path / "status.json").is_file() and read_json(path / "status.json")["state"] == "completed"
    ]
    if len(completed) > 1:
        raise ValueError(f"Multiple completed attempts: {parent}.")
    return next(iter(completed), None)


def _new_attempt(parent):
    attempts = sorted(parent.glob("attempt-*"))
    if attempts and read_json(attempts[-1] / "status.json")["state"] != "failed":
        raise ValueError(f"The previous attempt has not failed or completed: {attempts[-1]}.")
    return new_output(parent / f"attempt-{len(attempts) + 1:03d}")


def _evaluate_checkpoints(design, split, training, output):
    scenes = [Scene(**row) for row in design["evaluation_catalogs"][split]]
    env = MergeEnv(output, EnvConfig(**design["online"]["environment"]))
    try:
        snapshots = read_json(training / "checkpoints.json")
        if [point["step"] for point in snapshots] != design["budgets"]:
            raise ValueError(f"Budget snapshots do not match the frozen protocol: {training}.")
        for point in snapshots:
            checkpoint = training / point["checkpoint"]
            if file_hash(checkpoint) != point["sha256"]:
                raise ValueError(f"Checkpoint changed: {checkpoint}.")
            model = SAC.load(checkpoint, device="cpu")
            if model.num_timesteps != point["step"]:
                raise ValueError(f"Checkpoint carries a different decision count: {checkpoint}.")
            summary, episodes = evaluate(model, env, scenes)
            write_json(output / f"step-{point['step']}.json", {**point, "summary": summary, "episodes": episodes})
    finally:
        env.close()


def _warm_start_job(suite, protocol, run_meta, arm, seed, split):
    torch.set_num_threads(protocol["config"]["torch_cpu_threads_per_job"])
    design = protocol["config"]
    job = suite / arm / f"seed-{seed}"
    identity = {"protocol_sha256": run_meta["protocol_sha256"], "arm": arm, "seed": seed}
    meta = {**run_meta, **identity}
    training = _completed_attempt(job / "train")
    if training is None:
        if split == "final-stress":
            raise ValueError(f"Final evaluation requires a completed pilot job: {job}.")
        training = _new_attempt(job / "train")
        write_json(training / "manifest.json", meta)
        run_with_status(training, lambda: train_warm_start(design, arm, seed, training))
        print(f"{arm} seed {seed}: finished {design['budgets'][-1]} training decisions.", flush=True)
    training_meta = read_json(training / "manifest.json")
    if any(training_meta[key] != value for key, value in identity.items()):
        raise ValueError(f"Training provenance differs from the protocol: {training}.")
    output = _completed_attempt(job / split)
    if output is None:
        output = _new_attempt(job / split)
        evaluation = {**meta, "split": split, "evaluation_device": "cpu", "training": str(training)}
        write_json(output / "manifest.json", evaluation)
        run_with_status(output, lambda: _evaluate_checkpoints(design, split, training, output))
    check = read_json(output / "manifest.json")
    if any(check[key] != value for key, value in {**identity, "split": split, "training": str(training)}.items()):
        raise ValueError(f"Evaluation provenance differs from the protocol: {output}.")
    return arm, seed


def run_warm_start_suite(suite, protocol, run_meta, split, workers):
    design = protocol["config"]
    jobs = list(product(design["student_seeds"], ARMS))
    if not 1 <= workers <= len(jobs):
        raise ValueError(f"Use between 1 and {len(jobs)} workers for this suite.")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_warm_start_job, suite, protocol, run_meta, arm, seed, split)
                   for seed, arm in jobs]
        for future in as_completed(futures):
            arm, seed = future.result()
            print(f"{arm} seed {seed}: {split} evaluation complete.", flush=True)
