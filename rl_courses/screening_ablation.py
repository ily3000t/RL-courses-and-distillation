"""Compare verified acceleration corrections with matched unfiltered subsets."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .catalog import make_stress_holdout, serialize_catalog
from .correction_targets import (
    clip_targets, correction_replay_clips, load_correction_targets, sample_control_clips, write_correction_targets,
)
from .distillation import train_offline
from .distillation_report import _hard_indicators
from .environment import OBSERVATION_FIELDS, EnvConfig, MergeEnv
from .evaluation import evaluate, summarize
from .records import manifest, new_output, read_json, run_with_status, write_json
from .scenarios import Scene
from .warm_start import _completed_attempt, _new_attempt, file_hash


RANDOM_SEEDS = (104733, 104734)
SCREENED = "acceleration-equal"


def freeze_screening(reference, seeds, stress_seed, stress_count, output, allow_dirty):
    path = reference / "protocol.json"
    archived = read_json(path)["config"]
    seeds = sorted(seeds or archived["student_seeds"])
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(archived["student_seeds"]):
        raise ValueError("Use distinct student seeds with archived matched BC checkpoints.")
    sources = {str(seed): archived["offline_sources"][str(seed)] for seed in seeds}
    training = Path(sources[str(seeds[0])][SCREENED]["checkpoint"]).parent
    old_meta = read_json(training / "manifest.json")
    config = old_meta["config"]
    if config["correction"]["supervision"] != SCREENED:
        raise ValueError("The screened arm must use equal-weight acceleration supervision.")
    for seed in seeds:
        for arm, source in sources[str(seed)].items():
            checkpoint = Path(source["checkpoint"])
            directory = checkpoint.parent
            meta_path = directory / "manifest.json"
            meta = read_json(meta_path)
            expected = {**config}
            if arm == "bc":
                expected.pop("correction")
            if (read_json(directory / "status.json")["state"] != "completed"
                    or meta["config"] != expected or meta["versions"] != old_meta["versions"]
                    or file_hash(meta_path) != source["manifest_sha256"]
                    or file_hash(checkpoint) != source["checkpoint_sha256"]):
                raise ValueError(f"Archived training differs from the matched BC protocol: {directory}.")
    dataset = Path(old_meta["teacher_data"]["collection"])
    collection = read_json(dataset / "manifest.json")
    catalog = read_json(dataset / "scenes.json")
    dataset_hash = file_hash(dataset / "trajectories.npz")
    if (catalog != archived["catalog"] or dataset_hash != old_meta["teacher_data"]["dataset_sha256"]
            or collection["config"]["environment"] != config["environment"]
            or collection["observation_fields"] != list(OBSERVATION_FIELDS)):
        raise ValueError("The archived teacher data differ from the frozen matched BC inputs.")
    bundle = Path(old_meta["correction_data"]["bundle"])
    bundle_meta = read_json(bundle / "manifest.json")
    originals = bundle_meta["config"]["replays"]
    clips, replays = correction_replay_clips(
        [Path(replay["directory"]) for replay in originals],
        collection, dataset_hash, catalog["train"], include_unselected=True,
    )
    selected = [clip for clip in clips if clip.selected]
    rows = clip_targets(selected)
    obs, actions = load_correction_targets(bundle, (len(OBSERVATION_FIELDS),), (2,))
    rebuilt = [{"pilot_seed": seed, "scene_id": scene, "decision": step["decision"]} for seed, scene, step in rows]
    if (replays != originals or rebuilt != read_json(bundle / "rows.json")
            or not np.array_equal(obs, np.asarray([step["observation"] for _, _, step in rows], dtype=np.float32))
            or not np.array_equal(actions, np.asarray([step["teacher_action"] for _, _, step in rows], dtype=np.float32))
            or file_hash(bundle / "targets.npz") != old_meta["correction_data"]["targets_sha256"]):
        raise ValueError("The candidate pool does not reconstruct the archived screened targets.")
    evaluation = serialize_catalog({"stress": make_stress_holdout(catalog, stress_seed, stress_count)})["stress"]
    used = {row["seed"] for scenes in archived["evaluation_catalogs"].values() for row in scenes}
    if used & {row["seed"] for row in evaluation}:
        raise ValueError("Use new evaluation scenarios outside the warm-start catalogs.")
    design = {
        "training_config": config, "catalog": catalog, "student_seeds": seeds,
        "evaluation_catalog": evaluation, "stress_seed": stress_seed, "sources": sources,
        "reference_protocol_sha256": file_hash(path), "teacher_data": old_meta["teacher_data"],
        "steps_per_scene": collection["config"]["steps_per_scene"], "replays": replays,
        "pool": {"candidate_clips": len(clips), "selected_clips": len(selected), "selected_rows": len(rows)},
        "archived_costs": archived["archived_costs"], "random_controls": {},
        "primary": "screened_minus_average_random", "checkpoint": "final", "evaluation_device": "cpu",
        "checkpoint_training": "fixed_offline_updates", "new_online_training_decisions": 0,
    }
    meta = manifest(design, seeds, config["sac"]["device"], allow_dirty)
    if meta["versions"] != old_meta["versions"]:
        raise ValueError("Use the archived BC training device and runtime.")
    output = new_output(output)
    for random_seed in RANDOM_SEEDS:
        cohort = sample_control_clips(clips, random_seed)
        arm = f"random-{random_seed}"
        target = new_output(output / arm / "targets")
        run_with_status(target, lambda: write_correction_targets(target, clip_targets(cohort), OBSERVATION_FIELDS))
        design["random_controls"][arm] = {
            "seed": random_seed, "targets_sha256": file_hash(target / "targets.npz"),
            "rows_sha256": file_hash(target / "rows.json"), "clips": len(cohort),
            "also_screened_clips": sum(clip.selected for clip in cohort),
        }
    write_json(output / "protocol.json", meta)
    return output / "protocol.json"


def _train_controls(suite, design, meta):
    catalog = {split: [Scene(**row) for row in rows] for split, rows in design["catalog"].items()}
    data = Path(design["teacher_data"]["collection"]) / "trajectories.npz"
    if file_hash(data) != design["teacher_data"]["dataset_sha256"]:
        raise ValueError("Frozen teacher data changed.")
    for arm, targets in design["random_controls"].items():
        folder = suite / arm / "targets"
        if (file_hash(folder / "targets.npz") != targets["targets_sha256"]
                or file_hash(folder / "rows.json") != targets["rows_sha256"]):
            raise ValueError(f"Frozen random subset changed: {folder}.")
        for seed in design["student_seeds"]:
            parent = suite / arm / f"seed-{seed}" / "train"
            details = {**meta, "arm": arm, "seed": seed, "targets_sha256": targets["targets_sha256"]}
            output = _completed_attempt(parent)
            if output is None:
                output = _new_attempt(parent)
                write_json(output / "manifest.json", details)

                def train():
                    train_offline(design["training_config"], catalog, seed, output, data,
                                  design["steps_per_scene"], folder, validate=False)
                    write_json(output / "checkpoint.json", {"sha256": file_hash(output / "final.zip")})

                run_with_status(output, train)
                print(f"{arm} seed {seed}: offline training complete.", flush=True)
            recorded = read_json(output / "manifest.json")
            if any(recorded[key] != details[key] for key in ("protocol_sha256", "arm", "seed", "targets_sha256")):
                raise ValueError(f"Offline training differs from the frozen protocol: {output}.")
            summary = read_json(output / "training_summary.json")
            if summary["steps"] or summary["optimizer_updates"] != design["training_config"]["distillation"]["updates"]:
                raise ValueError(f"Unexpected training budget in {output}.")
            yield arm, seed, output / "final.zip", read_json(output / "checkpoint.json")["sha256"]


def _eval_job(suite, design, meta, arm, seed, checkpoint, digest):
    torch.set_num_threads(1)
    if file_hash(checkpoint) != digest:
        raise ValueError(f"The offline checkpoint changed: {checkpoint}.")
    parent = suite / arm / f"seed-{seed}" / "stress"
    identity = {"protocol_sha256": meta["protocol_sha256"], "seed": seed, "arm": arm, "checkpoint_sha256": digest}
    output = _completed_attempt(parent)
    if output is None:
        output = _new_attempt(parent)
        write_json(output / "manifest.json", {**meta, **identity, "checkpoint": str(checkpoint)})

        def rollout():
            policy = SAC.load(checkpoint, device="cpu")
            if policy.num_timesteps:
                raise ValueError("Expected a strictly offline SAC actor checkpoint.")
            env = MergeEnv(output, EnvConfig(**design["training_config"]["environment"]))
            try:
                scenes = [Scene(**row) for row in design["evaluation_catalog"]]
                _, episodes = evaluate(policy, env, scenes)
                write_json(output / "episodes.json", episodes)
            finally:
                env.close()

        run_with_status(output, rollout)
        print(f"{arm} seed {seed}: stress evaluation complete.", flush=True)
    recorded = read_json(output / "manifest.json")
    if any(recorded[key] != value for key, value in identity.items()):
        raise ValueError(f"Evaluation differs from the frozen protocol: {output}.")
    episodes = read_json(output / "episodes.json")
    if [(row["scene_id"], row["scene_seed"]) for row in episodes] != [
        (row["scene_id"], row["seed"]) for row in design["evaluation_catalog"]
    ]:
        raise ValueError(f"Evaluation changed the scenario pairing: {output}.")
    return arm, seed, episodes


def _summarize_screening(design, episodes):
    seeds = design["student_seeds"]
    indicators = {arm: _hard_indicators(by_seed, seeds) for arm, by_seed in episodes.items()}
    rng = np.random.default_rng(104731)
    a = rng.integers(len(seeds), size=(10000, len(seeds), 1))
    count = len(design["evaluation_catalog"])
    b = rng.integers(count, size=(10000, 1, count))
    random_arms = list(design["random_controls"])
    comparisons, random_vs_bc = {}, {}
    for metric in ("success_rate", "missed_merge_rate", "pre_merge_timeout_rate", "collision_rate"):
        y = indicators[SCREENED][metric].astype(float)
        random = [indicators[arm][metric] for arm in random_arms]
        controls = {**dict(zip(random_arms, random, strict=True)), "random_average": np.mean(random, axis=0),
                    "bc": indicators["bc"][metric]}

        def summarize_difference(diff):
            return {
                "difference_pp": float(100 * diff.mean()),
                "by_student_seed_pp": (100 * diff.mean(axis=1)).tolist(),
                "exploratory_95_interval_pp": (100 * np.quantile(diff[a, b].mean(axis=(1, 2)), [0.025, 0.975])).tolist(),
            }

        comparisons[metric] = {arm: summarize_difference(y - x) for arm, x in controls.items()}
        random_vs_bc[metric] = summarize_difference(controls["random_average"] - controls["bc"])
    return {
        "stress_seed": design["stress_seed"], "student_seeds": seeds, "scenarios": count,
        "new_online_training_decisions": 0, "pool": design["pool"], "random_controls": design["random_controls"],
        "archived_costs": design["archived_costs"],
        "groups": {arm: summarize([row for rows in by_seed.values() for row in rows])
                   for arm, by_seed in episodes.items()},
        "screened_minus_control": comparisons, "average_random_minus_bc": random_vs_bc,
        "random_average_scope": "Average two fixed draws within each seed and scenario.",
        "bootstrap": {"axes": ["student_seed", "scene"], "seed": 104731, "replicates": 10000},
    }


def run_screening(suite, workers, allow_dirty):
    path = suite / "protocol.json"
    frozen = read_json(path)
    design = frozen["config"]
    meta = manifest({"suite": str(suite), "workers": workers}, design["student_seeds"],
                    design["training_config"]["sac"]["device"], allow_dirty)
    if any(meta[key] != frozen[key] for key in ("git_status", "git_sha", "versions")):
        raise ValueError("Use the frozen screening suite commit and runtime.")
    meta["protocol_sha256"] = file_hash(path)
    if not 1 <= workers <= (len(RANDOM_SEEDS) + 2) * len(design["student_seeds"]):
        raise ValueError("Use a positive worker count no greater than the evaluation job count.")
    torch.set_num_threads(1)
    jobs = [(arm, int(seed), Path(source["checkpoint"]), source["checkpoint_sha256"])
            for seed, sources in design["sources"].items() for arm, source in sources.items()]
    jobs.extend(_train_controls(suite, design, meta))
    episodes = {}
    evaluation_meta = {**meta, "evaluation_device": "cpu"}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_eval_job, suite, design, evaluation_meta, *job) for job in jobs]
        for future in futures:
            arm, seed, rows = future.result()
            episodes.setdefault(arm, {})[seed] = rows
    write_json(suite / "report.json", {**_summarize_screening(design, episodes), "manifest": meta})
    return suite / "report.json"
