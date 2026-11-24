"""Export teacher targets from successful short acceleration interventions."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .correction_replay import PRE_MERGE_FAILURES
from .phase_sampling import PRIORITY_PHASES
from .records import read_json, write_json


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def correction_replay_targets(replays, collection, base_dataset_hash, catalog):
    phase = PRIORITY_PHASES[0]
    training_scenes = {row["scene_id"]: row for row in catalog if row["difficulty"] == "hard"}
    teacher = collection["teacher"]
    replay_protocol = None
    samples, sources = [], {}
    for directory in replays:
        meta = read_json(directory / "manifest.json")
        config = meta["config"]
        student = config["student"]
        seed = student["seed"]
        student_training = read_json(Path(student["checkpoint"]).parent / "manifest.json")
        if read_json(directory / "status.json")["state"] != "completed" or seed in sources:
            raise ValueError("Use one completed replay per distinct student seed.")
        if (config["split"] != "train" or set(row["scene_id"] for row in config["evaluation_catalog"]) != set(training_scenes)
                or any(training_scenes[row["scene_id"]] != row for row in config["evaluation_catalog"])
                or config["environment"] != collection["config"]["environment"]
                or meta["observation_fields"] != collection["observation_fields"]
                or config["teacher"]["checkpoint_sha256"] != teacher["checkpoint_sha256"]
                or student["method"] != "bc" or student["checkpoint_sha256"] != _sha(Path(student["checkpoint"]))
                or student["training_git_sha"] != student_training["git_sha"]
                or student_training["teacher_data"]["dataset_sha256"] != base_dataset_hash):
            raise ValueError(f"Replay is incompatible with the training collection: {directory}.")
        protocol = ({key: value for key, value in config.items() if key != "student"}, meta["git_sha"], meta["versions"])
        if replay_protocol is None:
            replay_protocol = protocol
        if replay_protocol != protocol:
            raise ValueError("Replay protocols must match across pilot seeds.")

        records = directory / "branches.jsonl"
        seen = set()
        with records.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["phase"] != phase:
                    continue
                scene = row["scene_id"]
                if scene in seen or row["scene_seed"] != training_scenes[scene]["seed"]:
                    raise ValueError(f"Duplicate or inconsistent replay anchor: {scene}.")
                seen.add(scene)
                if row["selected_branch"] != "acceleration":
                    continue
                branches = row["branches"]
                branch = branches["acceleration"]
                if (branches["student"]["terminal_outcome"] not in PRE_MERGE_FAILURES
                        or branch["terminal_outcome"] != "success" or branch["teacher_channels"] != [True, False]
                        or not 1 <= len(branch["intervention"]) <= config["intervention_steps"]
                        or len(branch["intervention"]) != branch["teacher_decisions"]):
                    raise ValueError(f"Invalid selected correction: {scene}.")
                for offset, step in enumerate(branch["intervention"]):
                    if (step["decision"] != row["anchor_decision"] + offset
                            or step["applied_action"] != [step["teacher_action"][0], step["student_action"][1]]):
                        raise ValueError(f"Correction steps do not match the intervention: {scene}.")
                    samples.append((seed, scene, step))
        summary = read_json(directory / "summary.json")
        if len(seen) != summary["anchors_by_phase"][phase]:
            raise ValueError(f"Replay records are incomplete: {directory}.")
        sources[seed] = {"directory": str(directory.resolve()), "manifest_sha256": _sha(directory / "manifest.json"),
                         "branches_sha256": _sha(records), "student_checkpoint_sha256": student["checkpoint_sha256"],
                         **{key: summary[key] for key in ("student_baseline_decision_steps", "branch_replay_decision_steps",
                                                          "teacher_decision_steps")}}
    if not samples:
        raise ValueError("No verified approach acceleration corrections were found.")
    samples.sort(key=lambda row: (row[0], row[1], row[2]["decision"]))
    return samples, [sources[seed] for seed in sorted(sources)]


def write_correction_targets(output, samples, observation_fields):
    observations = np.array([step["observation"] for _, _, step in samples], dtype=np.float32)
    actions = np.array([step["teacher_action"] for _, _, step in samples], dtype=np.float32)
    _validate_targets(observations, actions, len(samples), (len(observation_fields),), (2,))
    dataset = output / "targets.npz"
    np.savez_compressed(dataset, observations=observations, actions=actions)
    write_json(output / "rows.json", [
        {"pilot_seed": seed, "scene_id": scene, "decision": step["decision"]}
        for seed, scene, step in samples
    ])
    write_json(output / "summary.json", {
        "rows": len(samples), "distinct_scenes": len({scene for _, scene, _ in samples}),
        "distinct_student_scenes": len({(seed, scene) for seed, scene, _ in samples}),
        "targets_sha256": _sha(dataset),
    })


def _validate_targets(observations, actions, rows, observation_shape, action_shape):
    for values, shape in ((observations, observation_shape), (actions, action_shape)):
        if (values.shape != (rows, *shape) or values.dtype != np.float32
                or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
            raise ValueError("Correction targets must be finite float32 arrays matching the observation and action spaces.")


def load_correction_targets(directory, observation_shape, action_shape):
    summary = read_json(directory / "summary.json")
    path = directory / "targets.npz"
    if read_json(directory / "status.json")["state"] != "completed" or _sha(path) != summary["targets_sha256"]:
        raise ValueError("Correction target bundle is incomplete or has changed.")
    with np.load(path, allow_pickle=False) as data:
        observations, actions = data["observations"], data["actions"]
        _validate_targets(observations, actions, summary["rows"], observation_shape, action_shape)
    if not len(observations):
        raise ValueError("Use a nonempty correction target bundle.")
    return observations, actions
