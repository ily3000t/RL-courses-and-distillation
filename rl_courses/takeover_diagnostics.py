"""Compare continuous teacher takeover and teacher rollouts from episode start."""

from collections import Counter
from pathlib import Path
from time import perf_counter

from stable_baselines3 import SAC

from .correction_replay import BRANCHES, _branch, _terminal_outcome
from .environment import EGO, EnvConfig, MergeEnv
from .pre_wait_replay import ANCHORS, _capture_anchors, _zone_wait_seconds
from .records import read_json, write_json
from .scenarios import Scene
from .waiting_diagnostics import _digest, _rollout, _trace_summary


def takeover_inputs(suite, limit):
    if limit is not None and limit < 1:
        raise ValueError("Use a positive case limit.")
    protocol_path = suite / "protocol.json"
    protocol = read_json(protocol_path)
    catalog = protocol["training_catalog"]
    keys = [(row["scene_id"], row["seed"]) for row in catalog]
    if (len({scene_id for scene_id, _ in keys}) != len(keys)
            or any(row["difficulty"] != "hard" for row in catalog)):
        raise ValueError("Use a unique Hard training catalog.")
    teacher = protocol["teacher"]
    if _digest(Path(teacher["checkpoint"])) != teacher["checkpoint_sha256"]:
        raise ValueError("The recorded teacher checkpoint has changed.")
    jobs = sorted(protocol["jobs"], key=lambda job: job["seed"])
    if [job["seed"] for job in jobs] != sorted(set(protocol["student_seeds"])):
        raise ValueError("The pre-wait suite must contain one job per student seed.")
    cases, students = [], {}
    versions = environment = None
    for job in jobs:
        seed = job["seed"]
        folder = suite / f"seed-{seed}"
        manifest_path, cases_path = folder / "manifest.json", folder / "cases.json"
        meta = read_json(manifest_path)
        rows = read_json(cases_path)
        config = meta["config"]
        student = config["student"]
        if versions is None:
            versions, environment = meta["versions"], config["environment"]
        if (read_json(folder / "status.json")["state"] != "completed"
                or meta["git_sha"] != protocol["git_sha"] or meta["git_status"]
                or meta["argv"] != job["argv"] or meta["versions"] != versions
                or config["split"] != "train" or config["environment"] != environment
                or config["evaluation_catalog"] != catalog or config["anchors"] != list(ANCHORS)
                or config["teacher"]["checkpoint_sha256"] != teacher["checkpoint_sha256"]
                or student["seed"] != seed or student["checkpoint_sha256"] != job["checkpoint_sha256"]
                or _digest(Path(student["checkpoint"])) != job["checkpoint_sha256"]
                or [(row["scene_id"], row["scene_seed"]) for row in rows] != keys):
            raise ValueError(f"Pre-wait replay does not match the recorded suite: {folder}.")
        students[seed] = {**student, "manifest_sha256": _digest(manifest_path), "cases_sha256": _digest(cases_path)}
        for scene, row in zip(catalog, rows, strict=True):
            if row["sustained_zone_wait_timeout"]:
                anchor = row["probes"].get(ANCHORS[0])
                if row["baseline_terminal_outcome"] != "pre_merge_timeout" or anchor is None:
                    raise ValueError(f"Invalid waiting-timeout anchor for seed {seed}, {scene['scene_id']}.")
                cases.append({
                    "student_seed": seed, "scene": scene, "baseline": row["baseline"],
                    "longest_zone_wait_seconds": row["longest_zone_wait_seconds"],
                    "anchor_decision": anchor["decision"], "anchor_ego": anchor["ego"],
                    "short_acceleration_outcome": anchor["correction"]["terminal_outcome"],
                })
    if not cases:
        raise ValueError("No recorded sustained merge-lane waiting timeouts.")
    return {
        "suite": str(suite), "source_protocol_sha256": _digest(protocol_path),
        "source_replay_git_sha": protocol["git_sha"], "source_replay_versions": versions,
        "split": "train", "environment": environment, "teacher": teacher, "students": students,
        "cases": cases[:limit], "source_case_count": len(cases), "case_limit": limit, "anchor": ANCHORS[0],
        "teacher_channels": BRANCHES["joint"], "takeover_ends_at": "original_episode_terminal",
        "policy_actions": "deterministic_sac",
    }


def _predictor(checkpoint, env):
    model = SAC.load(checkpoint, device="cpu")
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise ValueError(f"The checkpoint does not share the SUMO spaces: {checkpoint}.")
    return lambda observation: model.predict(observation, deterministic=True)[0]


def run_takeover_diagnostics(config, output):
    start = perf_counter()
    settings = EnvConfig(**config["environment"])
    env = MergeEnv(output, settings)
    records, from_start, students = [], {}, {}
    try:
        teacher = _predictor(config["teacher"]["checkpoint"], env)
        for case in config["cases"]:
            scene, seed = Scene(**case["scene"]), case["student_seed"]
            if seed not in students:
                students[seed] = _predictor(config["students"][seed]["checkpoint"], env)
            student = students[seed]
            obs, actions, ticks, anchors, baseline = _rollout(env, scene, student, False, capture=_capture_anchors())
            anchor = anchors.get(config["anchor"])
            if (baseline != case["baseline"] or anchor is None or anchor.decision != case["anchor_decision"]
                    or anchor.state["vehicles"][EGO] != case["anchor_ego"]
                    or _zone_wait_seconds(ticks, settings.sim_step) != case["longest_zone_wait_seconds"]):
                raise RuntimeError(f"Student replay differs from the pre-wait suite for seed {seed}, {scene.scene_id}.")
            branch_args = (env, scene, student, teacher, obs, actions, baseline, anchor)
            steps_left = settings.max_decisions - anchor.decision
            control = _branch(*branch_args, BRANCHES["student"], steps_left)
            takeover = _branch(*branch_args, BRANCHES["joint"], steps_left, stop_on_merge=False)
            if takeover["teacher_decisions"] != takeover["decisions"] - anchor.decision:
                raise RuntimeError(f"Teacher did not control every decision after the anchor in {scene.scene_id}.")
            records.append({
                **case, "anchor_time_s": anchor.state["time_s"],
                "student_trace": _trace_summary(ticks, settings.sim_step), "student_control": control,
                "teacher_takeover": takeover,
            })
            if scene.scene_id not in from_start:
                _, _, teacher_ticks, _, terminal = _rollout(env, scene, teacher, False)
                from_start[scene.scene_id] = {
                    **terminal, "terminal_outcome": _terminal_outcome(terminal),
                    "trace": _trace_summary(teacher_ticks, settings.sim_step),
                }
    finally:
        env.close()
    write_json(output / "cases.json", records)
    write_json(output / "teacher_from_start.json", list(from_start.values()))
    write_json(output / "summary.json", {
        "paired_cases": len(records), "distinct_scenes": len(from_start),
        "student_control_outcomes": Counter(row["student_control"]["terminal_outcome"] for row in records),
        "short_acceleration_outcomes": Counter(row["short_acceleration_outcome"] for row in records),
        "teacher_takeover_outcomes": Counter(row["teacher_takeover"]["terminal_outcome"] for row in records),
        "teacher_from_start_outcomes": Counter(row["terminal_outcome"] for row in from_start.values()),
        "takeover_teacher_decision_steps": sum(row["teacher_takeover"]["teacher_decisions"] for row in records),
        "wall_seconds": perf_counter() - start,
    })
