"""Replay matched pressure cases and test short corrections from observed waits."""

from collections import Counter
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import median

from stable_baselines3 import SAC

from .correction_replay import Anchor, BRANCHES, _branch, _terminal_outcome
from .diagnostics import gap_counts
from .environment import EGO, MERGE_END_X, MERGE_START_X, EnvConfig, MergeEnv
from .phase_sampling import WINDOW_METERS, _phase_masks
from .records import read_json, write_json
from .scenarios import Scene


MODES = ("both", "acceleration-equal")
CASE_OUTCOMES = ("pre_merge_timeout", "success")
WAIT_SPEED_MPS = 1.0
WAIT_SECONDS = 1.0
PROBE_CHANNELS = {"low_speed_wait": "acceleration", "gap_without_request": "merge_request"}


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def waiting_replay_inputs(suite, limit):
    protocol_path = suite / "protocol.json"
    protocol = read_json(protocol_path)
    evaluation_catalog = protocol["catalogs"]["fresh-stress"]
    keys = [(row["scene_id"], row["seed"], row["difficulty"]) for row in evaluation_catalog]
    if len(set(keys)) != len(keys) or not all(difficulty == "hard" for _, _, difficulty in keys):
        raise ValueError("Use a unique Hard pressure catalog.")
    teacher = protocol["teacher"]
    teacher_checkpoint = Path(teacher["checkpoint"])
    teacher_run = teacher_checkpoint.parent
    environment = read_json(teacher_run / "manifest.json")["config"]["environment"]
    if _digest(teacher_checkpoint) != teacher["checkpoint_sha256"]:
        raise ValueError("The recorded teacher checkpoint has changed.")
    cases = []
    sources = {}
    versions = None
    selected_counts = Counter()
    for seed in sorted(protocol["student_seeds"]):
        outcomes = {}
        sources[seed] = {}
        for mode in MODES:
            folder = suite / mode / f"seed-{seed}/fresh-stress"
            manifest_path, episodes_path = folder / "manifest.json", folder / "episodes.json"
            meta = read_json(manifest_path)
            rows = read_json(episodes_path)
            checkpoint = Path(meta["checkpoint"])
            if versions is None:
                versions = meta["versions"]
            if (read_json(folder / "status.json")["state"] != "completed"
                    or meta["git_sha"] != protocol["git_sha"] or meta["git_status"]
                    or meta["seed"] != seed or meta["checkpoint_kind"] != "best"
                    or meta["stress_holdout_seed"] != protocol["stress_seeds"]["fresh-stress"]
                    or meta["evaluation_catalog"] != evaluation_catalog or meta["versions"] != versions
                    or meta["config"]["environment"] != environment
                    or meta["config"]["correction"]["supervision"] != mode
                    or _digest(checkpoint) != meta["checkpoint_sha256"]
                    or [(r["scene_id"], r["scene_seed"], r["difficulty"]) for r in rows] != keys):
                raise ValueError(f"Pressure evaluation does not match the recorded study: {folder}.")
            outcomes[mode] = rows
            sources[seed][mode] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": meta["checkpoint_sha256"],
                "manifest_sha256": _digest(manifest_path), "episodes_sha256": _digest(episodes_path),
            }
        for scenario, before, after in zip(evaluation_catalog, outcomes[MODES[0]], outcomes[MODES[1]], strict=True):
            category = _terminal_outcome(after)
            if (_terminal_outcome(before) == "missed_merge" and category in CASE_OUTCOMES
                    and (limit is None or selected_counts[category] < limit)):
                cases.append({"student_seed": seed, "scene": scenario, "transition": category,
                              "expected": dict(zip(MODES, (before, after), strict=True))})
                selected_counts[category] += 1
    if not cases:
        raise ValueError("No paired missed-merge transitions match the diagnostic.")
    return {
        "suite": str(suite), "source_protocol_sha256": _digest(protocol_path),
        "source_evaluation_git_sha": protocol["git_sha"], "source_evaluation_versions": versions,
        "stress_seed": protocol["stress_seeds"]["fresh-stress"], "teacher": teacher,
        "environment": environment, "students": sources, "cases": cases,
        "wait_speed_mps": WAIT_SPEED_MPS, "wait_seconds": WAIT_SECONDS, "probe_channels": PROBE_CHANNELS,
    }


def _near_merge(tick):
    if tick["lane"] == "merge_0":
        return True
    merge_length = MERGE_END_X - MERGE_START_X
    return tick["lane"] == "ramp_0" and merge_length <= tick["distance_to_merge_end_m"] <= merge_length + WINDOW_METERS


def _low_speed_wait(ticks, step_seconds):
    needed = ceil(WAIT_SECONDS / step_seconds)
    return len(ticks) >= needed and all(
        _near_merge(row) and row["speed_mps"] <= WAIT_SPEED_MPS for row in ticks[-needed:]
    )


def _gap_unrequested(tick):
    return tick["lane"] == "merge_0" and tick["gap_proxy_open"] and not tick["requested"]


def _trace_summary(ticks, step_seconds):
    start = ticks[0]["time_s"]
    entry = next((tick for tick in ticks if tick["lane"] == "merge_0"), None)
    merged = next((tick for tick in ticks if tick["lane_after"] == "merge_1"), None)
    zone = [t for t in ticks if t["lane"] == "merge_0"]
    slow = [t for t in ticks if _near_merge(t) and t["speed_mps"] <= WAIT_SPEED_MPS]
    unrequested = [t for t in zone if _gap_unrequested(t)]
    longest = streak = 0
    for tick in ticks:
        streak = streak + 1 if _near_merge(tick) and tick["speed_mps"] <= WAIT_SPEED_MPS else 0
        longest = max(longest, streak)
    return {
        "entered_merge_lane_seconds": None if entry is None else round(entry["time_s"] - start, 6),
        "speed_at_merge_lane_entry_mps": None if entry is None else entry["speed_mps"],
        "merged_seconds": None if merged is None else round(merged["time_s"] - start + step_seconds, 6),
        "seconds_in_merge_lane": round(len(zone) * step_seconds, 6),
        "low_speed_seconds_near_merge": round(len(slow) * step_seconds, 6),
        "longest_low_speed_seconds_near_merge": round(longest * step_seconds, 6),
        "low_speed_seconds_before_merge_lane": round(sum(t["lane"] == "ramp_0" for t in slow) * step_seconds, 6),
        "low_speed_seconds_in_merge_lane": round(sum(t["lane"] == "merge_0" for t in slow) * step_seconds, 6),
        "first_low_speed_distance_to_merge_end_m": None if not slow else slow[0]["distance_to_merge_end_m"],
        "last_pre_merge_distance_to_end_m": zone[-1]["distance_to_merge_end_m"] if zone else None,
        **gap_counts(ticks, step_seconds),
        "open_gap_seconds_without_request": round(len(unrequested) * step_seconds, 6),
        "requests": sum(t["requested"] for t in zone),
        "blocked_requests": sum(t["blocked_by_sumo"] is True for t in zone),
        "accepted_requests": sum(t["requested"] and t["lane_after"] == "merge_1" for t in zone),
    }


def _rollout(env, scene, predict, test_anchors):
    observation, info = env.reset(seed=scene.seed, options={"scene": scene})
    observations, actions, ticks, anchors = [], [], [], {}
    env.trace_tick = ticks.append
    reward_total = 0.0
    try:
        while True:
            action = predict(observation)
            decision, first_tick = len(actions), len(ticks)
            candidate_gap = False
            if test_anchors and not info["merged"]:
                _, near, zone, _ = _phase_masks(observation[None])
                candidate_gap = "gap_without_request" not in anchors and bool(zone[0]) and action[1] <= 0
                waiting = ("low_speed_wait" not in anchors and bool(near[0] or zone[0])
                           and _low_speed_wait(ticks, env.config.sim_step))
                if waiting or candidate_gap:
                    state = env.replay_state()
                    candidate_gap = candidate_gap and state["vehicles"][EGO]["lane"] == "merge_0"
                    if waiting and state["vehicles"][EGO]["speed"] <= WAIT_SPEED_MPS:
                        anchors["low_speed_wait"] = Anchor("low_speed_wait", decision, state)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            if candidate_gap and any(_gap_unrequested(t) for t in ticks[first_tick:]):
                anchors["gap_without_request"] = Anchor("gap_without_request", decision, state)
            if terminated or truncated:
                baseline = {**info, "return": reward_total, "decisions": len(actions)}
                return observations, actions, ticks, anchors, baseline
    finally:
        env.trace_tick = None


def run_waiting_diagnostics(config, horizon, output):
    settings = EnvConfig(**config["environment"])
    env = MergeEnv(output, settings)
    summaries = []
    probe_counts = {name: Counter() for name in PROBE_CHANNELS}
    try:
        teacher = SAC.load(config["teacher"]["checkpoint"], device="cpu")
        students = {}
        with (output / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
            for case in config["cases"]:
                scene = Scene(**case["scene"])
                seed = case["student_seed"]
                record = {"student_seed": seed, "scene_id": scene.scene_id, "scene_seed": scene.seed,
                          "transition": case["transition"], "policies": {}, "probes": {}}
                for mode in MODES:
                    if (seed, mode) not in students:
                        students[seed, mode] = SAC.load(config["students"][seed][mode]["checkpoint"], device="cpu")
                    student = students[seed, mode]
                    predict = lambda obs: student.predict(obs, deterministic=True)[0]
                    expert = lambda obs: teacher.predict(obs, deterministic=True)[0]
                    test_anchors = mode == MODES[1] and case["transition"] == "pre_merge_timeout"
                    obs, actions, ticks, anchors, baseline = _rollout(env, scene, predict, test_anchors)
                    if baseline != case["expected"][mode]:
                        raise RuntimeError(f"Replay differs from the recorded evaluation: {scene.scene_id}, seed {seed}, {mode}.")
                    record["policies"][mode] = {**baseline, **_trace_summary(ticks, settings.sim_step)}
                    stream.write(json.dumps({"student_seed": seed, "scene_id": scene.scene_id,
                                             "mode": mode, "ticks": ticks}, allow_nan=False) + "\n")
                    for name, anchor in anchors.items():
                        channel = PROBE_CHANNELS[name]
                        arguments = (env, scene, predict, expert, obs, actions, baseline, anchor)
                        control = _branch(*arguments, BRANCHES["student"], horizon)
                        correction = _branch(*arguments, BRANCHES[channel], horizon)
                        record["probes"][name] = {
                            "anchor_decision": anchor.decision, "anchor_ego": anchor.state["vehicles"][EGO],
                            "channel": channel, "control": control, "correction": correction,
                        }
                        probe_counts[name][correction["terminal_outcome"]] += 1
                summaries.append(record)
    finally:
        env.close()

    groups = {}
    for outcome in CASE_OUTCOMES:
        rows = [row for row in summaries if row["transition"] == outcome]
        groups[outcome] = {"case_pairs": len(rows), "policies": {}}
        for mode in MODES:
            traces = [row["policies"][mode] for row in rows]
            timing = {}
            for metric in ("entered_merge_lane_seconds", "speed_at_merge_lane_entry_mps", "seconds_in_merge_lane",
                           "low_speed_seconds_near_merge", "first_low_speed_distance_to_merge_end_m", "gap_seconds"):
                available = [trace[metric] for trace in traces if trace[metric] is not None]
                timing[f"median_{metric}"] = median(available) if available else None
            groups[outcome]["policies"][mode] = {
                **timing,
                "episodes_with_low_speed_wait": sum(t["longest_low_speed_seconds_near_merge"] >= WAIT_SECONDS for t in traces),
                "episodes_with_gap": sum(t["gap_seconds"] > 0 for t in traces),
                "episodes_with_gap_without_request": sum(t["open_gap_seconds_without_request"] > 0 for t in traces),
                "episodes_with_a_blocked_request": sum(t["blocked_requests"] > 0 for t in traces),
            }
    write_json(output / "cases.json", summaries)
    write_json(output / "summary.json", {"groups": groups, "probe_outcomes": probe_counts})
