"""Compare short teacher interventions from matched student traffic states."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter

import numpy as np
from stable_baselines3 import SAC

from .diagnostics import gap_counts
from .environment import EGO, EnvConfig, MergeEnv
from .phase_sampling import PRIORITY_PHASES, _phase_masks
from .records import write_json


BRANCHES = {
    "student": (False, False),
    "acceleration": (True, False),
    "merge_request": (False, True),
    "joint": (True, True),
}
PRE_MERGE_FAILURES = {"missed_merge", "pre_merge_timeout"}


@dataclass(frozen=True)
class Anchor:
    phase: str
    decision: int
    state: dict


def _terminal_outcome(info):
    if info["outcome"] == "timeout":
        return "post_merge_timeout" if info["merged"] else "pre_merge_timeout"
    return info["outcome"]


def _student_episode(env, scene, student):
    observation, _ = env.reset(seed=scene.seed, options={"scene": scene})
    observations, actions, anchors = [], [], {}
    reward_total = 0.0
    while True:
        if len(anchors) < len(PRIORITY_PHASES):
            _, near, zone, _ = _phase_masks(observation[None])
            for phase, eligible in zip(PRIORITY_PHASES, (near[0], zone[0]), strict=True):
                if eligible and phase not in anchors:
                    state = env.replay_state()
                    if phase != PRIORITY_PHASES[1] or state["vehicles"][EGO]["lane"] == "merge_0":
                        anchors[phase] = Anchor(phase, len(actions), state)
        action = student(observation)
        observations.append(observation.copy())
        actions.append(action.copy())
        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        if terminated or truncated:
            return observations, actions, anchors, {**info, "return": reward_total, "decisions": len(actions)}


def _branch(env, scene, student, teacher, observations, actions, baseline, anchor, mask, horizon):
    observation, info = env.reset(seed=scene.seed, options={"scene": scene})
    reward_total = 0.0
    for decision, action in enumerate(actions[:anchor.decision]):
        if not np.array_equal(observation, observations[decision]):
            raise RuntimeError(f"Replayed observation diverged in {scene.scene_id} at decision {decision}.")
        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        if terminated or truncated:
            raise RuntimeError(f"Prefix terminated before the anchor in {scene.scene_id}.")
    if not np.array_equal(observation, observations[anchor.decision]) or env.replay_state() != anchor.state:
        raise RuntimeError(f"Replayed traffic differs at the {anchor.phase} anchor in {scene.scene_id}.")

    ticks, intervention = [], []
    decision = anchor.decision
    teacher_channels = np.asarray(mask, dtype=bool)
    uses_teacher = any(mask)
    env.trace_tick = ticks.append
    try:
        while True:
            student_action = student(observation)
            if not uses_teacher and (decision >= len(actions)
                                     or not np.array_equal(observation, observations[decision])
                                     or not np.array_equal(student_action, actions[decision])):
                raise RuntimeError(f"Student control branch diverged in {scene.scene_id} at decision {decision}.")
            action = student_action
            if uses_teacher and decision - anchor.decision < horizon and not info["merged"]:
                teacher_action = teacher(observation)
                action = np.where(teacher_channels, teacher_action, student_action)
                intervention.append({
                    "decision": decision, "observation": observation.tolist(),
                    "teacher_action": teacher_action.tolist(), "student_action": student_action.tolist(),
                    "applied_action": action.tolist(),
                })
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            decision += 1
            if terminated or truncated:
                break
    finally:
        env.trace_tick = None
    terminal = {**info, "return": reward_total, "decisions": decision}
    if not uses_teacher and terminal != baseline:
        raise RuntimeError(f"Student control outcome differs from the original in {scene.scene_id}.")
    entering = next((row for row in ticks if row["lane_after"] == "merge_1"), None)
    merge_time = (None if entering is None else
                  round(entering["time_s"] + env.config.sim_step - anchor.state["time_s"], 6))
    return {
        **terminal, "terminal_outcome": _terminal_outcome(info), "teacher_channels": list(mask),
        "teacher_decisions": len(intervention), "intervention": intervention, "merge_seconds_after_anchor": merge_time,
        **gap_counts(ticks, env.config.sim_step), "requests": sum(row["requested"] for row in ticks),
        "blocked_requests": sum(row["blocked_by_sumo"] is True for row in ticks),
    }


def classify_correction(branches):
    baseline = branches["student"]["terminal_outcome"]
    if baseline not in PRE_MERGE_FAILURES:
        return ("student_success" if baseline == "success" else "other_student_failure"), None
    singles = [name for name in ("acceleration", "merge_request") if branches[name]["terminal_outcome"] == "success"]
    if singles:
        selected = min(singles, key=lambda name: branches[name]["merge_seconds_after_anchor"])
        return ("either_single" if len(singles) == 2 else f"{selected}_only"), selected
    if branches["joint"]["terminal_outcome"] == "success":
        return "joint_only", "joint"
    return "no_recovery", None


def run_correction_replay(inputs, scenes, horizon, output):
    start = perf_counter()
    teacher_model, student_model = (SAC.load(row["checkpoint"], device="cpu") for row in inputs)
    settings = EnvConfig(**inputs[0]["environment"])
    env = MergeEnv(output, settings)
    per_phase = {
        phase: {"classes": Counter(), "branch_outcomes": {name: Counter() for name in BRANCHES},
                "baseline_failure_to_success": Counter({name: 0 for name in BRANCHES}),
                "baseline_success_to_failure": Counter({name: 0 for name in BRANCHES})}
        for phase in PRIORITY_PHASES
    }
    student_episodes = []
    baseline_decisions = replay_decisions = teacher_decisions = 0
    try:
        for model in (teacher_model, student_model):
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError("The teacher and student must share the SUMO observation and action spaces.")

        def teacher(obs):
            return teacher_model.predict(obs, deterministic=True)[0]

        def student(obs):
            return student_model.predict(obs, deterministic=True)[0]

        with (output / "branches.jsonl").open("w", encoding="utf-8") as stream:
            for scene in scenes:
                obs, actions, anchors, baseline = _student_episode(env, scene, student)
                student_episodes.append({**baseline, "terminal_outcome": _terminal_outcome(baseline), "anchors": list(anchors)})
                baseline_decisions += len(actions)
                for phase, anchor in anchors.items():
                    branches = {
                        name: _branch(env, scene, student, teacher, obs, actions, baseline, anchor, mask, horizon)
                        for name, mask in BRANCHES.items()
                    }
                    category, correction = classify_correction(branches)
                    stats = per_phase[phase]
                    stats["classes"][category] += 1
                    for name, result in branches.items():
                        outcome = result["terminal_outcome"]
                        stats["branch_outcomes"][name][outcome] += 1
                        stats["baseline_failure_to_success"][name] += int(
                            _terminal_outcome(baseline) in PRE_MERGE_FAILURES and outcome == "success"
                        )
                        stats["baseline_success_to_failure"][name] += int(
                            _terminal_outcome(baseline) == "success" and outcome != "success"
                        )
                        replay_decisions += result["decisions"]
                        teacher_decisions += result["teacher_decisions"]
                    state_json = json.dumps(anchor.state, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    record = {
                        "scene_id": scene.scene_id, "scene_seed": scene.seed, "phase": phase,
                        "anchor_decision": anchor.decision, "anchor_observation": obs[anchor.decision].tolist(),
                        "anchor_state": anchor.state, "anchor_state_sha256": sha256(state_json.encode("utf-8")).hexdigest(),
                        "prefix_actions": [action.tolist() for action in actions[:anchor.decision]],
                        "classification": category, "selected_branch": correction, "branches": branches,
                    }
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
    finally:
        env.close()
    write_json(output / "student_episodes.json", student_episodes)
    write_json(output / "summary.json", {
        "scenes": len(scenes), "student_outcomes": dict(Counter(row["terminal_outcome"] for row in student_episodes)),
        "intervention_steps": horizon, "intervention_seconds": horizon * settings.steps_per_action * settings.sim_step,
        "anchors_by_phase": {phase: sum(stats["classes"].values()) for phase, stats in per_phase.items()},
        "phases": per_phase, "student_baseline_decision_steps": baseline_decisions,
        "branch_replay_decision_steps": replay_decisions, "teacher_decision_steps": teacher_decisions,
        "wall_seconds": perf_counter() - start,
    })
