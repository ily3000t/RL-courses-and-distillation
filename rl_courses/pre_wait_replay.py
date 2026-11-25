"""Test short longitudinal corrections just before low-speed merge waiting."""

from collections import Counter
from itertools import groupby
from time import perf_counter

from stable_baselines3 import SAC

from .correction_replay import Anchor, BRANCHES, _branch, _terminal_outcome
from .environment import EGO, EnvConfig, MergeEnv
from .phase_sampling import _phase_masks
from .records import write_json
from .waiting_diagnostics import WAIT_SECONDS, WAIT_SPEED_MPS, _rollout, _trace_summary


SLOWING_SPEED_MPS = 5.0
ANCHORS = ("merge_lane_entry", "slowing_before_wait")


def _is_zone_wait(tick):
    return tick["lane"] == "merge_0" and tick["speed_mps"] <= WAIT_SPEED_MPS


def _anchor_names(speed, seen):
    if speed <= WAIT_SPEED_MPS:
        return ()
    return tuple(name for name, eligible in zip(ANCHORS, (True, speed <= SLOWING_SPEED_MPS), strict=True)
                 if eligible and name not in seen)


def _capture_anchors():
    checked = 0
    reached_low_speed = False

    def capture(env, observation, ticks, decision, anchors):
        nonlocal checked, reached_low_speed
        if reached_low_speed or len(anchors) == len(ANCHORS):
            return
        reached_low_speed = any(_is_zone_wait(tick) for tick in ticks[checked:])
        checked = len(ticks)
        if reached_low_speed or not _phase_masks(observation[None])[2][0]:
            return
        # The observation filters expensive snapshots. Use the exact simulator speed in the recorded anchor.
        estimated = float(observation[0]) * 31
        if len(anchors) and estimated > SLOWING_SPEED_MPS + 1e-6:
            return
        state = env.replay_state()
        ego = state["vehicles"][EGO]
        if ego["lane"] == "merge_0":
            for name in _anchor_names(ego["speed"], anchors):
                anchors[name] = Anchor(name, decision, state)

    return capture


def _zone_wait_seconds(ticks, sim_step):
    longest = max((sum(1 for _ in consecutive) for slow, consecutive in groupby(ticks, _is_zone_wait) if slow),
                  default=0)
    return round(longest * sim_step, 6)


def run_pre_wait_replay(inputs, scenes, horizon, output):
    start = perf_counter()
    teacher, student = (SAC.load(row["checkpoint"], device="cpu") for row in inputs)
    settings = EnvConfig(**inputs[0]["environment"])
    env = MergeEnv(output, settings)
    baseline_outcomes = Counter()
    waiting_timeouts = 0
    records = []
    probes_by_anchor = {anchor: Counter() for anchor in ANCHORS}
    outcomes_by_anchor = {anchor: Counter() for anchor in ANCHORS}
    baseline_decisions = replay_decisions = teacher_decisions = unique_probes = 0

    def predict(obs):
        return student.predict(obs, deterministic=True)[0]

    def expert(obs):
        return teacher.predict(obs, deterministic=True)[0]

    try:
        for model in (teacher, student):
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError("The teacher and student must share the SUMO spaces.")
        for scene in scenes:
            obs, actions, ticks, anchors, baseline = _rollout(env, scene, predict, False, capture=_capture_anchors())
            outcome = _terminal_outcome(baseline)
            longest_zone_wait = _zone_wait_seconds(ticks, settings.sim_step)
            primary = outcome == "pre_merge_timeout" and longest_zone_wait >= WAIT_SECONDS
            baseline_outcomes[outcome] += 1
            waiting_timeouts += primary
            baseline_decisions += len(actions)
            shared = {}
            probes = {}
            for name, anchor in anchors.items():
                steps_left = settings.max_decisions - anchor.decision
                counter = probes_by_anchor[name]
                counter["eligible"] += 1
                if primary:
                    counter["eligible_waiting_timeout"] += 1
                if anchor.decision not in shared:
                    args = (env, scene, predict, expert, obs, actions, baseline, anchor)
                    control = _branch(*args, BRANCHES["student"], horizon)
                    corrected = _branch(*args, BRANCHES["acceleration"], horizon)
                    shared[anchor.decision] = (control, corrected)
                    replay_decisions += control["decisions"] + corrected["decisions"]
                    teacher_decisions += corrected["teacher_decisions"]
                    unique_probes += 1
                control, corrected = shared[anchor.decision]
                terminal = corrected["terminal_outcome"]
                handed_back = corrected["decisions"] > anchor.decision + corrected["teacher_decisions"]
                if primary:
                    outcomes_by_anchor[name][terminal] += 1
                    if terminal == "success" and handed_back:
                        counter["waiting_timeout_rescued_after_handback"] += 1
                        seconds = corrected["teacher_decisions"] * settings.steps_per_action * settings.sim_step
                        if corrected["merge_seconds_after_anchor"] > seconds:
                            counter["waiting_timeout_merged_after_handback"] += 1
                if outcome == "success" and terminal != "success":
                    counter["original_success_regressed"] += 1
                if outcome == "missed_merge" and terminal == "success":
                    counter["missed_merge_rescued"] += 1
                if outcome == "missed_merge" and terminal == "pre_merge_timeout":
                    counter["missed_merge_became_pre_merge_timeout"] += 1
                probes[name] = {
                    "decision": anchor.decision, "steps_left": steps_left,
                    "ego": anchor.state["vehicles"][EGO], "control": control, "correction": corrected,
                    "student_continued_after_intervention": handed_back,
                }
            records.append({
                "scene_id": scene.scene_id, "scene_seed": scene.seed, "baseline": baseline,
                "baseline_terminal_outcome": outcome, "sustained_zone_wait_timeout": primary,
                "longest_zone_wait_seconds": longest_zone_wait, "baseline_trace": _trace_summary(ticks, settings.sim_step),
                "probes": probes,
            })
    finally:
        env.close()
    write_json(output / "cases.json", records)
    write_json(output / "summary.json", {
        "scenes": len(records), "baseline_outcomes": baseline_outcomes, "sustained_zone_wait_timeouts": waiting_timeouts,
        "probes_by_anchor": probes_by_anchor, "wait_timeout_correction_outcomes": outcomes_by_anchor,
        "unique_paired_probes": unique_probes, "student_baseline_decision_steps": baseline_decisions,
        "branch_replay_decision_steps": replay_decisions, "teacher_decision_steps": teacher_decisions,
        "wall_seconds": perf_counter() - start,
    })
