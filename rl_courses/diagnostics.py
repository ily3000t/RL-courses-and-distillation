"""Replay saved policies and fixed-speed reference approaches on identical scenes."""

import hashlib
import json

import numpy as np
from stable_baselines3 import SAC

from .environment import EnvConfig, MergeEnv
from .records import read_json, write_json
from .scenarios import Scene


def reference_action(observation, speed_mps):
    """Track a repeatable approach speed and request each allowed merge."""
    speed = float(observation[0]) * 31
    acceleration = 0.8 * (speed_mps - speed)
    return np.array([np.clip(acceleration / (3 if acceleration >= 0 else 5), -1, 1), 1],
                    dtype=np.float32)


def gap_counts(rows, step_seconds):
    windows = 0
    longest = 0
    streak = 0
    ticks = 0
    for row in rows:
        if row["gap_proxy_open"]:
            ticks += 1
            windows += int(streak == 0)
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return {"gap_windows": windows, "gap_seconds": round(ticks * step_seconds, 6),
            "longest_gap_s": round(longest * step_seconds, 6)}


def replay(env, scene, action_for):
    observation, _ = env.reset(seed=scene.seed, options={"scene": scene})
    reward_total = 0.0
    while True:
        observation, reward, terminated, truncated, info = env.step(action_for(observation))
        reward_total += reward
        if terminated or truncated:
            return {**info, "return": reward_total}


def comparison_inputs(runs, split, selection):
    inputs = []
    for run in runs:
        manifest = read_json(run / "manifest.json")
        catalog = read_json(run / "scenes.json")
        scenes = [Scene(**row) for row in catalog[split]]
        checkpoint = (run / "best.zip").resolve(strict=True)
        inputs.append({
            "method": manifest["config"]["method"], "seed": manifest["seed"],
            "training_git_sha": manifest["git_sha"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "scenes": scenes, "environment": manifest["config"]["environment"],
        })
    if not all(row["scenes"] == inputs[0]["scenes"] and
               row["environment"] == inputs[0]["environment"] for row in inputs):
        raise ValueError("Diagnostic checkpoints must share the selected catalog and environment.")
    selected = [scene for scene in inputs[0]["scenes"] if not selection or scene.scene_id in selection]
    if not selected or (selection and set(selection) != {scene.scene_id for scene in selected}):
        raise ValueError("Unknown scene ID in diagnostic selection.")
    return inputs, selected


def run_diagnostics(inputs, scenes, speeds, output):
    config = EnvConfig(**inputs[0]["environment"])
    collect = []
    env = MergeEnv(output, config, trace_tick=collect.append)
    summaries = []
    try:
        policies = []
        for row in inputs:
            model = SAC.load(row["checkpoint"], device="cpu")
            policies.append((f'{row["method"]}:{row["seed"]}',
                             lambda obs, model=model: model.predict(obs, deterministic=True)[0]))
        for speed in speeds:
            policies.append((f"reference:{speed:g}mps",
                             lambda obs, speed=speed: reference_action(obs, speed)))
        with (output / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
            for scene in scenes:
                for name, act in policies:
                    collect.clear()
                    result = replay(env, scene, act)
                    summary = {
                        "policy": name, **result, **gap_counts(collect, config.sim_step),
                        "requests": sum(row["requested"] for row in collect),
                        "blocked_requests": sum(row["blocked_by_sumo"] is True for row in collect),
                        "open_ticks_without_request": sum(
                            row["gap_proxy_open"] and not row["requested"] for row in collect
                        ),
                    }
                    summaries.append(summary)
                    stream.write(json.dumps({"scene_id": scene.scene_id, "policy": name,
                                             "ticks": collect}, allow_nan=False) + "\n")
    finally:
        env.close()
    write_json(output / "episodes.json", summaries)
