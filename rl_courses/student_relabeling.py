"""Substitute phase-matched student observations and query the fixed SAC teacher."""

from collections import Counter, defaultdict
from hashlib import sha256
from time import perf_counter

import numpy as np
import torch

from .distillation import load_teacher_targets
from .phase_sampling import PHASES, PRIORITY_PHASES, _phase_masks
from .records import write_json
from .teacher_data import TARGET_COLUMNS


def matched_relabeling_rows(teacher_observations, student_observations, rng):
    maximum = len(teacher_observations) // 4
    teacher_phases = dict(zip(PHASES, _phase_masks(teacher_observations), strict=True))
    student_phases = dict(zip(PHASES, _phase_masks(student_observations), strict=True))
    matches = {}
    for phase in PRIORITY_PHASES:
        slots = np.flatnonzero(teacher_phases[phase])
        states = np.flatnonzero(student_phases[phase])
        count = min(maximum, len(slots), len(states))
        matches[phase] = (
            np.sort(rng.choice(slots, size=count, replace=False)),
            np.sort(rng.choice(states, size=count, replace=False)),
        )
    return matches


def _student_rollout(student, env, scene):
    observation, _ = env.reset(seed=scene.seed, options={"scene": scene})
    observations, actions = [], []
    while True:
        action, _ = student.predict(observation, deterministic=True)
        observations.append(observation.copy())
        actions.append(action.copy())
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return np.stack(observations), np.stack(actions), info


@torch.no_grad()
def relabel_student_rollouts(teacher, student, env, scenes, seed, quota, source, summary, output):
    start = perf_counter()
    if teacher.actor.use_sde:
        raise ValueError("Relabeling requires a diagonal Gaussian teacher without gSDE.")
    teacher.actor.eval()
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,)))
    row_count = len(scenes) * quota
    targets = load_teacher_targets(source, env.observation_space.shape, env.action_space.shape, row_count, quota)
    observations, actions, means, log_stds = targets
    source_indices = np.full(row_count, -1, dtype=np.int64)
    raw_observations, raw_actions, raw_scene_indices, episodes = [], [], [], []
    counts = defaultdict(Counter)
    raw_steps = 0
    for index, scene in enumerate(scenes):
        student_obs, student_actions, info = _student_rollout(student, env, scene)
        first = index * quota
        matched = matched_relabeling_rows(observations[first:first + quota], student_obs, rng)
        phase_counts = {phase: len(rows[0]) for phase, rows in matched.items()}
        slots = first + np.concatenate([rows[0] for rows in matched.values()])
        picks = np.concatenate([rows[1] for rows in matched.values()])
        if len(slots):
            selected_obs = student_obs[picks]
            tensor, _ = teacher.policy.obs_to_tensor(selected_obs)
            mean, log_std, _ = teacher.actor.get_action_dist_params(tensor)
            observations[slots] = selected_obs
            actions[slots] = mean.tanh().cpu().numpy()
            means[slots] = mean.cpu().numpy()
            log_stds[slots] = log_std.cpu().numpy()
            source_indices[slots] = raw_steps + picks
        counts[scene.difficulty].update(phase_counts)
        raw_observations.append(student_obs)
        raw_actions.append(student_actions)
        raw_scene_indices.extend([index] * len(student_obs))
        episodes.append({**info, "start": raw_steps, "stop": raw_steps + len(student_obs),
                         "relabeled_rows_by_phase": phase_counts})
        raw_steps += len(student_obs)

    replaced = source_indices >= 0
    if not replaced.any():
        raise ValueError("Student rollouts supplied no phase-matched replacement rows.")
    dataset = output / "trajectories.npz"
    np.savez_compressed(dataset, **dict(zip(TARGET_COLUMNS, targets, strict=True)),
                        scene_indices=np.repeat(np.arange(len(scenes), dtype=np.int64), quota),
                        student_rollout_indices=source_indices)
    student_pool = output / "student_rollouts.npz"
    np.savez_compressed(student_pool, observations=np.concatenate(raw_observations),
                        actions=np.concatenate(raw_actions), scene_indices=np.asarray(raw_scene_indices, dtype=np.int64))
    write_json(output / "student_episodes.json", episodes)
    write_json(output / "collection_summary.json", {
        **summary, "sampling_mode": "student-relabeling",
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "source_uniform_dataset_sha256": summary["dataset_sha256"],
        "student_rollout_sha256": sha256(student_pool.read_bytes()).hexdigest(),
        "student_rollout_decision_steps": raw_steps,
        "student_simulated_seconds_after_spawn": sum(row["simulated_seconds"] for row in episodes),
        "student_rollout_outcomes": dict(Counter(row["outcome"] for row in episodes)),
        "student_rollout_episodes": len(episodes), "teacher_query_rows": int(replaced.sum()),
        "relabeled_rows_by_difficulty_and_phase": {label: dict(values) for label, values in counts.items()},
        "student_rollout_and_label_wall_seconds": perf_counter() - start,
    })
