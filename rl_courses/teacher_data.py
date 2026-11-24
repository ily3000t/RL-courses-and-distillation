"""Capture matched-budget SAC teacher observations and action targets."""

from collections import Counter
from hashlib import sha256
from time import perf_counter

import numpy as np
import torch

from .records import write_json


COLLECTION_MODES = ("truncate", "full-episode")


@torch.no_grad()
def collect_trajectories(model, env, scenes, seed, steps_per_scene, output, collection_mode="truncate"):
    if collection_mode not in COLLECTION_MODES or steps_per_scene < 1:
        raise ValueError("Use a supported collection mode and a positive per-scene quota.")
    start = perf_counter()
    actor = model.actor
    if actor.use_sde:
        raise ValueError("Distillation requires a diagonal Gaussian SAC actor without gSDE.")
    actor.eval()
    generator = torch.Generator(device=model.device).manual_seed(seed)
    complete_episode = collection_mode == "full-episode"
    tail_seed, sampling_seed = np.random.SeedSequence(seed).spawn(2)
    tail_generator = torch.Generator(device=model.device).manual_seed(int(tail_seed.generate_state(1, np.uint64)[0]))
    sampler = np.random.default_rng(sampling_seed)
    targets, scene_indices, rollout_indices = [], [], []
    episodes, outcome_counts = [], Counter()
    steps_by_difficulty = Counter()
    rollout_steps_by_difficulty = Counter()
    retained_tail_steps_by_difficulty = Counter()
    rollout_steps = 0
    for index, scene in enumerate(scenes):
        scene_targets, scene_episodes = [], []
        while len(scene_targets) < steps_per_scene:
            observation, _ = env.reset(seed=scene.seed, options={"scene": scene})
            episode_start = len(scene_targets)
            while True:
                observation_tensor, _ = model.policy.obs_to_tensor(observation)
                mean, log_std, _ = actor.get_action_dist_params(observation_tensor)
                # Preserve the truncated rollout's noise stream through each scene's quota.
                active_generator = generator if len(scene_targets) < steps_per_scene else tail_generator
                noise = torch.randn(mean.shape, generator=active_generator, device=model.device)
                action = torch.tanh(mean + log_std.exp() * noise)
                scene_targets.append((observation.copy(), torch.tanh(mean)[0].cpu().numpy(),
                                      mean[0].cpu().numpy(), log_std[0].cpu().numpy()))
                observation, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
                reached_quota = len(scene_targets) == steps_per_scene
                if terminated or truncated or (reached_quota and not complete_episode):
                    outcome = info["outcome"] if terminated or truncated else "collection_cut"
                    outcome_counts[outcome] += 1
                    scene_episodes.append((episode_start, len(scene_targets), {**info, "outcome": outcome}))
                    break

        scene_steps = len(scene_targets)
        selected = np.arange(steps_per_scene)
        if scene_steps > steps_per_scene:
            selected = np.sort(sampler.choice(scene_steps, size=steps_per_scene, replace=False))
        for first, last, row in scene_episodes:
            begin, end = np.searchsorted(selected, (first, last))
            episodes.append({**row, "start": len(targets) + int(begin), "stop": len(targets) + int(end),
                             "rollout_start": rollout_steps + first, "rollout_stop": rollout_steps + last})
        targets.extend(scene_targets[position] for position in selected)
        scene_indices.extend([index] * steps_per_scene)
        rollout_indices.extend(int(position) + rollout_steps for position in selected)
        rollout_steps += scene_steps
        steps_by_difficulty[scene.difficulty] += steps_per_scene
        rollout_steps_by_difficulty[scene.difficulty] += scene_steps
        retained_tail_steps_by_difficulty[scene.difficulty] += int(np.count_nonzero(selected >= steps_per_scene))

    dataset = output / "trajectories.npz"
    observations, actions, means, log_stds = zip(*targets, strict=True)
    index_columns = {"rollout_indices": np.asarray(rollout_indices, dtype=np.int64)} if complete_episode else {}
    np.savez_compressed(dataset, observations=np.stack(observations),
                        actions=np.stack(actions), means=np.stack(means), log_stds=np.stack(log_stds),
                        scene_indices=np.asarray(scene_indices, dtype=np.int64), **index_columns)
    write_json(output / "episodes.json", episodes)
    write_json(output / "collection_summary.json", {
        "decision_steps": len(observations), "steps_by_difficulty": dict(steps_by_difficulty),
        "collection_mode": collection_mode, "rollout_decision_steps": rollout_steps,
        "rollout_steps_by_difficulty": dict(rollout_steps_by_difficulty),
        "retained_tail_steps_by_difficulty": dict(retained_tail_steps_by_difficulty),
        "episodes": len(episodes), "outcomes": dict(outcome_counts),
        "simulated_seconds_after_spawn": sum(row["simulated_seconds"] for row in episodes),
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "wall_seconds": perf_counter() - start,
    })
