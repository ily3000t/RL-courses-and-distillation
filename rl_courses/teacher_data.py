"""Capture matched-budget SAC teacher observations and action targets."""

from collections import Counter
from hashlib import sha256
from time import perf_counter

import numpy as np
import torch

from .records import write_json


@torch.no_grad()
def collect_trajectories(model, env, scenes, seed, steps_per_scene, output):
    start = perf_counter()
    actor = model.actor
    if actor.use_sde:
        raise ValueError("Distillation requires a diagonal Gaussian SAC actor without gSDE.")
    actor.eval()
    generator = torch.Generator(device=model.device).manual_seed(seed)
    observations, actions, means, log_stds, scene_indices = [], [], [], [], []
    episodes, outcome_counts = [], Counter()
    steps_by_difficulty = Counter()
    for index, scene in enumerate(scenes):
        scene_steps = 0
        while scene_steps < steps_per_scene:
            observation, _ = env.reset(seed=scene.seed, options={"scene": scene})
            episode_start = len(observations)
            while True:
                observation_tensor, _ = model.policy.obs_to_tensor(observation)
                mean, log_std, _ = actor.get_action_dist_params(observation_tensor)
                noise = torch.randn(mean.shape, generator=generator, device=model.device)
                action = torch.tanh(mean + log_std.exp() * noise)
                observations.append(observation.copy())
                actions.append(torch.tanh(mean)[0].cpu().numpy())
                means.append(mean[0].cpu().numpy())
                log_stds.append(log_std[0].cpu().numpy())
                scene_indices.append(index)
                observation, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
                scene_steps += 1
                if terminated or truncated or scene_steps == steps_per_scene:
                    outcome = info["outcome"] if terminated or truncated else "collection_cut"
                    outcome_counts[outcome] += 1
                    episodes.append({**info, "outcome": outcome, "start": episode_start,
                                     "stop": len(observations)})
                    break
        steps_by_difficulty[scene.difficulty] += scene_steps

    dataset = output / "trajectories.npz"
    np.savez_compressed(dataset, observations=np.stack(observations),
                        actions=np.stack(actions), means=np.stack(means), log_stds=np.stack(log_stds),
                        scene_indices=np.asarray(scene_indices, dtype=np.int64))
    write_json(output / "episodes.json", episodes)
    write_json(output / "collection_summary.json", {
        "decision_steps": len(observations), "steps_by_difficulty": dict(steps_by_difficulty),
        "episodes": len(episodes), "outcomes": dict(outcome_counts),
        "simulated_seconds_after_spawn": sum(row["simulated_seconds"] for row in episodes),
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "wall_seconds": perf_counter() - start,
    })
