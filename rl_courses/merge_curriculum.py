"""Revisit pre-merge failures with same-scene evidence of a collision-free merge."""

from collections import deque

import numpy as np

from .catalog import group_by_difficulty
from .evaluation import OUTCOMES, is_pre_merge_failure
from .opportunities import opportunity_groups


def reference_evidence(bundle, scenes, environment):
    opportunity_groups(bundle, scenes, environment)
    evidence = {}
    for row in bundle["records"]:
        scene_id = row["scene"]["scene_id"]
        probes = row["probes"]
        if any(probe.get("outcome") not in OUTCOMES for probe in probes):
            raise ValueError(f"Invalid reference outcome for {scene_id}.")
        evidence[scene_id] = any(
            probe["accepted_merge"] and probe["outcome"] != "collision" for probe in probes
        )
    return evidence


class MergeCurriculumSampler:
    block_order = None

    def __init__(self, scenes, seed, *, warmup_steps, window_episodes, exploration,
                 reference=None, use_failure=True):
        if (type(warmup_steps) is not int or warmup_steps < 0
                or type(window_episodes) is not int or window_episodes < 1):
            raise ValueError("Use nonnegative integer warmup and a positive integer failure window.")
        if not 0 < exploration <= 1:
            raise ValueError("Exploration must be in (0, 1].")
        self.groups = group_by_difficulty(scenes)
        self.densities = tuple(sorted(self.groups))
        density_seed, *scene_seeds = np.random.SeedSequence(seed).spawn(1 + len(self.densities))
        self.density_rng = np.random.default_rng(density_seed)
        self.scene_rngs = {
            difficulty: np.random.default_rng(child)
            for difficulty, child in zip(self.densities, scene_seeds)
        }
        self.warmup_steps = warmup_steps
        self.exploration = exploration
        self.reference = reference
        self.history = (
            {scene.scene_id: deque(maxlen=window_episodes) for scene in scenes} if use_failure else {}
        )

    def pick(self, step):
        difficulty = self.densities[self.density_rng.integers(len(self.densities))]
        scenes = self.groups[difficulty]
        ids = [scene.scene_id for scene in scenes]
        histories = [self.history.get(scene_id, ()) for scene_id in ids]
        counts = [len(history) for history in histories]
        failures = [sum(history) for history in histories]
        # A Beta(1, 1) prior gives unseen scenes a failure estimate of one half.
        estimates = (np.array(failures) + 1) / (np.array(counts) + 2)
        eligible = [self.reference[scene_id] for scene_id in ids] if self.reference is not None else None
        scores = estimates if self.history else np.ones(len(scenes))
        if eligible is not None:
            scores = scores * eligible
        warmup = step < self.warmup_steps
        total = float(scores.sum())
        probabilities = np.full(len(scenes), 1 / len(scenes))
        if not warmup and total > 0:
            probabilities = self.exploration * probabilities + (1 - self.exploration) * scores / total
        index = int(self.scene_rngs[difficulty].choice(len(scenes), p=probabilities))
        self.selection = {
            "start_step": step, "difficulty": difficulty, "warmup": warmup,
            "uniform_fallback": not warmup and total == 0,
            "density_probability": 1 / len(self.densities),
            "scene_probability_given_density": float(probabilities[index]),
            "reference_evidence": eligible[index] if eligible is not None else None,
            "window_episodes": counts[index] if self.history else None,
            "window_pre_merge_failures": failures[index] if self.history else None,
            "estimated_pre_merge_failure_rate": float(estimates[index]) if self.history else None,
            "priority": float(scores[index]),
        }
        return scenes[index]

    def observe(self, scene, episode):
        if episode["outcome"] == "schedule_cut":
            return None
        failed = is_pre_merge_failure(episode)
        if self.history:
            self.history[scene.scene_id].append(failed)
        return failed
