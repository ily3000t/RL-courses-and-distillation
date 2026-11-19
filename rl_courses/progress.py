"""Adaptive scenario-group sampling shared by both learning-progress comparisons."""

from collections import Counter, deque
import math

import numpy as np


class ProgressSampler:
    block_order = None

    def __init__(self, scenes, group_by_scene, seed, *, warmup_steps,
                 window_episodes, exploration, temperature):
        if (not isinstance(window_episodes, int) or window_episodes < 1
                or not isinstance(warmup_steps, int) or warmup_steps < 0):
            raise ValueError("Progress windows must be positive and warmup_steps nonnegative.")
        if not 0 < exploration <= 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Exploration must be in (0, 1] and temperature must be positive.")
        self.groups = {}
        self.group_by_scene = group_by_scene
        group_density = {}
        for scene in scenes:
            group = group_by_scene[scene.scene_id]
            self.groups.setdefault(group, []).append(scene)
            group_density[group] = scene.difficulty
        self.group_ids = sorted(self.groups)
        density_counts = Counter(scene.difficulty for scene in scenes)
        self.base = np.array([
            len(self.groups[group]) / (len(density_counts) * density_counts[group_density[group]])
            for group in self.group_ids
        ])
        self.rng = np.random.default_rng(seed)
        self.warmup_steps = warmup_steps
        self.window_episodes = window_episodes
        self.exploration = exploration
        self.temperature = temperature
        self.history = {group: deque(maxlen=2 * window_episodes) for group in self.groups}

    def _progress(self, group):
        records = list(self.history[group])
        window = self.window_episodes
        if len(records) < 2 * window:
            return 0.0
        return max(0, sum(records[window:]) - sum(records[:window])) / window

    def pick(self, step):
        scores = np.array([self._progress(group) for group in self.group_ids])
        if step < self.warmup_steps:
            probabilities = self.base
        else:
            priority = self.base * np.exp((scores - max(scores)) / self.temperature)
            probabilities = (1 - self.exploration) * priority / priority.sum() + self.exploration * self.base
        index = int(self.rng.choice(len(self.group_ids), p=probabilities))
        group = self.group_ids[index]
        self.selection = {
            "group": group, "warmup": step < self.warmup_steps,
            "group_probabilities": dict(zip(self.group_ids, probabilities.tolist())),
            "group_progress": dict(zip(self.group_ids, scores.tolist())),
        }
        scenes = self.groups[group]
        return scenes[int(self.rng.integers(len(scenes)))]

    def observe(self, scene, outcome):
        if outcome != "schedule_cut":
            self.history[self.group_by_scene[scene.scene_id]].append(outcome == "success")
