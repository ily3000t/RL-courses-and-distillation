"""Adaptive scenario-group sampling with optional density-level backoff."""

from collections import Counter, deque
import math

import numpy as np


class ProgressSampler:
    block_order = None

    def __init__(self, scenes, group_by_scene, seed, *, warmup_steps,
                 window_episodes, exploration, temperature, density_backoff=False):
        if (not isinstance(window_episodes, int) or window_episodes < 1
                or not isinstance(warmup_steps, int) or warmup_steps < 0):
            raise ValueError("Progress windows must be positive and warmup_steps nonnegative.")
        if not 0 < exploration <= 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Exploration must be in (0, 1] and temperature must be positive.")
        self.groups = {}
        self.group_by_scene = group_by_scene
        self.group_density = {}
        for scene in scenes:
            group = group_by_scene[scene.scene_id]
            self.groups.setdefault(group, []).append(scene)
            self.group_density[group] = scene.difficulty
        self.group_ids = sorted(self.groups)
        density_counts = Counter(scene.difficulty for scene in scenes)
        self.base = np.array([
            len(self.groups[group]) / (len(density_counts) * density_counts[self.group_density[group]])
            for group in self.group_ids
        ])
        self.rng = np.random.default_rng(seed)
        self.warmup_steps = warmup_steps
        self.window_episodes = window_episodes
        self.exploration = exploration
        self.temperature = temperature
        history_size = 2 * window_episodes
        self.history = {group: deque(maxlen=history_size) for group in self.groups}
        self.density_history = (
            {density: deque(maxlen=history_size) for density in density_counts}
            if density_backoff else None
        )

    @staticmethod
    def _progress(history, window):
        if window == 0 or len(history) < 2 * window:
            return 0.0
        records = list(history)[-2 * window:]
        return max(0, sum(records[window:]) - sum(records[:window])) / window

    def _backoff_scores(self):
        density_progress = {
            density: self._progress(history, self.window_episodes)
            for density, history in self.density_history.items()
        }
        counts, local, weights, scores = {}, {}, {}, {}
        for group in self.group_ids:
            history = self.history[group]
            # Equal windows use the most recent even number of available episodes.
            window = len(history) // 2
            counts[group] = len(history)
            local[group] = self._progress(history, window)
            weights[group] = window / self.window_episodes
            parent = density_progress[self.group_density[group]]
            scores[group] = weights[group] * local[group] + (1 - weights[group]) * parent
        return scores, {
            "density_progress": density_progress,
            "group_window_counts": counts,
            "group_local_progress": local,
            "group_local_weights": weights,
        }

    def pick(self, step):
        details = {}
        if self.density_history is None:
            progress = {
                group: self._progress(self.history[group], self.window_episodes)
                for group in self.group_ids
            }
        else:
            progress, details = self._backoff_scores()
        scores = np.array(list(progress.values()))
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
            "group_progress": progress,
            **details,
        }
        scenes = self.groups[group]
        return scenes[int(self.rng.integers(len(scenes)))]

    def observe(self, scene, outcome):
        if outcome != "schedule_cut":
            success = outcome == "success"
            self.history[self.group_by_scene[scene.scene_id]].append(success)
            if self.density_history is not None:
                self.density_history[scene.difficulty].append(success)
