"""Episode-level sampling and matched training schedules."""

import numpy as np

from .catalog import group_by_difficulty
from .scenarios import DEMAND


BASELINES = ("hard", "uniform", "fixed", "shuffled")
BASELINE_METHODS = (*BASELINES, "plr")
BACKOFF_METHOD = "opportunity-progress-backoff"
OPPORTUNITY_METHODS = ("opportunity-progress", BACKOFF_METHOD)
PROGRESS_METHODS = ("density-progress", *OPPORTUNITY_METHODS)
REFERENCE_ONLY_METHOD = "merge-reference"
REFERENCE_METHODS = (REFERENCE_ONLY_METHOD, "merge-reference-failure")
MERGE_METHODS = ("merge-failure", *REFERENCE_METHODS)
PROFILE_METHODS = (*OPPORTUNITY_METHODS, *REFERENCE_METHODS)
COMPARISONS = {
    "baselines": BASELINE_METHODS,
    "curriculum": ("plr", "density-progress", "opportunity-progress"),
    "backoff": OPPORTUNITY_METHODS,
    "merge-curriculum": ("plr", *MERGE_METHODS),
}
METHODS = (*BASELINE_METHODS, *PROGRESS_METHODS, *MERGE_METHODS)


class BaselineSampler:
    def __init__(self, method, scenes, seed, block_steps, blocks_per_difficulty):
        if method not in BASELINES:
            raise ValueError(f"Unknown baseline: {method}")
        if block_steps < 1 or blocks_per_difficulty < 1:
            raise ValueError("Both training budget values must be positive.")
        if method == "shuffled" and blocks_per_difficulty < 2:
            raise ValueError("The shuffled schedule needs at least two blocks per difficulty.")
        self.method = method
        self.groups = group_by_difficulty(scenes)
        self.block_steps = block_steps
        order_seed, sample_seed = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(sample_seed)
        # Keep the within-difficulty seed stream independent of the order of the blocks.
        self.scene_rngs = {
            difficulty: np.random.default_rng(child)
            for difficulty, child in zip(DEMAND, sample_seed.spawn(len(DEMAND)))
        }

        if method == "uniform":
            self.block_order = None
            return
        if method == "hard":
            self.block_order = ["hard"] * (len(DEMAND) * blocks_per_difficulty)
            return
        order = [difficulty for difficulty in DEMAND for _ in range(blocks_per_difficulty)]
        if method == "shuffled":
            rng = np.random.default_rng(order_seed)
            tail = order.pop()
            original = order.copy()
            while order == original:
                rng.shuffle(order)
            order.append(tail)
        self.block_order = order

    def pick(self, step):
        if self.block_order is None:
            difficulty = tuple(DEMAND)[int(self.rng.integers(len(DEMAND)))]
        else:
            # DummyVecEnv resets eagerly after the final training transition.
            block = min(step // self.block_steps, len(self.block_order) - 1)
            difficulty = self.block_order[block]
        scenes = self.groups[difficulty]
        return scenes[int(self.scene_rngs[difficulty].integers(len(scenes)))]
