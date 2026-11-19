"""Build independent training, validation and test scenario catalogs."""

from collections import defaultdict
from dataclasses import asdict

import numpy as np

from .scenarios import DEMAND, Scene


def make_catalog(*, seed, train_per_difficulty, validation_per_difficulty,
                 test_per_difficulty, stress_hard):
    sizes = {"train": train_per_difficulty, "validation": validation_per_difficulty,
             "test": test_per_difficulty}
    if min(*sizes.values(), stress_hard) < 1:
        raise ValueError("Each scenario split must be nonempty.")
    rng = np.random.default_rng(seed)
    total = len(DEMAND) * sum(sizes.values()) + stress_hard
    seeds = iter(rng.choice(2**31, size=total, replace=False))
    catalog = {}
    for split, size in sizes.items():
        catalog[split] = [
            Scene(f"{split}-{difficulty}-{index:03d}", difficulty, int(next(seeds)), demand)
            for difficulty, demand in DEMAND.items()
            for index in range(size)
        ]
    catalog["stress"] = [
        Scene(f"stress-hard-{index:03d}", "hard", int(next(seeds)),
              round(DEMAND["hard"] * 1.25))
        for index in range(stress_hard)
    ]
    return catalog


def group_by_difficulty(scenes):
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene.difficulty].append(scene)
    return dict(groups)


def make_stress_holdout(catalog, seed, count):
    if not 0 <= seed < 2**31:
        raise ValueError("Stress holdout seed must be a nonnegative signed 31-bit integer.")
    used = {row["seed"] for scenes in catalog.values() for row in scenes}
    if not 1 <= count <= 2**31 - len(used):
        raise ValueError("Stress holdout size must be positive and fit the seed space.")
    rng = np.random.default_rng(seed)
    scenes = []
    while len(scenes) < count:
        candidate = int(rng.integers(2**31))
        if candidate not in used:
            used.add(candidate)
            scenes.append(Scene(f"holdout-{seed}-{len(scenes):03d}", "hard", candidate,
                                round(DEMAND["hard"] * 1.25)))
    return scenes


def serialize_catalog(catalog):
    return {split: [asdict(scene) for scene in scenes] for split, scenes in catalog.items()}
