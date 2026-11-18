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


def serialize_catalog(catalog):
    return {split: [asdict(scene) for scene in scenes] for split, scenes in catalog.items()}
