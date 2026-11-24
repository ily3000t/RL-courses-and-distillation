"""Allocate fixed teacher sample quotas around the merge decision."""

from collections import Counter, defaultdict
from hashlib import sha256
from time import perf_counter

import numpy as np

from .environment import OBSERVATION_FIELDS
from .records import read_json, write_json
from .teacher_data import TARGET_COLUMNS


WINDOW_METERS = 50
PRIORITY_PHASES = ("approach_within_50m", "in_merge_zone_unmerged")
PHASES = ("approach_over_50m", *PRIORITY_PHASES, "merged")


def _phase_masks(observations):
    distance = observations[:, OBSERVATION_FIELDS.index("distance_to_merge_start")]
    before_end = observations[:, OBSERVATION_FIELDS.index("distance_to_merge_end")] > 0
    merged = observations[:, OBSERVATION_FIELDS.index("in_target_lane")] == 1
    near = WINDOW_METERS / 300
    return (
        ~merged & (distance > near),
        ~merged & (distance > 0) & (distance <= near),
        ~merged & (distance <= 0) & before_end,
        merged,
    )


def select_phase_rows(observations, quota, rng):
    if not 1 <= quota <= len(observations):
        raise ValueError("A scene must have at least as many rollout rows as its positive retained quota.")
    minimum = quota // 4
    marked = np.zeros(len(observations), dtype=bool)
    masks = dict(zip(PHASES, _phase_masks(observations), strict=True))
    for name in PRIORITY_PHASES:
        eligible = np.flatnonzero(masks[name])
        marked[rng.choice(eligible, size=min(minimum, len(eligible)), replace=False)] = True
    remaining = quota - int(marked.sum())
    marked[rng.choice(np.flatnonzero(~marked), size=remaining, replace=False)] = True
    return np.flatnonzero(marked)


def sample_merge_phases(source, output, catalog, seed, quota):
    start = perf_counter()
    summary = read_json(source / "collection_summary.json")
    pool_path, uniform_path = (source / name for name in ("rollout_pool.npz", "trajectories.npz"))
    if (sha256(pool_path.read_bytes()).hexdigest() != summary["rollout_pool_sha256"]
            or sha256(uniform_path.read_bytes()).hexdigest() != summary["dataset_sha256"]
            or summary["outcomes"].get("collection_cut", 0)):
        raise ValueError("The full-episode pool or uniform dataset differs from the saved collection record.")
    scenes = catalog["train"]
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
    counts = defaultdict(Counter)
    with (np.load(pool_path, allow_pickle=False) as pool, np.load(uniform_path, allow_pickle=False) as uniform):
        raw_scenes, uniform_indices = pool["scene_indices"], uniform["rollout_indices"]
        if (len(raw_scenes) != summary["rollout_decision_steps"]
                or not np.all((raw_scenes >= 0) & (raw_scenes < len(scenes)))
                or np.any(np.diff(raw_scenes) < 0)
                or not np.array_equal(uniform["scene_indices"], np.repeat(np.arange(len(scenes)), quota))
                or np.any(np.diff(uniform_indices) <= 0)
                or np.any((uniform_indices < 0) | (uniform_indices >= len(raw_scenes)))):
            raise ValueError("Invalid raw or retained scene indices.")
        for key in (*TARGET_COLUMNS, "scene_indices"):
            if not np.array_equal(pool[key][uniform_indices], uniform[key]):
                raise ValueError(f"Uniform dataset is not a subset of the rollout pool: {key}.")
        bounds = np.searchsorted(raw_scenes, np.arange(len(scenes) + 1))
        observations = pool["observations"]
        masks = _phase_masks(observations)
        if not np.all(np.sum(masks, axis=0) == 1):
            raise ValueError("Each rollout observation must belong to exactly one collection phase.")
        indices = []
        for i, scene in enumerate(scenes):
            first, last = bounds[i:i + 2]
            selected = first + select_phase_rows(observations[first:last], quota, rng)
            counts[scene["difficulty"]].update({
                name: int(mask[selected].sum()) for name, mask in zip(PHASES, masks, strict=True)
            })
            indices.extend(selected)
        indices = np.asarray(indices, dtype=np.int64)
        dataset = output / "trajectories.npz"
        np.savez_compressed(dataset, **{key: pool[key][indices] for key in (*TARGET_COLUMNS, "scene_indices")},
                            rollout_indices=indices)

    episodes = read_json(source / "episodes.json")
    for row in episodes:
        first, last = np.searchsorted(indices, (row["rollout_start"], row["rollout_stop"]))
        row.update(start=int(first), stop=int(last))
    write_json(output / "episodes.json", episodes)
    retained_tails = defaultdict(int)
    for i, scene in enumerate(scenes):
        retained_tails[scene["difficulty"]] += int((indices[i * quota:(i + 1) * quota] >= bounds[i] + quota).sum())
    write_json(output / "collection_summary.json", {
        **summary, "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "source_uniform_dataset_sha256": summary["dataset_sha256"], "sampling_mode": "merge-phase",
        "retained_tail_steps_by_difficulty": dict(retained_tails),
        "retained_phase_rows_by_difficulty": {label: dict(values) for label, values in counts.items()},
        "resampling_wall_seconds": perf_counter() - start,
    })
