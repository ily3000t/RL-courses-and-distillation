from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rl_courses.records import ROOT, read_json
from rl_courses.scenarios import Scene
from rl_courses.teacher_data import collect_trajectories


class TeacherDataTest(unittest.TestCase):
    def test_collection_preserves_quotas_and_separates_behavior_from_targets(self):
        scenes = [Scene("e", "easy", 1, 450), Scene("h", "hard", 2, 2000)]
        mean = torch.tensor([[0.5, -0.25]])

        class Env:
            def reset(self, *, seed, options):
                self.scene = options["scene"]
                self.steps = 0
                self.asserted_seed = seed
                return np.zeros(3, dtype=np.float32), {}

            def step(self, action):
                self.steps += 1
                done = self.steps == 2
                return np.array([*action, 0], dtype=np.float32), 0, done, False, {
                    "outcome": "success" if done else None, "simulated_seconds": self.steps / 2,
                    "scene_id": self.scene.scene_id, "scene_seed": self.asserted_seed,
                }

        actor = SimpleNamespace(use_sde=False, eval=lambda: None,
                                get_action_dist_params=lambda _: (mean, torch.zeros_like(mean), {}))
        teacher = SimpleNamespace(actor=actor, device=torch.device("cpu"), policy=SimpleNamespace(
            obs_to_tensor=lambda obs: (torch.as_tensor(obs), False),
        ))
        root = ROOT / "runs"
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as folder:
            output = Path(folder)
            collect_trajectories(teacher, Env(), scenes, 42, 3, output)
            with np.load(output / "trajectories.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["scene_indices"], [0, 0, 0, 1, 1, 1])
                np.testing.assert_allclose(data["actions"], np.repeat(mean.tanh().numpy(), 6, axis=0))
                self.assertFalse(np.allclose(data["observations"][1, :2], data["actions"][0]))
            episodes = read_json(output / "episodes.json")
            self.assertEqual([row["outcome"] for row in episodes], ["success", "collection_cut"] * 2)
            self.assertEqual([row["stop"] - row["start"] for row in episodes], [2, 1, 2, 1])
            self.assertEqual(read_json(output / "collection_summary.json")["decision_steps"], 6)

    def test_full_episode_extends_only_the_last_rollout_and_keeps_the_training_quota(self):
        scenes = [Scene("e", "easy", 1, 450), Scene("h", "hard", 2, 2000)]

        class Env:
            def reset(self, *, seed, options):
                self.steps = 0
                self.scene = options["scene"]
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                self.steps += 1
                done = self.steps == (2 if self.scene.difficulty == "easy" else 6)
                return action, 0, done and self.scene.difficulty == "easy", done and self.scene.difficulty == "hard", {
                    "outcome": ("success" if self.scene.difficulty == "easy" else "timeout") if done else None,
                    "simulated_seconds": self.steps / 2,
                }

        actor = SimpleNamespace(use_sde=False, eval=lambda: None, get_action_dist_params=lambda obs: (
            obs.unsqueeze(0), torch.zeros((1, 2)), {}
        ))
        teacher = SimpleNamespace(actor=actor, device=torch.device("cpu"), policy=SimpleNamespace(
            obs_to_tensor=lambda obs: (torch.as_tensor(obs), False),
        ))
        root = ROOT / "runs"
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as folder:
            truncated, full, repeated = (Path(folder) / name for name in ("truncate", "full", "repeated"))
            for path, mode in ((truncated, "truncate"), (full, "full-episode"), (repeated, "full-episode")):
                path.mkdir()
                collect_trajectories(teacher, Env(), scenes, 43, 3, path, mode, save_rollout_pool=path == full)

            baseline = read_json(truncated / "collection_summary.json")
            summary = read_json(full / "collection_summary.json")
            self.assertEqual(summary["dataset_sha256"], read_json(repeated / "collection_summary.json")["dataset_sha256"])
            self.assertEqual(baseline["rollout_decision_steps"], 6)
            self.assertEqual(summary["decision_steps"], 6)
            self.assertEqual(summary["rollout_decision_steps"], 10)
            self.assertEqual(summary["rollout_steps_by_difficulty"], {"easy": 4, "hard": 6})
            episodes = read_json(full / "episodes.json")
            self.assertEqual([row["outcome"] for row in episodes], ["success", "success", "timeout"])
            self.assertEqual([(row["rollout_start"], row["rollout_stop"]) for row in episodes],
                             [(0, 2), (2, 4), (4, 10)])
            with (np.load(truncated / "trajectories.npz", allow_pickle=False) as cutoff,
                  np.load(full / "trajectories.npz", allow_pickle=False) as complete):
                np.testing.assert_array_equal(complete["scene_indices"], [0, 0, 0, 1, 1, 1])
                raw = complete["rollout_indices"]
                self.assertTrue((np.diff(raw) > 0).all())
                self.assertTrue((raw[:3] < 4).all())
                self.assertTrue(((4 <= raw[3:]) & (raw[3:] < 10)).all())
                self.assertTrue((raw[3:] >= 7).any())
                for episode in episodes:
                    selected = raw[episode["start"]:episode["stop"]]
                    self.assertTrue(((episode["rollout_start"] <= selected) & (selected < episode["rollout_stop"])).all())
                local = raw - np.array([0] * 3 + [4] * 3)
                shared = local < 3
                self.assertTrue(shared[3:].any())
                for key in ("observations", "actions", "means", "log_stds"):
                    np.testing.assert_array_equal(complete[key][shared],
                                                  cutoff[key][complete["scene_indices"][shared] * 3 + local[shared]])
                with np.load(full / "rollout_pool.npz", allow_pickle=False) as pool:
                    np.testing.assert_array_equal(pool["scene_indices"], [0] * 4 + [1] * 6)
                    for key in ("observations", "actions", "means", "log_stds", "scene_indices"):
                        np.testing.assert_array_equal(complete[key], pool[key][raw])
