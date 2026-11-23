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
