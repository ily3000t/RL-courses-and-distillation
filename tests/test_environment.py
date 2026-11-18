from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from rl_courses.environment import MergeEnv
from rl_courses.scenarios import Scene


class MergeEnvTest(unittest.TestCase):
    def setUp(self):
        root = Path("runs")
        root.mkdir(exist_ok=True)
        temp = TemporaryDirectory(dir=root)
        self.addCleanup(temp.cleanup)
        self.env = MergeEnv(Path(temp.name))
        self.addCleanup(self.env.close)

    def rollout(self, scene, intent):
        observation, _ = self.env.reset(seed=1, options={"scene": scene})
        records = [(observation.tolist(),)]
        for _ in range(self.env.config.max_decisions):
            observation, reward, terminated, truncated, info = self.env.step(
                np.array([0.5, intent], dtype=np.float32)
            )
            self.assertTrue(self.env.observation_space.contains(observation))
            records.append((observation.tolist(), reward, terminated, truncated, info))
            if terminated or truncated:
                return records
        self.fail("Rollout did not end.")

    def test_reset_replays_traffic_and_actions(self):
        scene = Scene("replay", "hard", 42, 1350)
        self.assertEqual(self.rollout(scene, 1), self.rollout(scene, 1))

    def test_ego_needs_a_lane_change_to_complete_merge(self):
        scene = Scene("quiet", "easy", 25, 1)
        self.assertEqual(self.rollout(scene, 1)[-1][-1]["outcome"], "success")
        self.assertEqual(self.rollout(scene, -1)[-1][-1]["outcome"], "missed_merge")
