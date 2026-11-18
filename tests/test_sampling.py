from collections import Counter
import unittest

from rl_courses.catalog import make_catalog
from rl_courses.sampling import BaselineSampler


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.catalog = make_catalog(
            seed=1009, train_per_difficulty=6, validation_per_difficulty=2,
            test_per_difficulty=3, stress_hard=2,
        )
        self.train = self.catalog["train"]

    def test_catalog_seeds_are_unique(self):
        all_scenes = [scene for scenes in self.catalog.values() for scene in scenes]
        self.assertEqual(len({scene.seed for scene in all_scenes}), len(all_scenes))
        self.assertEqual(len({scene.scene_id for scene in all_scenes}), len(all_scenes))

    def test_shuffled_matches_fixed_budget_and_ends_on_hard(self):
        fixed = BaselineSampler("fixed", self.train, 42, 12, 4)
        shuffled = BaselineSampler("shuffled", self.train, 42, 12, 4)
        self.assertNotEqual(fixed.block_order, shuffled.block_order)
        self.assertEqual(fixed.block_order[-1], "hard")
        self.assertEqual(shuffled.block_order[-1], "hard")
        actual = lambda sampler: Counter(sampler.pick(step).difficulty for step in range(144))
        self.assertEqual(actual(fixed), actual(shuffled))
        self.assertEqual(shuffled.pick(144).difficulty, "hard")

    def test_seeded_uniform_only_draws_from_training_catalog(self):
        first = BaselineSampler("uniform", self.train, 42, 12, 4)
        second = BaselineSampler("uniform", self.train, 42, 12, 4)
        scenes = [first.pick(step) for step in range(144)]
        self.assertEqual(scenes, [second.pick(step) for step in range(144)])
        self.assertEqual({scene.difficulty for scene in scenes}, {"easy", "medium", "hard"})
        self.assertTrue(set(scenes) <= set(self.train))
