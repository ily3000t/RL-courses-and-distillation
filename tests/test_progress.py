import unittest

from rl_courses.progress import ProgressSampler
from rl_courses.scenarios import Scene


class ProgressTest(unittest.TestCase):
    def test_warmup_balances_density_and_feedback_increases_probability(self):
        scenes = [Scene("e", "easy", 1, 450), Scene("h1", "hard", 2, 2000),
                  Scene("h2", "hard", 3, 2000)]
        sampler = ProgressSampler(
            scenes, {scene.scene_id: scene.scene_id for scene in scenes}, 42,
            warmup_steps=5, window_episodes=1, exploration=0.2, temperature=0.1,
        )
        sampler.pick(0)
        self.assertEqual(sampler.selection["group_probabilities"], {"e": .5, "h1": .25, "h2": .25})
        sampler.observe(scenes[2], "timeout")
        sampler.observe(scenes[2], "schedule_cut")
        self.assertEqual(list(sampler.history["h2"]), [False])
        sampler.observe(scenes[2], "success")
        sampler.pick(5)
        self.assertEqual(sampler.selection["group_progress"]["h2"], 1)
        probabilities = sampler.selection["group_probabilities"]
        self.assertGreater(probabilities["h2"], .25)
        self.assertGreaterEqual(probabilities["h1"], .05)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
