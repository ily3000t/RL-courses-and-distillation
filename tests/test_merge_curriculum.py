from dataclasses import asdict
import unittest

from rl_courses.merge_curriculum import MergeCurriculumSampler, reference_evidence
from rl_courses.opportunities import profile_source_hash
from rl_courses.scenarios import Scene


class MergeCurriculumTest(unittest.TestCase):
    def setUp(self):
        self.scenes = [Scene(name, "hard", seed, 2000)
                       for seed, name in enumerate(("a", "b", "c"), start=1)]
        self.scenes.append(Scene("e", "easy", 4, 450))
        self.reference = {"a": True, "b": True, "c": False, "e": False}

    def test_reference_evidence_requires_a_collision_free_accepted_merge(self):
        outcomes = [(True, "timeout"), (True, "success"), (True, "collision"), (False, "timeout")]
        bundle = {
            "environment": {}, "profile_source_sha256": profile_source_hash(),
            "reference_speeds_mps": [12],
            "records": [
                {"scene": asdict(scene), "probes": [
                    {"speed_mps": 12, "accepted_merge": accepted, "outcome": outcome},
                ]}
                for scene, (accepted, outcome) in zip(self.scenes, outcomes)
            ],
        }
        self.assertEqual(reference_evidence(bundle, self.scenes, {}), self.reference)
        del bundle["records"][0]["probes"][0]["outcome"]
        with self.assertRaisesRegex(ValueError, "Invalid reference outcome"):
            reference_evidence(bundle, self.scenes, {})

    def test_priorities_and_density_are_matched_across_ablations(self):
        condition = {
            "failure": {"reference": None},
            "reference": {"reference": self.reference, "use_failure": False},
            "combined": {"reference": self.reference},
        }
        estimate = {"a": .75, "b": .25, "c": .75, "e": .5}
        for_mode = {
            "failure": {"a": .75, "b": .25, "c": .75, "e": .5},
            "reference": {"a": 1., "b": 1., "c": 0., "e": 0.},
            "combined": {"a": .75, "b": .25, "c": 0., "e": 0.},
        }
        trajectories = {}
        for mode, options in condition.items():
            with self.subTest(mode=mode):
                sampler = MergeCurriculumSampler(
                    self.scenes, 42, warmup_steps=10, window_episodes=2,
                    exploration=.2, **options,
                )
                a, b, c, _ = self.scenes
                sampler.observe(a, {"outcome": "success", "merged": True})
                self.assertTrue(sampler.observe(a, {"outcome": "timeout", "merged": False}))
                self.assertTrue(sampler.observe(a, {"outcome": "missed_merge", "merged": False}))
                self.assertIsNone(sampler.observe(a, {"outcome": "schedule_cut", "merged": False}))
                self.assertFalse(sampler.observe(b, {"outcome": "timeout", "merged": True}))
                self.assertFalse(sampler.observe(b, {"outcome": "collision", "merged": False}))
                for _ in range(2):
                    sampler.observe(c, {"outcome": "timeout", "merged": False})
                density_sequence, seen = [], set()
                for step in range(256):
                    scene = sampler.pick(step)
                    density_sequence.append(scene.difficulty)
                    row = sampler.selection
                    priority = for_mode[mode][scene.scene_id]
                    self.assertEqual(row["priority"], priority)
                    self.assertEqual(row["density_probability"], .5)
                    if mode != "reference":
                        self.assertEqual(row["estimated_pre_merge_failure_rate"], estimate[scene.scene_id])
                    if scene.difficulty == "easy":
                        self.assertEqual(row["uniform_fallback"], step >= 10 and mode != "failure")
                        self.assertEqual(row["scene_probability_given_density"], 1)
                    elif step < 10:
                        self.assertAlmostEqual(row["scene_probability_given_density"], 1/3)
                    else:
                        total = sum(for_mode[mode][key] for key in ("a", "b", "c"))
                        self.assertAlmostEqual(row["scene_probability_given_density"], .2/3 + .8*priority/total)
                        seen.add(scene.scene_id)
                self.assertEqual(seen, {"a", "b", "c"})
                trajectories[mode] = density_sequence
        self.assertEqual(trajectories["combined"], trajectories["failure"])
        self.assertEqual(trajectories["combined"], trajectories["reference"])
