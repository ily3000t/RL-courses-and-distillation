from dataclasses import asdict
import unittest

from rl_courses.opportunities import opportunity_groups, profile_source_hash
from rl_courses.scenarios import Scene


class OpportunitiesTest(unittest.TestCase):
    def test_profiles_group_equal_density_scenes_by_accepted_references(self):
        scenes = [Scene(str(index), "hard", index, 2000) for index in range(3)]
        speeds = [12, 19, 26]
        bundle = {
            "environment": {}, "profile_source_sha256": profile_source_hash(),
            "reference_speeds_mps": speeds,
            "records": [
                {"scene": asdict(scene), "probes": [
                    {"speed_mps": speed, "accepted_merge": index < accepted}
                    for index, speed in enumerate(speeds)
                ]}
                for scene, accepted in zip(scenes, (0, 1, 3))
            ],
        }
        self.assertEqual(opportunity_groups(bundle, scenes, {}), {
            "0": "hard:none-observed", "1": "hard:some", "2": "hard:all",
        })
        with self.assertRaisesRegex(ValueError, "training catalog"):
            opportunity_groups(bundle, scenes[1:], {})
