import unittest

import numpy as np

from rl_courses.environment import OBSERVATION_FIELDS
from rl_courses.student_relabeling import matched_relabeling_rows


class StudentRelabelingTest(unittest.TestCase):
    def test_replaces_only_matching_phases_and_caps_each_phase(self):
        def observations(distances):
            result = np.zeros((len(distances), len(OBSERVATION_FIELDS)), dtype=np.float32)
            result[:, OBSERVATION_FIELDS.index("distance_to_merge_start")] = distances
            result[:, OBSERVATION_FIELDS.index("distance_to_merge_end")] = 0.2
            return result

        teacher = observations([0.8, 0.5, 0.15, 0.1, 0.1, -0.1, -0.1, -0.1])
        student = observations([0.5, 0.1, 0.1, 0.1, -0.1])
        first = matched_relabeling_rows(teacher, student, np.random.default_rng(2))
        repeat = matched_relabeling_rows(teacher, student, np.random.default_rng(2))
        near, zone = first.values()
        self.assertEqual([len(near[0]), len(zone[0])], [2, 1])
        self.assertTrue(set(near[0]) <= {2, 3, 4} and set(near[1]) <= {1, 2, 3})
        self.assertTrue(set(zone[0]) <= {5, 6, 7} and set(zone[1]) == {4})
        for phase in first:
            np.testing.assert_array_equal(first[phase], repeat[phase])
        no_matches = matched_relabeling_rows(teacher, student[:1], np.random.default_rng(2))
        self.assertFalse(any(len(slots) or len(rows) for slots, rows in no_matches.values()))
