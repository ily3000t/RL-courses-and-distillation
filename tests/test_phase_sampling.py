import unittest

import numpy as np

from rl_courses.environment import OBSERVATION_FIELDS
from rl_courses.phase_sampling import select_phase_rows


class PhaseSamplingTest(unittest.TestCase):
    def test_reserves_rows_when_available_and_backfills_shortages(self):
        obs = np.zeros((60, len(OBSERVATION_FIELDS)), dtype=np.float32)
        obs[:, OBSERVATION_FIELDS.index("distance_to_merge_end")] = 0.2
        obs[:30, OBSERVATION_FIELDS.index("distance_to_merge_start")] = 0.5
        obs[30:40, OBSERVATION_FIELDS.index("distance_to_merge_start")] = 0.1
        obs[43:, OBSERVATION_FIELDS.index("in_target_lane")] = 1
        indices = select_phase_rows(obs, 16, np.random.default_rng(42))
        np.testing.assert_array_equal(indices, select_phase_rows(obs, 16, np.random.default_rng(42)))
        self.assertEqual(len(indices), 16)
        self.assertTrue((np.diff(indices) > 0).all())
        self.assertGreaterEqual(int(((30 <= indices) & (indices < 40)).sum()), 4)
        self.assertEqual(int(((40 <= indices) & (indices < 43)).sum()), 3)
        np.testing.assert_array_equal(select_phase_rows(obs[:16], 16, np.random.default_rng(42)), np.arange(16))
