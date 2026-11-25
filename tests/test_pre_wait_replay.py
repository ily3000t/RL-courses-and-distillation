import unittest

import numpy as np

from rl_courses.pre_wait_replay import _anchor_names, _capture_anchors, _zone_wait_seconds


class PreWaitReplayTest(unittest.TestCase):
    def test_entry_and_slowing_require_strictly_more_than_wait_speed(self):
        self.assertEqual(_anchor_names(12, {}), ("merge_lane_entry",))
        self.assertEqual(_anchor_names(5, {}), ("merge_lane_entry", "slowing_before_wait"))
        self.assertEqual(_anchor_names(5, {"merge_lane_entry": True}), ("slowing_before_wait",))
        self.assertEqual(_anchor_names(1, {}), ())

    def test_later_speed_rise_does_not_create_a_pre_wait_anchor(self):
        class Env:
            def replay_state(self):
                return {"vehicles": {"ego": {"speed": 3, "lane": "merge_0"}}}

        capture = _capture_anchors()
        observation = np.zeros(11, dtype=np.float32)
        observation[[0, 2, 3]] = [3 / 31, -0.05, 0.1]
        ticks = [{"lane": "merge_0", "speed_mps": 0.5}, {"lane": "merge_0", "speed_mps": 3}]
        anchors = {}
        capture(Env(), observation, ticks, 1, anchors)
        self.assertEqual(anchors, {})

    def test_zone_wait_streak_excludes_ramp_and_breaks_on_higher_speed(self):
        slow = {"lane": "merge_0", "speed_mps": 0.8}
        fast = {"lane": "merge_0", "speed_mps": 1.2}
        ramp = {"lane": "ramp_0", "speed_mps": 0.1}
        self.assertEqual(_zone_wait_seconds([ramp] * 10 + [slow] * 9 + [fast] + [slow], 0.1), 0.9)
