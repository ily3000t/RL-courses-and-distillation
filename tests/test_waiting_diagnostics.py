import unittest

from rl_courses.waiting_diagnostics import _low_speed_wait, _trace_summary


def tick(time, *, lane="merge_0", speed=0.5, distance=25, opened=False, request=True, blocked=False, after=None):
    return {"time_s": time, "lane": lane, "speed_mps": speed, "distance_to_merge_end_m": distance,
            "gap_proxy_open": opened, "requested": request, "blocked_by_sumo": blocked,
            "lane_after": lane if after is None else after}


class WaitingDiagnosticsTest(unittest.TestCase):
    def test_wait_requires_contiguous_low_speed_close_to_merge(self):
        slow = tick(0)
        fast = tick(1, speed=2)
        far = tick(1, lane="ramp_0", distance=150)
        self.assertTrue(_low_speed_wait([slow] * 10, 0.1))
        self.assertFalse(_low_speed_wait([slow] * 9 + [fast], 0.1))
        self.assertFalse(_low_speed_wait([slow] * 9 + [far], 0.1))

    def test_trace_separates_unrequested_gaps_from_blocked_and_accepted_requests(self):
        rows = [
            tick(25.0, lane="ramp_0", distance=100, request=False),
            tick(25.1, opened=True, request=False),
            tick(25.2, opened=False, blocked=True),
            tick(25.3, opened=True, after="merge_1"),
        ]
        result = _trace_summary(rows, 0.1)
        self.assertEqual(result["entered_merge_lane_seconds"], 0.1)
        self.assertEqual(result["merged_seconds"], 0.4)
        self.assertEqual(result["open_gap_seconds_without_request"], 0.1)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["blocked_requests"], 1)
        self.assertEqual(result["accepted_requests"], 1)
