import unittest

from rl_courses.evaluation import summarize


class EvaluationTest(unittest.TestCase):
    def test_pre_merge_timeout_is_separate_from_merge_completed_timeout(self):
        result = summarize([
            {"outcome": "timeout", "merged": False, "return": -15},
            {"outcome": "timeout", "merged": True, "return": -15},
            {"outcome": "missed_merge", "merged": False, "return": -20},
            {"outcome": "success", "merged": True, "return": 25},
        ])
        self.assertEqual(result["timeout_rate"], .5)
        self.assertEqual(result["pre_merge_timeout_rate"], .25)
        self.assertEqual(result["pre_merge_failure_rate"], .5)
        self.assertEqual(result["missed_merge_rate"], .25)
