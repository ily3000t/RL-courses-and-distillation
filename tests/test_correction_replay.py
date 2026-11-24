from types import SimpleNamespace
import unittest

import numpy as np

from rl_courses.correction_replay import Anchor, BRANCHES, _branch, classify_correction
from rl_courses.scenarios import Scene


class CorrectionReplayTest(unittest.TestCase):
    def test_distinguishes_single_and_joint_recovery(self):
        def result(outcome, time=None):
            return {"terminal_outcome": outcome, "merge_seconds_after_anchor": time}

        branches = {name: result("missed_merge") for name in BRANCHES}
        self.assertEqual(classify_correction(branches), ("no_recovery", None))
        branches["joint"] = result("success", 3)
        self.assertEqual(classify_correction(branches), ("joint_only", "joint"))
        branches["acceleration"] = result("success", 4)
        self.assertEqual(classify_correction(branches), ("acceleration_only", "acceleration"))
        branches["merge_request"] = result("success", 2)
        self.assertEqual(classify_correction(branches), ("either_single", "merge_request"))
        branches["student"] = result("success", 6)
        self.assertEqual(classify_correction(branches), ("student_success", None))

    def test_restores_prefix_and_hands_back_after_merge(self):
        student_action = np.array([0.3, -0.6], dtype=np.float32)
        teacher_action = np.array([-0.4, 0.8], dtype=np.float32)

        class Env:
            config = SimpleNamespace(sim_step=0.1)

            def reset(self, *, seed, options):
                self.actions = []
                return np.array([0], dtype=np.float32), {"merged": False}

            def replay_state(self):
                return {"time_s": 0.5 * len(self.actions)}

            def step(self, action):
                self.actions.append(action.copy())
                index = len(self.actions)
                return np.array([index], dtype=np.float32), 1, index == 3, False, {
                    "merged": index >= 2, "outcome": "success" if index == 3 else None,
                }

        env = Env()
        observations = [np.array([0], dtype=np.float32), np.array([1], dtype=np.float32)]
        output = _branch(
            env, Scene("test", "hard", 42, 2000), lambda obs: student_action, lambda obs: teacher_action,
            observations, [student_action], {}, Anchor("in_merge_zone_unmerged", 1, {"time_s": 0.5}),
            BRANCHES["acceleration"], 4,
        )
        self.assertEqual(output["teacher_decisions"], 1)
        np.testing.assert_array_equal(env.actions, [student_action, [teacher_action[0], student_action[1]], student_action])
        self.assertEqual(output["terminal_outcome"], "success")
