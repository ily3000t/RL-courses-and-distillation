from collections import Counter
import unittest

from rl_courses.correction_targets import CorrectionClip, sample_control_clips


class ControlClipTest(unittest.TestCase):
    def test_sampling_matches_stratum_counts_without_replacement(self):
        clips = [
            CorrectionClip(seed, str(index), [{}] * length, index < length)
            for seed in (42, 43) for length in (2, 4) for index in range(6)
        ]
        cohort = sample_control_clips(clips, 104733)
        strata = lambda rows: Counter((clip.pilot_seed, len(clip.steps)) for clip in rows)
        self.assertEqual(strata(cohort), strata(clip for clip in clips if clip.selected))
        self.assertEqual(cohort, sample_control_clips(clips, 104733))
        picks = {(clip.pilot_seed, clip.scene_id, len(clip.steps)) for clip in cohort}
        self.assertEqual(len(picks), len(cohort))
