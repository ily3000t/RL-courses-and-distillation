import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from rl_courses.catalog import make_catalog
from rl_courses.opportunities import build_profiles
from rl_courses.records import ROOT, read_json
from rl_courses.sampling import BACKOFF_METHOD, BASELINE_METHODS, PROGRESS_METHODS
from rl_courses.training import train


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def test_all_baselines_use_the_same_decision_budget(self):
        root = ROOT / "runs"
        root.mkdir(exist_ok=True)
        for method in BASELINE_METHODS:
            with self.subTest(method=method), TemporaryDirectory(dir=root) as folder:
                output = Path(folder)
                config = read_json(ROOT / "configs" / "smoke.json")
                config["method"] = method
                train(config, make_catalog(**config["scenarios"]), 42, output)
                summary = read_json(output / "training_summary.json")
                self.assertEqual(summary["steps"], 72)
                self.assertEqual(sum(summary["steps_by_difficulty"].values()), 72)
                self.assertTrue((output / "best.zip").is_file())
                self.assertTrue((output / "final.zip").is_file())
                if method == "hard":
                    self.assertEqual(summary["steps_by_difficulty"], {"hard": 72})
                if method in ("fixed", "shuffled"):
                    self.assertEqual(summary["steps_by_difficulty"],
                                     {"easy": 24, "medium": 24, "hard": 24})
                if method == "plr":
                    rows = [json.loads(line) for line in
                            (output / "plr_episodes.jsonl").read_text().splitlines()]
                    self.assertTrue(any(row["score"] is not None for row in rows))
                    self.assertTrue(any(row["selection"] == "replay" for row in rows))

    def test_progress_methods_accept_feedback_from_real_rollouts(self):
        root = ROOT / "runs"
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as folder:
            output = Path(folder)
            config = read_json(ROOT / "configs" / "smoke.json")
            config["budget"]["block_steps"] = 42
            catalog = make_catalog(**config["scenarios"])
            build_profiles(catalog["train"], config["environment"], (12, 19, 26), output)
            profiles = read_json(output / "opportunities.json")
            for method in PROGRESS_METHODS:
                with self.subTest(method=method):
                    run = output / method
                    run.mkdir()
                    config["method"] = method
                    train(config, catalog, 42, run, profiles)
                    self.assertEqual(read_json(run / "training_summary.json")["steps"], 252)
                    rows = [json.loads(line) for line in
                            (run / "curriculum_episodes.jsonl").read_text().splitlines()]
                    self.assertTrue(any(row["used_for_progress"] for row in rows))
                    self.assertTrue(any(not row["warmup"] for row in rows))
                    self.assertEqual("density_progress" in rows[0], method == BACKOFF_METHOD)
