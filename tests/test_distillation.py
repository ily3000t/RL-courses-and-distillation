import unittest

import torch

from rl_courses.distillation import gaussian_policy_loss, student_config
from rl_courses.records import ROOT, read_json


class DistillationTest(unittest.TestCase):
    def test_gaussian_loss_matches_teacher_to_student_kl(self):
        teacher = torch.tensor([[0.0, 0.0]])
        unit_scale = torch.zeros_like(teacher)
        student = torch.tensor([[1.0, -1.0]], requires_grad=True)
        self.assertEqual(float(gaussian_policy_loss(teacher, unit_scale, teacher, unit_scale)), 0)
        loss = gaussian_policy_loss(teacher, unit_scale, student, unit_scale)
        self.assertEqual(float(loss.detach()), 1)
        loss.backward()
        torch.testing.assert_close(student.grad, student.detach())

    def test_three_methods_share_the_same_actor_specification(self):
        base = read_json(ROOT / "configs/stage1.json")
        settings = read_json(ROOT / "configs/distillation.json")
        for method in ("scratch", "bc", "soft"):
            config = student_config(base, settings, method, "cpu")
            self.assertEqual(config["sac"]["policy_kwargs"]["net_arch"], [64, 64])
            self.assertEqual(config["environment"], base["environment"])
        self.assertEqual(base["sac"]["policy_kwargs"]["net_arch"], [128, 128])
