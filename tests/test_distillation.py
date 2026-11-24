from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import torch
from stable_baselines3 import SAC

from rl_courses.actor_export import export_actor
from rl_courses.distillation import correction_bc_loss, gaussian_policy_loss, student_config
from rl_courses.records import ROOT, read_json


class DistillationTest(unittest.TestCase):
    def test_acceleration_only_masks_the_correction_intent_gradient_and_keeps_row_weight(self):
        predicted = torch.tensor([[2.0, 2.0], [1.0, 4.0]], requires_grad=True)
        target = torch.zeros_like(predicted)
        loss = correction_bc_loss(predicted, target, 1, "acceleration")
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
        full_loss = correction_bc_loss(predicted.detach(), target, 1, "both")
        self.assertEqual(full_loss.item(), 6.25)

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

    def test_export_reloads_with_same_deterministic_actions(self):
        root = ROOT / "runs"
        root.mkdir(exist_ok=True)
        env = gym.make("Pendulum-v1")
        self.addCleanup(env.close)
        model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [8, 8]}, device="cpu")
        with TemporaryDirectory(dir=root) as folder:
            output = Path(folder)
            export_actor(model, output)
            actor = torch.jit.load(str(output / "actor.pt"))
            observation = torch.tensor([[1.0, 0.0, 0.25]])
            torch.testing.assert_close(actor(observation), model.actor(observation, deterministic=True))
            self.assertGreater(read_json(output / "deployment.json")["actor_parameters"], 0)
