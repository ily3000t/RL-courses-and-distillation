from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from rl_courses.records import ROOT
from rl_courses.warm_start import file_hash, initialize_actor


class ActorInitializationTest(unittest.TestCase):
    def test_transfer_keeps_fresh_sac_state_and_random_streams(self):
        class Env(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (11,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        settings = {"policy_kwargs": {"net_arch": [64, 64]}, "device": "cpu", "buffer_size": 16}
        source = SAC("MlpPolicy", Env(), seed=42, **settings)
        with torch.no_grad():
            source.actor.mu.bias.fill_(0.5)
        learner = SAC("MlpPolicy", Env(), seed=43, **settings)
        log_std = learner.actor.log_std.weight.detach().clone()
        critic = next(learner.critic.parameters()).detach().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        root = ROOT / "runs"
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as folder:
            path = Path(folder) / "offline.zip"
            source.save(path)
            initialize_actor(learner, {"checkpoint": str(path), "checkpoint_sha256": file_hash(path)})

        observations = np.eye(11, dtype=np.float32)
        np.testing.assert_array_equal(
            learner.predict(observations, deterministic=True)[0],
            source.predict(observations, deterministic=True)[0],
        )
        torch.testing.assert_close(learner.actor.log_std.weight, log_std, rtol=0, atol=0)
        torch.testing.assert_close(next(learner.critic.parameters()), critic, rtol=0, atol=0)
        self.assertEqual(len(learner.actor.optimizer.state), 0)
        self.assertEqual(learner.replay_buffer.size(), 0)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
