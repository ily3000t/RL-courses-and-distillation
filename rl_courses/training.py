"""Common SAC training loop and experiment logging."""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .catalog import group_by_difficulty
from .environment import EnvConfig, MergeEnv
from .evaluation import evaluate, validation_key
from .records import append_json, write_json


class TrainingEnv(gym.Wrapper):
    def __init__(self, env, pick_scene, block_steps, output):
        super().__init__(env)
        self.pick_scene = pick_scene
        self.block_steps = block_steps
        self.output = output
        self.decisions = 0
        self.episode_decisions = 0
        self.episode_return = 0.0
        self.steps_by_difficulty = {}

    def reset(self, *, seed=None, options=None):
        self.scene = self.pick_scene(self.decisions)
        self.episode_decisions = 0
        self.episode_return = 0.0
        return self.env.reset(seed=seed, options={"scene": self.scene})

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.decisions += 1
        self.episode_decisions += 1
        self.episode_return += reward
        difficulty = self.scene.difficulty
        self.steps_by_difficulty[difficulty] = self.steps_by_difficulty.get(difficulty, 0) + 1
        if not (terminated or truncated) and self.decisions % self.block_steps == 0:
            truncated = True
            info = {**info, "outcome": "schedule_cut"}
        if terminated or truncated:
            append_json(self.output / "training_episodes.jsonl", {
                **info, "global_step": self.decisions, "episode_steps": self.episode_decisions,
                "return": self.episode_return,
            })
        return observation, reward, terminated, truncated, info


class CheckpointEvaluation:
    def __init__(self, env, scenes, block_steps, output):
        self.validation_env = env
        self.scenes = scenes
        self.block_steps = block_steps
        self.output = output
        self.best_key = None

    def checkpoint(self, model, step):
        summary, _ = evaluate(model, self.validation_env, self.scenes)
        append_json(self.output / "validation.jsonl", {"step": step, "summary": summary})
        key = validation_key(summary)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            model.save(self.output / "best")
            write_json(self.output / "best.json", {"step": step, "validation": summary})
        print(f"step {step}: Hard validation {summary['hard']}", flush=True)


def train(config, catalog, seed, output):
    block_steps = config["budget"]["block_steps"]
    blocks = 3 * config["budget"]["blocks_per_difficulty"]
    if block_steps < 1 or blocks < 1:
        raise ValueError("Training budget must be positive.")
    if config["sac"].get("train_freq") != 1:
        raise ValueError("Use train_freq=1 for an exact decision budget at block boundaries.")
    rng = np.random.default_rng(seed)
    hard_scenes = group_by_difficulty(catalog["train"])["hard"]

    def pick_scene(_step):
        return hard_scenes[int(rng.integers(len(hard_scenes)))]

    environment = EnvConfig(**config["environment"])
    train_env = TrainingEnv(MergeEnv(output, environment), pick_scene, block_steps, output)
    try:
        validation_env = MergeEnv(output, environment)
        try:
            model = SAC("MlpPolicy", train_env, seed=seed, **config["sac"])
            evaluation = CheckpointEvaluation(validation_env, catalog["validation"], block_steps, output)
            # One continuous learn() call keeps progress-based SAC settings comparable.
            callback = BlockCallback(evaluation)
            model.learn(total_timesteps=blocks * block_steps, callback=callback)
            evaluation.checkpoint(model, model.num_timesteps)
            model.save(output / "final")
            write_json(output / "training_summary.json", {
                "steps": model.num_timesteps,
                "steps_by_difficulty": train_env.steps_by_difficulty,
            })
        finally:
            validation_env.close()
    finally:
        train_env.close()


class BlockCallback(BaseCallback):
    def __init__(self, evaluation):
        super().__init__()
        self.evaluation = evaluation

    def _on_rollout_start(self):
        # With train_freq=1, the preceding block has finished updating SAC.
        step = self.model.num_timesteps
        if step > 0 and step % self.evaluation.block_steps == 0:
            self.evaluation.checkpoint(self.model, step)

    def _on_step(self):
        return True
