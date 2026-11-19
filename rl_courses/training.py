"""Common SAC training loop and experiment logging."""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .environment import EnvConfig, MergeEnv
from .evaluation import evaluate, validation_key
from .opportunities import opportunity_groups
from .plr import PrioritizedSampler, sac_td_score
from .progress import ProgressSampler
from .records import append_json, write_json
from .sampling import BaselineSampler, PROGRESS_METHODS
from .scenarios import DEMAND


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
        self.score_episode = None
        self.observe_episode = None

    def reset(self, *, seed=None, options=None):
        self.scene = self.pick_scene(self.decisions)
        self.episode_decisions = 0
        self.episode_return = 0.0
        observation, info = self.env.reset(seed=seed, options={"scene": self.scene})
        if self.score_episode is not None:
            self._last_observation = observation
            self._transitions = []
        return observation, info

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
        if self.score_episode is not None:
            self._transitions.append((self._last_observation, np.asarray(action, dtype=np.float32),
                                      reward, observation, terminated))
            self._last_observation = observation
        if terminated or truncated:
            append_json(self.output / "training_episodes.jsonl", {
                **info, "global_step": self.decisions, "episode_steps": self.episode_decisions,
                "return": self.episode_return,
            })
            if self.score_episode is not None:
                self.score_episode(self.scene, self._transitions, self.decisions)
            if self.observe_episode is not None:
                self.observe_episode(self.scene, info["outcome"], self.decisions)
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


def train(config, catalog, seed, output, profiles=None):
    budget = config["budget"]
    block_steps = budget["block_steps"]
    blocks = len(DEMAND) * budget["blocks_per_difficulty"]
    if block_steps < 1 or blocks < 1:
        raise ValueError("Training budget must be positive.")
    if config["sac"].get("train_freq") != 1:
        raise ValueError("Use train_freq=1 for an exact decision budget at block boundaries.")
    method = config["method"]
    scenes = catalog["train"]
    progress_groups = None
    if method == "plr":
        sampler = PrioritizedSampler(scenes, seed, **config["plr"])
    elif method in PROGRESS_METHODS:
        if method == "opportunity-progress":
            if profiles is None:
                raise ValueError("Opportunity-progress requires a training profile bundle.")
            progress_groups = opportunity_groups(profiles, scenes, config["environment"])
        else:
            progress_groups = {scene.scene_id: scene.difficulty for scene in scenes}
        sampler = ProgressSampler(scenes, progress_groups, seed, **config["progress"])
    else:
        sampler = BaselineSampler(method, scenes, seed, **budget)
    write_json(output / "schedule.json", {
        "method": config["method"], "block_steps": block_steps, "total_blocks": blocks,
        "block_order": sampler.block_order,
    })
    if progress_groups is not None:
        write_json(output / "curriculum_groups.json", progress_groups)

    environment = EnvConfig(**config["environment"])
    train_env = TrainingEnv(MergeEnv(output, environment), sampler.pick, block_steps, output)
    try:
        validation_env = MergeEnv(output, environment)
        try:
            model = SAC("MlpPolicy", train_env, seed=seed, **config["sac"])
            if method == "plr":
                train_env.score_episode = make_plr_scorer(model, sampler, seed, output)
            if progress_groups is not None:
                train_env.observe_episode = make_progress_recorder(sampler, output)
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


def make_plr_scorer(model, sampler, seed, output):
    def score_episode(scene, transitions, step):
        score = None
        if model._n_updates > 0:
            score = sac_td_score(model, transitions, seed + step)
            sampler.update(scene.scene_id, score)
        append_json(output / "plr_episodes.jsonl", {
            "step": step, "scene_id": scene.scene_id, "score": score,
            "selection": sampler.selection, "scored_scenes": len(sampler.scores),
        })
    return score_episode


def make_progress_recorder(sampler, output):
    def observe(scene, outcome, step):
        append_json(output / "curriculum_episodes.jsonl", {
            "step": step, "scene_id": scene.scene_id, "outcome": outcome,
            "used_for_progress": outcome != "schedule_cut", **sampler.selection,
        })
        sampler.observe(scene, outcome)
    return observe


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
