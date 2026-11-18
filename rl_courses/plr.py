"""Prioritized Level Replay adapted to the SAC critic target."""

import math

import numpy as np
import torch


class PrioritizedSampler:
    block_order = None

    def __init__(self, scenes, seed, *, min_seen_fraction, replay_probability,
                 staleness_weight, temperature):
        if not all(0 <= value <= 1 for value in
                   (min_seen_fraction, replay_probability, staleness_weight)):
            raise ValueError("PLR fractions and probabilities must be in [0, 1].")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("PLR temperature must be finite and positive.")
        self.scenes = scenes
        self.rng = np.random.default_rng(seed)
        self.minimum_seen = math.ceil(len(scenes) * min_seen_fraction)
        self.replay_probability = replay_probability
        self.staleness_weight = staleness_weight
        self.temperature = temperature
        self.last_seen = {}
        self.scores = {}

    def _weights(self, ranked, step):
        ranks = np.arange(1, len(ranked) + 1, dtype=np.float64)
        priority = np.exp(-np.log(ranks) / self.temperature)
        priority /= priority.sum()
        ages = np.array([max(1, step - self.last_seen[scene.scene_id]) for scene in ranked])
        ages = ages / ages.sum()
        return (1 - self.staleness_weight) * priority + self.staleness_weight * ages

    def pick(self, step):
        new = [scene for scene in self.scenes if scene.scene_id not in self.last_seen]
        replay = self.scores and (
            not new or (len(self.last_seen) >= self.minimum_seen
                        and self.rng.random() < self.replay_probability)
        )
        if replay:
            ranked = [scene for scene in self.scenes if scene.scene_id in self.scores]
            ranked.sort(key=lambda scene: self.scores[scene.scene_id], reverse=True)
            probabilities = self._weights(ranked, step)
            index = int(self.rng.choice(len(ranked), p=probabilities))
            scene = ranked[index]
            self.selection = "replay"
        else:
            candidates = new or self.scenes
            scene = candidates[int(self.rng.integers(len(candidates)))]
            self.selection = "new" if new else "warmup"
        self.last_seen[scene.scene_id] = step
        return scene

    def update(self, scene_id, score):
        if not math.isfinite(score):
            raise ValueError(f"Non-finite critic error for {scene_id}.")
        self.scores[scene_id] = score


@torch.no_grad()
def sac_td_score(model, transitions, seed):
    observations, actions, rewards, next_observations, terminated = zip(*transitions)
    device = model.device
    observations = torch.as_tensor(np.stack(observations), device=device)
    actions = torch.as_tensor(np.stack(actions), device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(np.stack(next_observations), device=device)
    bootstrap = ~torch.as_tensor(terminated, dtype=torch.bool, device=device)
    # The extra target-action samples should not advance SAC's own RNG stream.
    cuda_index = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    devices = [] if cuda_index is None else [cuda_index]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_index is not None:
            torch.cuda.default_generators[cuda_index].manual_seed(seed)
        next_actions, log_prob = model.actor.action_log_prob(next_observations)
    next_q = torch.cat(model.critic_target(next_observations, next_actions), dim=1).min(dim=1).values
    entropy = (model.log_ent_coef.detach().exp() if model.log_ent_coef is not None
               else model.ent_coef_tensor)
    target = rewards + bootstrap * model.gamma * (next_q - entropy * log_prob.flatten())
    predicted = torch.cat(model.critic(observations, actions), dim=1)
    return float((predicted - target[:, None]).abs().mean().item())
