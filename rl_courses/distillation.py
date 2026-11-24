"""Train fixed-size SAC students with hard actions or Gaussian policy targets."""

from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import SAC
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

from .correction_targets import load_correction_targets
from .environment import EnvConfig, MergeEnv
from .records import write_json
from .training import CheckpointEvaluation


STUDENT_METHODS = ("scratch", "bc", "soft")
CORRECTION_SUPERVISION = ("both", "acceleration", "acceleration-equal")


def student_config(config, settings, method, device):
    if set(settings) != {"net_arch", "updates", "batch_size", "validate_every"}:
        raise ValueError("Student config requires net_arch, updates, batch_size and validate_every.")
    architecture = settings["net_arch"]
    if (not isinstance(architecture, list) or not architecture
            or any(type(width) is not int or width < 1 for width in architecture)):
        raise ValueError("Use a nonempty list of positive integer student layer widths.")
    if any(type(settings[key]) is not int or settings[key] < 1
           for key in ("updates", "batch_size", "validate_every")):
        raise ValueError("Student optimization settings must be positive integers.")
    return {**config, "method": method, "distillation": settings,
            "sac": {**config["sac"], "device": device or config["sac"]["device"],
                    "policy_kwargs": {**config["sac"]["policy_kwargs"], "net_arch": architecture}}}


def load_teacher_targets(dataset, observation_shape, action_shape, expected_rows, steps_per_scene):
    with np.load(dataset, allow_pickle=False) as data:
        indices = data["scene_indices"]
        scene_count = expected_rows // steps_per_scene
        if not np.array_equal(indices, np.repeat(np.arange(scene_count), steps_per_scene)):
            raise ValueError("Teacher data do not match the fixed per-scene collection quota.")
        columns = []
        for key, shape in (("observations", observation_shape), ("actions", action_shape),
                           ("means", action_shape), ("log_stds", action_shape)):
            values = data[key]
            if values.shape != (expected_rows, *shape) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid teacher array: {key}.")
            columns.append(values)
        return columns


def gaussian_policy_loss(teacher_mean, teacher_log_std, student_mean, student_log_std):
    teacher = Normal(teacher_mean, teacher_log_std.exp())
    student = Normal(student_mean, student_log_std.exp())
    return kl_divergence(teacher, student).sum(dim=-1).mean()


def correction_bc_loss(actions, targets, ordinary_rows, supervision):
    error = (actions - targets).square()
    if supervision == "both":
        return error.mean()
    correction_error = error[ordinary_rows:, 0]
    if supervision == "acceleration-equal":
        correction_error = correction_error / error.shape[-1]
    return torch.cat((error[:ordinary_rows].mean(dim=-1), correction_error)).mean()


def train_offline(config, catalog, seed, output, dataset, steps_per_scene, correction_directory=None):
    start = perf_counter()
    settings = config["distillation"]
    env = MergeEnv(output, EnvConfig(**config["environment"]))
    try:
        model = SAC("MlpPolicy", env, seed=seed, **config["sac"])
        if model.actor.use_sde:
            raise ValueError("Distillation requires a diagonal Gaussian SAC actor without gSDE.")
        targets = load_teacher_targets(dataset, env.observation_space.shape, env.action_space.shape,
                                       len(catalog["train"]) * steps_per_scene, steps_per_scene)
        observations, actions, means, log_stds = [torch.as_tensor(array, device=model.device) for array in targets]
        correction = config.get("correction")
        correction_batch = correction["batch_size"] if correction else 0
        ordinary_batch = settings["batch_size"] - correction_batch
        if correction:
            verified = load_correction_targets(correction_directory, env.observation_space.shape, env.action_space.shape)
            corrected_obs, corrected_actions = [torch.as_tensor(array, device=model.device) for array in verified]
        generator = torch.Generator(device=model.device).manual_seed(seed)
        optimizer = model.actor.optimizer
        validation = CheckpointEvaluation(env, catalog["validation"], settings["validate_every"], output)
        updates = settings["updates"]
        for update in range(1, updates + 1):
            model.actor.train()
            indices = torch.randint(len(observations), (ordinary_batch,),
                                    device=model.device, generator=generator)
            obs_batch = observations[indices]
            if correction:
                selected = torch.randint(len(corrected_obs), (correction_batch,),
                                         device=model.device, generator=generator)
                obs_batch = torch.cat((obs_batch, corrected_obs[selected]))
            mean, log_std, _ = model.actor.get_action_dist_params(obs_batch)
            if correction:
                labels = torch.cat((actions[indices], corrected_actions[selected]))
                loss = correction_bc_loss(mean.tanh(), labels, ordinary_batch, correction["supervision"])
            elif config["method"] == "bc":
                loss = F.mse_loss(mean.tanh(), actions[indices])
            else:
                loss = gaussian_policy_loss(means[indices], log_stds[indices], mean, log_std)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite student loss at optimizer step {update}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            if update % settings["validate_every"] == 0 or update == updates:
                print(f"update {update}: {config['method']} loss {float(loss.detach()):.6g}", flush=True)
                validation.checkpoint(model, update)
        model.save(output / "final")
        summary = {
            "steps": 0, "optimizer_updates": updates, "batch_size": settings["batch_size"],
            "dataset_rows": len(observations), "validation_decision_steps": validation.decision_steps,
            "wall_seconds": perf_counter() - start,
        }
        if correction:
            summary.update(correction_dataset_rows=len(corrected_obs), correction_batch_size=correction_batch,
                           correction_draws=updates * correction_batch, correction_supervision=correction["supervision"])
        write_json(output / "training_summary.json", summary)
    finally:
        env.close()
