"""Export and benchmark the deterministic deployment actor on one CPU thread."""

from copy import deepcopy
from hashlib import sha256
import platform
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn

from .records import write_json


def export_actor(model, output):
    actor = deepcopy(model.actor).cpu().eval()
    deployment = nn.Sequential(actor.features_extractor, actor.latent_pi, actor.mu, nn.Tanh()).eval()
    observation = torch.zeros((1, *model.observation_space.shape), dtype=torch.float32)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            exported = torch.jit.freeze(torch.jit.trace(deployment, observation))
            checks = torch.rand((8, *model.observation_space.shape), generator=torch.Generator().manual_seed(0)) * 2 - 1
            torch.testing.assert_close(exported(checks), actor(checks, deterministic=True))
            path = output / "actor.pt"
            exported.save(str(path))
            for _ in range(100):
                exported(observation)
            latencies = []
            for _ in range(1000):
                start = perf_counter_ns()
                exported(observation)
                latencies.append((perf_counter_ns() - start) / 1000)
    finally:
        torch.set_num_threads(threads)
    write_json(output / "deployment.json", {
        "actor_parameters": sum(parameter.numel() for parameter in deployment.parameters()),
        "file_bytes": path.stat().st_size, "format": "torchscript_fp32", "device": "cpu",
        "actor_sha256": sha256(path.read_bytes()).hexdigest(), "cpu_model": platform.processor(),
        "cpu_threads": 1, "batch_size": 1, "timed_calls": len(latencies),
        "latency_us_p50": float(np.median(latencies)), "latency_us_p95": float(np.percentile(latencies, 95)),
    })
