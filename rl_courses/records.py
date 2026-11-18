"""Experiment provenance and local JSON records."""

from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch

from .environment import OBSERVATION_FIELDS


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, data):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


def new_output(path):
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _command_output(*args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def manifest(config, seed, device, allow_dirty):
    git_status = _command_output("git", "status", "--porcelain=v1", "--untracked-files=all")
    if git_status and not allow_dirty:
        raise ValueError("Commit changes before running an experiment, or use --allow-dirty for a check.")
    sumo = shutil.which("sumo")
    if sumo is None:
        raise FileNotFoundError("sumo is missing from PATH.")
    argv = [sys.executable, *sys.orig_argv[1:]]
    return {
        "git_sha": _command_output("git", "rev-parse", "HEAD"),
        "git_status": git_status,
        "argv": argv,
        "command": subprocess.list2cmdline(argv),
        "cwd": str(Path.cwd()),
        "seed": seed,
        "config": config,
        "observation_fields": list(OBSERVATION_FIELDS),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device,
            "cuda_compiled": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(torch.device(device))
            if torch.device(device).type == "cuda" else None,
            "sumo": _command_output(sumo, "--version").splitlines()[0],
            "sumo_binary": sumo,
            **{name: version(name) for name in
               ("torch", "numpy", "gymnasium", "stable-baselines3", "traci", "sumolib")},
        },
    }
