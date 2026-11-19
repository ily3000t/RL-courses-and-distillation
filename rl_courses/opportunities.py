"""Measure reproducible opportunities using fixed approach-speed probes."""

from dataclasses import asdict
import hashlib
import math

from .diagnostics import reference_action, replay
from .environment import EnvConfig, MergeEnv
from .records import ROOT, write_json


DEFAULT_REFERENCE_SPEEDS = (12, 19, 26)


def validate_speeds(speeds):
    if (not speeds or len(set(speeds)) != len(speeds)
            or not all(math.isfinite(speed) and 0 < speed <= 31 for speed in speeds)):
        raise ValueError("Reference speeds must be distinct and in (0, 31] m/s.")


def profile_source_hash():
    files = sorted((ROOT / "network").glob("*.xml"))
    files.extend(ROOT / "rl_courses" / name for name in
                 ("environment.py", "scenarios.py", "diagnostics.py", "opportunities.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def build_profiles(scenes, environment, speeds, output):
    validate_speeds(speeds)
    env = MergeEnv(output, EnvConfig(**environment))
    records = []
    try:
        for scene in scenes:
            probes = []
            for speed in speeds:
                result = replay(env, scene, lambda obs: reference_action(obs, speed))
                probes.append({
                    "speed_mps": speed, "accepted_merge": result["merged"],
                    "outcome": result["outcome"], "simulated_seconds": result["simulated_seconds"],
                })
            records.append({"scene": asdict(scene), "probes": probes})
    finally:
        env.close()
    write_json(output / "opportunities.json", {
        "environment": environment, "reference_speeds_mps": speeds,
        "profile_source_sha256": profile_source_hash(), "records": records,
        "probe_episodes": len(scenes) * len(speeds),
        "probe_simulated_seconds_after_spawn": round(sum(
            probe["simulated_seconds"] for row in records for probe in row["probes"]
        ), 6),
    })


def opportunity_groups(bundle, scenes, environment):
    if bundle["environment"] != environment or bundle["profile_source_sha256"] != profile_source_hash():
        raise ValueError("Opportunity profiles were produced for a different environment or probe implementation.")
    expected = {scene.scene_id: asdict(scene) for scene in scenes}
    speeds = bundle["reference_speeds_mps"]
    validate_speeds(speeds)
    groups = {}
    for row in bundle["records"]:
        scene_id = row["scene"]["scene_id"]
        if scene_id in groups or row["scene"] != expected.get(scene_id):
            raise ValueError("Opportunity profiles must match the training catalog exactly.")
        probes = row["probes"]
        if [probe["speed_mps"] for probe in probes] != speeds or not all(
                isinstance(probe["accepted_merge"], bool) for probe in probes):
            raise ValueError(f"Malformed reference profile for {scene_id}.")
        accepted = sum(probe["accepted_merge"] for probe in probes)
        group = "none-observed" if accepted == 0 else "all" if accepted == len(speeds) else "some"
        groups[scene_id] = f'{row["scene"]["difficulty"]}:{group}'
    if groups.keys() != expected.keys():
        raise ValueError("Opportunity profiles are missing training scenes.")
    return groups
