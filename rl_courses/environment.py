"""Gymnasium environment for driving a single vehicle onto a SUMO highway."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import traci

from .scenarios import DEMAND, Scene, write_routes


EGO = "ego"
MERGE_START_X = 600.0
MERGE_END_X = 850.0
OBSERVATION_FIELDS = (
    "speed", "acceleration", "distance_to_merge_start", "distance_to_merge_end",
    "in_target_lane", "front_gap", "front_relative_speed", "front_present",
    "rear_gap", "rear_relative_speed", "rear_present",
)


@dataclass(frozen=True)
class EnvConfig:
    sim_step: float = 0.1
    steps_per_action: int = 5
    warmup_seconds: float = 25.0
    max_decisions: int = 160

    def __post_init__(self):
        if min(self.sim_step, self.steps_per_action, self.max_decisions) <= 0:
            raise ValueError("Simulation step and decision limits must be positive.")
        if self.warmup_seconds < 0:
            raise ValueError("Warm-up time must be nonnegative.")


class MergeEnv(gym.Env):
    """The first action is acceleration. A positive second action requests merging."""

    metadata = {"render_modes": []}

    def __init__(self, work_dir: Path, config: EnvConfig = EnvConfig()):
        self.config = config
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(len(OBSERVATION_FIELDS),), dtype=np.float32
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        self._temp = TemporaryDirectory(prefix="sumo-", dir=work_dir)
        self._folder = Path(self._temp.name).resolve()
        self._connection = None
        try:
            self._sumo = self._binary("sumo")
            self._network = self._build_network()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _binary(name):
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(f"{name} is missing from PATH. Add your SUMO bin directory.")
        return executable

    def _build_network(self):
        inputs = Path(__file__).resolve().parents[1] / "network"
        result = self._folder / "merge.net.xml"
        subprocess.run([
            self._binary("netconvert"),
            "--node-files", str(inputs / "merge.nod.xml"),
            "--edge-files", str(inputs / "merge.edg.xml"),
            "--connection-files", str(inputs / "merge.con.xml"),
            "--output-file", str(result),
        ], check=True, capture_output=True, text=True)
        return result

    def _stop_simulation(self):
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scene = (options or {}).get("scene")
        if scene is None:
            scene_seed = int(self.np_random.integers(2**31))
            scene = Scene(f"hard-{scene_seed}", "hard", scene_seed, DEMAND["hard"])
        self._scene = scene
        self._decisions = 0
        self._merged = False
        self._episode_done = False
        config = self.config
        route_file = self._folder / "traffic.rou.xml"
        write_routes(scene, route_file, config.warmup_seconds,
                     config.sim_step * config.steps_per_action * config.max_decisions)
        label = str(self._folder)
        arguments = [
            "--net-file", str(self._network), "--route-files", str(route_file),
            "--seed", str(scene.seed), "--step-length", str(config.sim_step),
            "--collision.action", "warn", "--collision.check-junctions", "true",
            "--time-to-teleport", "-1", "--no-step-log", "--duration-log.disable",
        ]
        try:
            if self._connection is None:
                traci.start([self._sumo, *arguments], label=label, doSwitch=False, numRetries=3)
                self._connection = traci.getConnection(label)
            else:
                self._connection.load(arguments)
            for _ in range(round((config.warmup_seconds + 10) / config.sim_step) + 1):
                self._connection.simulationStep()
                if EGO in self._connection.simulation.getDepartedIDList():
                    # Keep SUMO's safe-speed checks and enforce safe gaps for explicit lane requests.
                    self._connection.vehicle.setLaneChangeMode(EGO, 512)
                    self._episode_start = self._connection.simulation.getTime()
                    return self._observation(), self._info()
            raise RuntimeError(f"Ego insertion timed out in scene {scene.scene_id}.")
        except BaseException:
            self._stop_simulation()
            raise

    def _info(self, outcome=None):
        return {
            "scene_id": self._scene.scene_id,
            "difficulty": self._scene.difficulty,
            "scene_seed": self._scene.seed,
            "outcome": outcome,
            "merged": self._merged,
            "simulated_seconds": round(
                self._connection.simulation.getTime() - self._episode_start, 6
            ),
        }

    def _observation(self):
        vehicles = self._connection.vehicle
        x, _ = vehicles.getPosition(EGO)
        speed = vehicles.getSpeed(EGO)
        front_gap, rear_gap = 150.0, 150.0
        front_relative, rear_relative = 0.0, 0.0
        front_present, rear_present = False, False
        for neighbor in vehicles.getIDList():
            if neighbor == EGO:
                continue
            lane = vehicles.getLaneID(neighbor)
            if lane not in ("main_0", "merge_1", "out_0") and not lane.startswith(":"):
                continue
            offset = vehicles.getPosition(neighbor)[0] - x
            if 0 <= offset < front_gap:
                front_gap = offset
                front_relative = vehicles.getSpeed(neighbor) - speed
                front_present = True
            if -rear_gap < offset < 0:
                rear_gap = -offset
                rear_relative = vehicles.getSpeed(neighbor) - speed
                rear_present = True
        values = [
            speed / 31.0, vehicles.getAcceleration(EGO) / 8.0,
            (MERGE_START_X - x) / 300.0, (MERGE_END_X - x) / 300.0,
            float(self._merged), front_gap / 150.0, front_relative / 31.0,
            float(front_present), rear_gap / 150.0, rear_relative / 31.0, float(rear_present),
        ]
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    def step(self, action):
        if self._connection is None or self._episode_done:
            raise RuntimeError("Call reset before stepping.")
        acceleration, merge_intent = np.clip(action, -1.0, 1.0)
        vehicles = self._connection.vehicle
        start_x = vehicles.getPosition(EGO)[0]
        delta_t = self.config.steps_per_action * self.config.sim_step
        vehicles.setAcceleration(EGO, float(acceleration * (3 if acceleration >= 0 else 5)), delta_t)
        outcome = None
        end_x = start_x
        for _ in range(self.config.steps_per_action):
            if merge_intent > 0 and vehicles.getLaneID(EGO) == "merge_0":
                vehicles.changeLane(EGO, 1, self.config.sim_step)
            self._connection.simulationStep()
            if EGO in self._connection.simulation.getCollidingVehiclesIDList():
                outcome = "collision"
                break
            if EGO in self._connection.simulation.getArrivedIDList():
                outcome = "success" if self._merged else "missed_merge"
                break
            if EGO not in vehicles.getIDList():
                raise RuntimeError(f"Ego disappeared from scene {self._scene.scene_id}.")
            end_x = vehicles.getPosition(EGO)[0]
            lane = vehicles.getLaneID(EGO)
            self._merged |= lane == "merge_1"
            if not self._merged and lane == "merge_0":
                remaining = self._connection.lane.getLength(lane) - vehicles.getLanePosition(EGO)
                if remaining <= 3.0:
                    outcome = "missed_merge"
                    break
            if self._merged and lane == "out_0" and end_x >= MERGE_END_X + 50:
                outcome = "success"
                break

        self._decisions += 1
        terminated = outcome is not None
        truncated = not terminated and self._decisions >= self.config.max_decisions
        if truncated:
            outcome = "timeout"
        self._episode_done = terminated or truncated
        if EGO in vehicles.getIDList():
            observation = self._observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        terminal_reward = {"success": 25, "collision": -30, "missed_merge": -20, "timeout": -15}
        reward = (end_x - start_x) / 25.0 - 0.1 + terminal_reward.get(outcome, 0)
        return observation, float(reward), terminated, truncated, self._info(outcome)

    def close(self):
        try:
            self._stop_simulation()
        finally:
            self._temp.cleanup()
