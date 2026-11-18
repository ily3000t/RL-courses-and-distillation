"""Reproducible traffic inputs for the merge network."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DEMAND = {"easy": 450, "medium": 850, "hard": 1350}


@dataclass(frozen=True)
class Scene:
    scene_id: str
    difficulty: str
    seed: int
    demand_vph: int

    def __post_init__(self):
        if self.difficulty not in DEMAND or self.demand_vph <= 0:
            raise ValueError(f"Invalid scene: {self}")
        if not 0 <= self.seed < 2**31:
            raise ValueError("Scene seed must fit SUMO's nonnegative signed integer range.")


def write_routes(scene: Scene, path: Path, warmup: float, horizon: float) -> None:
    rng = np.random.default_rng(scene.seed)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="traffic", length="5", sigma="0.5", maxSpeed="31")
    ET.SubElement(root, "vType", id="agent", length="5", sigma="0", maxSpeed="31",
                  accel="3", decel="5")
    ET.SubElement(root, "route", id="through", edges="main merge out")
    ET.SubElement(root, "route", id="onramp", edges="ramp merge out")

    departures = [(warmup, "ego", "agent", "onramp", 19.0)]
    time = float(rng.exponential(3600 / scene.demand_vph))
    while time < warmup + horizon + 5:
        speed = float(rng.uniform(20, 29))
        departures.append((time, f"traffic-{len(departures)}", "traffic", "through", speed))
        time += float(rng.exponential(3600 / scene.demand_vph))
    for depart, vehicle, kind, route, speed in sorted(departures):
        ET.SubElement(root, "vehicle", id=vehicle, type=kind, route=route,
                      depart=f"{depart:.3f}", departSpeed=f"{speed:.2f}")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
