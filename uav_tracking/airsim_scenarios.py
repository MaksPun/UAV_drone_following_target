"""Scripted target motion profiles used by repeatable AirSim metric runs."""

from __future__ import annotations

import math
from dataclasses import dataclass


SCENARIO_NAMES = (
    "manual",
    "static",
    "lateral_sine",
    "depth_step",
    "vertical_step",
    "zigzag",
    "frame_exit_right",
    "frame_exit_left",
    "frame_exit_top",
    "frame_exit_bottom",
    "frame_exit_top_right",
    "frame_exit_top_left",
    "frame_exit_bottom_right",
    "frame_exit_bottom_left",
    "occlusion_like",
)


@dataclass
class CubeCmd:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class ScenarioState:
    """Stateful scenario generator used by metric-suite scripts."""

    name: str = "manual"
    started_at: float | None = None
    enabled: bool = False
    speed_scale: float = 1.0

    def start(self, now: float):
        self.started_at = now
        self.enabled = self.name != "manual"

    def elapsed(self, now: float) -> float:
        if self.started_at is None:
            self.start(now)
        return max(0.0, now - float(self.started_at))

    def step(self, now: float) -> CubeCmd:
        if not self.enabled:
            return CubeCmd()
        t = self.elapsed(now)
        return scenario_cmd(self.name, t, self.speed_scale)


def _scale(cmd: CubeCmd, speed_scale: float) -> CubeCmd:
    return CubeCmd(cmd.vx * speed_scale, cmd.vy * speed_scale, cmd.vz * speed_scale)


def scenario_cmd(name: str, t: float, speed_scale: float = 1.0) -> CubeCmd:
    if name == "static":
        return CubeCmd()
    if name == "lateral_sine":
        return _scale(CubeCmd(vx=0.35, vy=1.7 * math.sin(0.85 * t), vz=0.0), speed_scale)
    if name == "depth_step":
        return _scale(CubeCmd(vx=1.35 if t < 7.0 else -1.15, vy=0.0, vz=0.0), speed_scale)
    if name == "vertical_step":
        return _scale(CubeCmd(vx=0.25, vy=0.0, vz=-0.75 if t < 6.0 else 0.75), speed_scale)
    if name == "zigzag":
        return _scale(CubeCmd(
            vx=0.70,
            vy=2.0 if int(t / 1.6) % 2 == 0 else -2.0,
            vz=-0.45 if int(t / 2.8) % 2 == 0 else 0.45,
        ), speed_scale)
    if name == "frame_exit_right":
        return _scale(CubeCmd(vx=0.35, vy=2.25 if t < 5.0 else -1.35, vz=0.0), speed_scale)
    if name == "frame_exit_left":
        return _scale(CubeCmd(vx=0.35, vy=-2.25 if t < 5.0 else 1.35, vz=0.0), speed_scale)
    if name == "frame_exit_top":
        return _scale(CubeCmd(vx=0.25, vy=0.0, vz=-1.25 if t < 4.5 else 0.75), speed_scale)
    if name == "frame_exit_bottom":
        return _scale(CubeCmd(vx=0.25, vy=0.0, vz=1.10 if t < 4.5 else -0.75), speed_scale)
    if name == "frame_exit_top_right":
        return _scale(CubeCmd(vx=0.25, vy=2.0 if t < 4.8 else -1.0, vz=-1.05 if t < 4.8 else 0.65), speed_scale)
    if name == "frame_exit_top_left":
        return _scale(CubeCmd(vx=0.25, vy=-2.0 if t < 4.8 else 1.0, vz=-1.05 if t < 4.8 else 0.65), speed_scale)
    if name == "frame_exit_bottom_right":
        return _scale(CubeCmd(vx=0.25, vy=2.0 if t < 4.8 else -1.0, vz=0.95 if t < 4.8 else -0.65), speed_scale)
    if name == "frame_exit_bottom_left":
        return _scale(CubeCmd(vx=0.25, vy=-2.0 if t < 4.8 else 1.0, vz=0.95 if t < 4.8 else -0.65), speed_scale)
    if name == "occlusion_like":
        if 3.0 <= t <= 5.2:
            return _scale(CubeCmd(vx=0.0, vy=3.1, vz=0.0), speed_scale)
        if 5.2 < t <= 8.2:
            return _scale(CubeCmd(vx=0.0, vy=-2.7, vz=0.0), speed_scale)
        return _scale(CubeCmd(vx=0.45, vy=0.35 * math.sin(1.1 * t), vz=0.0), speed_scale)
    return CubeCmd()
