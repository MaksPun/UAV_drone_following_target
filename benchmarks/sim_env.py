"""Small deterministic simulation helpers used by offline benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace


def vec3(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x_val=float(x), y_val=float(y), z_val=float(z))


def quat_from_yaw(yaw_rad: float):
    return SimpleNamespace(
        w_val=math.cos(yaw_rad * 0.5),
        x_val=0.0,
        y_val=0.0,
        z_val=math.sin(yaw_rad * 0.5),
    )


def pose(x=0.0, y=0.0, z=0.0, yaw_rad=0.0):
    return SimpleNamespace(position=vec3(x, y, z), orientation=quat_from_yaw(yaw_rad))


@dataclass
class SimState:
    drone: object
    target: object
    yaw_rad: float = 0.0

    @classmethod
    def create(cls, drone_xyz=(0.0, 0.0, -3.0), target_xyz=(20.0, 0.0, -3.0)):
        return cls(drone=pose(*drone_xyz), target=pose(*target_xyz), yaw_rad=0.0)

    def step_drone_body(self, cmd, dt: float):
        cy = math.cos(self.yaw_rad)
        sy = math.sin(self.yaw_rad)
        vx_w = cy * cmd.vx - sy * cmd.vy
        vy_w = sy * cmd.vx + cy * cmd.vy
        self.drone.position.x_val += vx_w * dt
        self.drone.position.y_val += vy_w * dt
        self.drone.position.z_val += cmd.vz * dt
        self.yaw_rad += math.radians(cmd.yaw) * dt
        self.drone.orientation = quat_from_yaw(self.yaw_rad)

    def move_target_world(self, vx: float, vy: float, vz: float, dt: float):
        self.target.position.x_val += vx * dt
        self.target.position.y_val += vy * dt
        self.target.position.z_val += vz * dt


class DummyKalman:
    def __init__(self):
        self.ok = True
        self._vel = [0.0, 0.0, 0.0]
        self._acc = [0.0, 0.0, 0.0]
        self._last = None
        self._last_t = None

    def update(self, target_pose, t):
        p = target_pose.position
        cur = [float(p.x_val), float(p.y_val), float(p.z_val)]
        if self._last is not None:
            dt = max(1e-3, t - self._last_t)
            new_vel = [(cur[i] - self._last[i]) / dt for i in range(3)]
            self._acc = [(new_vel[i] - self._vel[i]) / dt for i in range(3)]
            self._vel = new_vel
        self._last = cur
        self._last_t = t

    @property
    def vel(self):
        import numpy as np

        return np.array(self._vel, dtype=float)

    @property
    def acc(self):
        import numpy as np

        return np.array(self._acc, dtype=float)

    def predict(self, ahead):
        import numpy as np

        if self._last is None:
            return None, None
        p = np.array(self._last, dtype=float)
        v = self.vel
        a = self.acc
        return p + v * ahead + 0.5 * a * ahead * ahead, v + a * ahead


def body_relative(drone_pose, target_pose):
    from uav_tracking.common import world_to_body

    dp = drone_pose.position
    tp = target_pose.position
    return world_to_body(
        drone_pose,
        float(tp.x_val - dp.x_val),
        float(tp.y_val - dp.y_val),
        float(tp.z_val - dp.z_val),
    )


def project_detection(drone_pose, target_pose, frame=(720, 1280), h_fov_deg=90.0):
    from uav_tracking.vision import Det

    h, w = frame
    bx, by, bz = body_relative(drone_pose, target_pose)
    if bx <= 0.5:
        return None, (bx, by, bz), (999.0, 999.0)
    htan = math.tan(math.radians(h_fov_deg) * 0.5)
    vtan = htan * h / w
    x_norm = by / max(1e-6, bx * htan)
    y_norm = bz / max(1e-6, bx * vtan)
    if abs(x_norm) > 1.1 or abs(y_norm) > 1.1:
        return None, (bx, by, bz), (x_norm, y_norm)
    cx = w * (0.5 + 0.5 * x_norm)
    cy = h * (0.5 + 0.5 * y_norm)
    size = max(18.0, min(180.0, 850.0 / max(bx, 1.0)))
    det = Det((int(cx - size), int(cy - size), int(cx + size), int(cy + size)), 0.90, 1)
    return det, (bx, by, bz), (x_norm, y_norm)

