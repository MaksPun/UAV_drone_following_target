"""Adaptive command limit manager shared by PID and LQR controllers."""

from __future__ import annotations

from dataclasses import dataclass

import airsim
import numpy as np

from .common import clamp
from .config import CFG
from .vision import Det


@dataclass
class SpeedLimits:
    """Active command limits used by PID, LQR and reacquisition."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0


class AdaptiveSpeedManager:
    """Adapts command limits from target motion and image-edge proximity."""

    def __init__(self):
        self._vx = CFG.base_vx
        self._vy = CFG.base_vy
        self._vz = CFG.base_vz
        self._yaw = CFG.base_yaw_dps

        self._tvx = CFG.base_vx
        self._tvy = CFG.base_vy
        self._tvz = CFG.base_vz
        self._tyaw = CFG.base_yaw_dps

        self.cube_speed_ms = 0.0
        self.edge_boost_x = 1.0
        self.edge_boost_y = 1.0
        self.edge_boost_z = 1.0
        self.speed_scale = 0.0

    def reset(self):
        self._vx = CFG.base_vx
        self._vy = CFG.base_vy
        self._vz = CFG.base_vz
        self._yaw = CFG.base_yaw_dps

    def update(self, kalman: "KalmanCube", det: Det | None, fshape, drone_pose: airsim.Pose) -> SpeedLimits:
        """Call once per frame before controller.step()."""

        cube_spd = 0.0
        cube_acc = 0.0
        if kalman.ok:
            cube_spd = float(np.linalg.norm(kalman.vel))
            cube_acc = float(np.linalg.norm(kalman.acc))

        self.cube_speed_ms = cube_spd
        # A single scale combines target speed and acceleration.
        spd_scale = min(
            1.0,
            cube_spd / max(CFG.cube_v_ref, 0.1) * 0.70
            + cube_acc / max(CFG.cube_a_ref, 0.1) * 0.30,
        )
        self.speed_scale = spd_scale

        tvx = CFG.base_vx + (CFG.max_vx - CFG.base_vx) * spd_scale
        tvy = CFG.base_vy + (CFG.max_vy - CFG.base_vy) * spd_scale
        tvz = CFG.base_vz + (CFG.max_vz - CFG.base_vz) * spd_scale
        tyaw = CFG.base_yaw_dps + (CFG.max_yaw_dps - CFG.base_yaw_dps) * spd_scale

        bx = by = bz = 1.0
        if det is not None and fshape is not None:
            h, w = fshape[:2]
            exi = (det.cx - w * 0.5) / (w * 0.5)
            eyi = (det.cy - h * 0.5) / (h * 0.5)

            # 0 near center, 1 near image edge.
            ex_prox = max(0.0, (abs(exi) - CFG.edge_thr) / (1.0 - CFG.edge_thr + 1e-6))
            ey_prox = max(0.0, (abs(eyi) - CFG.edge_thr) / (1.0 - CFG.edge_thr + 1e-6))
            ex_prox = min(1.0, ex_prox)
            ey_prox = min(1.0, ey_prox)

            bx = 1.0 + (CFG.edge_vx_boost - 1.0) * ex_prox
            by = 1.0 + (CFG.edge_vy_boost - 1.0) * ex_prox
            bz = 1.0 + (CFG.edge_vz_boost - 1.0) * ey_prox
            byaw = 1.0 + (CFG.edge_yaw_boost - 1.0) * ex_prox

            tvx *= bx
            tvy *= by
            tvz *= bz
            tyaw *= byaw

        self.edge_boost_x = bx
        self.edge_boost_y = by
        self.edge_boost_z = bz

        tvx = min(tvx, CFG.max_vx)
        tvy = min(tvy, CFG.max_vy)
        tvz = min(tvz, CFG.max_vz)
        tyaw = min(tyaw, CFG.max_yaw_dps)

        self._tvx = tvx
        self._tvy = tvy
        self._tvz = tvz
        self._tyaw = tyaw

        def smooth(cur, tgt):
            alpha = CFG.adapt_alpha_up if tgt > cur else CFG.adapt_alpha_dn
            return cur + alpha * (tgt - cur)

        self._vx = smooth(self._vx, tvx)
        self._vy = smooth(self._vy, tvy)
        self._vz = smooth(self._vz, tvz)
        self._yaw = smooth(self._yaw, tyaw)

        return SpeedLimits(vx=self._vx, vy=self._vy, vz=self._vz, yaw=self._yaw)

    @property
    def current(self) -> SpeedLimits:
        return SpeedLimits(vx=self._vx, vy=self._vy, vz=self._vz, yaw=self._yaw)
