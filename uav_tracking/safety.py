"""Simple safety checks for altitude and geofence limits."""

from __future__ import annotations

import math

import airsim

from .config import CFG


class SafetyMonitor:
    def __init__(self): self.alt_fault=False; self.fence_fault=False; self._home=None
    def arm_home(self,pose:airsim.Pose):
        p=pose.position; self._home=(float(p.x_val),float(p.y_val))
    def check(self,drone_pose:airsim.Pose)->bool:
        p=drone_pose.position; z=float(p.z_val)
        self.alt_fault=(z>-CFG.alt_floor_m+.2)
        if self._home:
            r=math.hypot(float(p.x_val)-self._home[0],float(p.y_val)-self._home[1])
            self.fence_fault=(r>CFG.geofence_r_m)
        else: self.fence_fault=False
        return not self.fence_fault
    def altitude_correction(self,raw_vz:float,drone_pose:airsim.Pose)->float:
        z=float(drone_pose.position.z_val)
        if z>-CFG.alt_floor_m and raw_vz>0.: return 0.
        if z>-CFG.alt_floor_m+.3:            return min(raw_vz,-.25)
        return raw_vz


#  MAVSDK
