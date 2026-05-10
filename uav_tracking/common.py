"""Small shared math helpers and command data structures."""

from __future__ import annotations

from dataclasses import dataclass

import airsim
import numpy as np


@dataclass
class Cmd:
    vx:float=0.; vy:float=0.; vz:float=0.; yaw:float=0.

@dataclass
class CubeCmd:
    vx:float=0.; vy:float=0.; vz:float=0.



def sf(x) -> float:
    return float(x.item()) if isinstance(x, np.generic) else float(x)

def clamp(x, lo, hi) -> float:
    return float(max(lo, min(hi, sf(x))))

def clamp3(v, mag):
    return clamp(v, -mag, mag)

def quat_to_R(w, x, y, z) -> np.ndarray:
    ww,xx,yy,zz = w*w,x*x,y*y,z*z
    wx,wy,wz    = w*x,w*y,w*z
    xy,xz,yz    = x*y,x*z,y*z
    return np.array([
        [ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz]], dtype=np.float64)

def world_to_body(pose: airsim.Pose, vx, vy, vz):
    q  = pose.orientation
    R  = quat_to_R(float(q.w_val),float(q.x_val),float(q.y_val),float(q.z_val))
    vb = R.T @ np.array([float(vx),float(vy),float(vz)])
    return float(vb[0]), float(vb[1]), float(vb[2])
