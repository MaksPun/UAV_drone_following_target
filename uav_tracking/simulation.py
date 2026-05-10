from __future__ import annotations

import airsim

from .common import CubeCmd, clamp, sf


def move_cube(client,name,cmd:CubeCmd,dt,z_min=-30.,z_max=-.5):
    pose=client.simGetObjectPose(name)
    if pose is None: return
    x=float(pose.position.x_val+cmd.vx*dt)
    y=float(pose.position.y_val+cmd.vy*dt)
    z=clamp(float(pose.position.z_val+cmd.vz*dt),z_min,z_max)
    pose.position=airsim.Vector3r(sf(x),sf(y),sf(z))
    client.simSetObjectPose(name,pose,True)


#  HUD
