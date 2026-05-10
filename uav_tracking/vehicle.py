"""MAVSDK wrapper around PX4 arm, takeoff, offboard and landing actions."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

from .common import Cmd
from .config import CFG

log = logging.getLogger("UAV")


class DroneState(Enum):
    IDLE=auto(); CONNECTING=auto(); ARMING=auto()
    TAKEOFF=auto(); OFFBOARD=auto(); LANDING=auto(); FAULT=auto()

class MavsdkVehicle:
    """Small async wrapper that hides MAVSDK setup and offboard boilerplate."""

    def __init__(self): self.drone=System(); self.state=DroneState.IDLE; self.offboard_ok=False
    async def connect(self,addr:str):
        self.state=DroneState.CONNECTING; log.info("Connecting -> %s",addr)
        await self.drone.connect(system_address=addr)
        async for s in self.drone.core.connection_state():
            if s.is_connected: log.info("MAVSDK connected"); break
        log.info("Waiting armable ...")
        async for h in self.drone.telemetry.health():
            log.info("armable=%s gps=%s home=%s local=%s",
                     h.is_armable,h.is_global_position_ok,
                     h.is_home_position_ok,h.is_local_position_ok)
            if h.is_armable: log.info("PX4 armable"); break
            await asyncio.sleep(.5)
    async def arm_takeoff_offboard(self,alt:float):
        self.state=DroneState.ARMING; await asyncio.sleep(2.)
        for i in range(CFG.arm_retry):
            try: log.info("Arm %d/%d",i+1,CFG.arm_retry); await self.drone.action.arm(); log.info("Armed"); break
            except Exception as e: log.warning("arm(): %s",e); await asyncio.sleep(1.)
        else: self.state=DroneState.FAULT; raise RuntimeError("Could not arm")
        self.state=DroneState.TAKEOFF
        await self.drone.action.set_takeoff_altitude(float(abs(alt)))
        await self.drone.action.takeoff(); log.info("Takeoff wait %.1f s",CFG.takeoff_wait)
        await asyncio.sleep(CFG.takeoff_wait)
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
        try:
            await self.drone.offboard.start(); self.offboard_ok=True
            self.state=DroneState.OFFBOARD; log.info("Offboard started")
        except OffboardError as e:
            self.state=DroneState.FAULT; log.error("Offboard: %s",e._result.result); raise
    async def send(self,cmd:Cmd):
        if not self.offboard_ok: return
        # PX4 expects body-frame velocity and yaw-rate setpoints in offboard mode.
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(float(cmd.vx),float(cmd.vy),float(cmd.vz),float(cmd.yaw)))
    async def hold(self): await self.send(Cmd())
    async def land(self):
        self.state=DroneState.LANDING
        if self.offboard_ok:
            try: await self.drone.offboard.stop()
            except OffboardError: pass
            self.offboard_ok=False
        await self.drone.action.land()
    async def wait_landed(self, timeout_s: float = 20.0):
        async def _wait():
            async for in_air in self.drone.telemetry.in_air():
                if not in_air:
                    return
        try:
            await asyncio.wait_for(_wait(), timeout=max(1.0, float(timeout_s)))
        except Exception:
            pass
    async def disarm(self):
        try: await self.drone.action.disarm()
        except Exception: pass


#  Cube
