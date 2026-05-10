"""PID visual tracker with filtering, feed-forward and command shaping."""

from __future__ import annotations

import math

import airsim

from ..common import Cmd, clamp, world_to_body
from ..config import CFG
from ..estimation import KalmanCube
from ..filters import ExpSmooth, OnePoleLP, SlewLim
from ..speed import SpeedLimits
from ..vision import Det


class PIDAxis:
    """One filtered PID channel with optional integral limiting."""
    def __init__(self, kp, ki, kd, int_limit, d_alpha=None):
        self.kp=float(kp); self.ki=float(ki); self.kd=float(kd)
        self.int_limit=float(int_limit); self.i=0.; self.last_e=None
        self.d=0.; self.d_alpha=CFG.pid_deriv_alpha if d_alpha is None else float(d_alpha)
    def reset(self): self.i=0.; self.last_e=None; self.d=0.
    def step(self, e:float, dt:float, integrate=True):
        e=float(e); dt=max(float(dt),1e-3)
        if integrate and self.int_limit > 0.:
            self.i=clamp(self.i*CFG.pid_int_leak+e*dt,-self.int_limit,self.int_limit)
        raw_d=(e-self.last_e)/dt if self.last_e is not None else 0.
        self.d=self.d_alpha*raw_d+(1.-self.d_alpha)*self.d
        self.last_e=e
        return float(self.kp*e+self.ki*self.i+self.kd*self.d)
    def bleed(self, f=0.92): self.i*=float(f)
    def antiwindup(self, raw_u, limit):
        if self.int_limit <= 0. or limit <= 1e-6:
            return
        if abs(raw_u) > abs(limit)*CFG.pid_aw_margin:
            self.bleed(0.88)

class PIDTracker:
    def __init__(self):
        self.pid_x   =PIDAxis(*CFG.pid_x,   CFG.int_max_x)
        self.pid_y   =PIDAxis(*CFG.pid_y,   0.)
        self.pid_z   =PIDAxis(*CFG.pid_z,   CFG.int_max_z)
        self.pid_yaw =PIDAxis(*CFG.pid_yaw, 0.)
        self.pid_imgz=PIDAxis(*CFG.pid_imgz,0.)

        self.f_ex =OnePoleLP(CFG.body_err_alpha); self.f_ey=OnePoleLP(CFG.body_err_alpha)
        self.f_ez =OnePoleLP(CFG.body_err_alpha)
        self.f_exi=OnePoleLP(CFG.img_err_alpha);  self.f_eyi=OnePoleLP(CFG.img_err_alpha)

        self.ffx_lp=OnePoleLP(CFG.ff_vel_alpha); self.ffy_lp=OnePoleLP(CFG.ff_vel_alpha)
        self.ffz_lp=OnePoleLP(CFG.ff_vel_alpha); self.ffyaw_lp=OnePoleLP(CFG.ff_yaw_alpha)

        self.svx =ExpSmooth(CFG.pid_smooth_vx); self.svy =ExpSmooth(CFG.pid_smooth_vy)
        self.svz =ExpSmooth(CFG.pid_smooth_vz); self.syaw=ExpSmooth(CFG.pid_smooth_yaw)
        self.rvx =SlewLim(CFG.pid_slew_vx); self.rvy=SlewLim(CFG.pid_slew_vy)
        self.rvz =SlewLim(CFG.pid_slew_vz); self.ryaw=SlewLim(CFG.pid_slew_yaw)

    def reset(self):
        for p in (self.pid_x,self.pid_y,self.pid_z,self.pid_yaw,self.pid_imgz): p.reset()
        for f in (self.f_ex,self.f_ey,self.f_ez,self.f_exi,self.f_eyi,
                  self.ffx_lp,self.ffy_lp,self.ffz_lp,self.ffyaw_lp): f.reset()
        for s in (self.svx,self.svy,self.svz,self.syaw): s.reset()
        for r in (self.rvx,self.rvy,self.rvz,self.ryaw): r.reset()

    @staticmethod
    def _db(v,t): return 0. if abs(v)<t else v

    def step(self, det:Det|None, fshape,
             drone_pose:airsim.Pose, cube_pose:airsim.Pose,
             dt:float, kalman:KalmanCube,
             lim:SpeedLimits, body_xyz=None) -> Cmd:
        """lim = AdaptiveSpeedManager.update() result"""

        if det is None:
            self.reset()
            return Cmd(vx=self.rvx.update(self.svx.update(0.),dt),
                       vy=self.rvy.update(self.svy.update(0.),dt),
                       vz=self.rvz.update(self.svz.update(0.),dt),
                       yaw=self.ryaw.update(self.syaw.update(0.),dt))

        if body_xyz is None:
            dx=float(cube_pose.position.x_val-drone_pose.position.x_val)
            dy=float(cube_pose.position.y_val-drone_pose.position.y_val)
            dz=float(cube_pose.position.z_val-drone_pose.position.z_val)
            bx,by,bz=world_to_body(drone_pose,dx,dy,dz)
        else:
            bx,by,bz=body_xyz

        ex =self.f_ex.update( self._db(bx-CFG.target_dist_m, CFG.db_x))
        ey =self.f_ey.update( self._db(by,                   CFG.db_y))
        ez =self.f_ez.update( self._db(bz-CFG.target_alt_off,CFG.db_z))
        h,w=fshape[:2]
        exi=self.f_exi.update(self._db((det.cx-w*.5)/(w*.5),             CFG.db_exi))
        eyi=self.f_eyi.update(self._db((det.cy-h*CFG.aim_y_ratio)/(h*.5),CFG.db_eyi))

        # Image-edge gain makes yaw/lateral/vertical response stronger near frame borders.
        edge_x=clamp((abs(exi)-0.15)/0.85,0.,1.)
        edge_y=clamp((abs(eyi)-0.15)/0.85,0.,1.)
        edge=max(edge_x,edge_y)
        img_brake=1.-CFG.image_forward_brake*edge
        yaw_edge_gain=1.+CFG.img_edge_gain_yaw*edge_x
        vy_edge_gain =1.+CFG.img_edge_gain_vy *edge_x
        vz_edge_gain =1.+CFG.img_edge_gain_vz *edge_y

        ux  =self.pid_x.step(ex,  dt)
        uy  =self.pid_y.step(ey,  dt)
        uz  =self.pid_z.step(ez,  dt)
        uyaw=self.pid_yaw.step(exi,dt)
        uzi =self.pid_imgz.step(eyi,dt)

        # Target velocity from Kalman is used only as a bounded feed-forward term.
        ff_x=ff_y=ff_z=ff_yaw=0.
        if kalman.ok:
            vw=kalman.vel; aw=kalman.acc; vw_p=vw+aw*.10
            fbx,fby,fbz=world_to_body(drone_pose,vw_p[0],vw_p[1],vw_p[2])
            ff_x  =self.ffx_lp.update(clamp(fbx*CFG.ff_body,-CFG.max_ff_x_pid,CFG.max_ff_x_pid))
            ff_y  =self.ffy_lp.update(clamp(fby*CFG.ff_body*.48,-CFG.max_ff_y_pid,CFG.max_ff_y_pid))
            ff_z  =self.ffz_lp.update(clamp(fbz*CFG.ff_body*.72,-CFG.max_ff_z_pid,CFG.max_ff_z_pid))
            ff_yaw=self.ffyaw_lp.update(clamp(
                math.degrees(math.atan2(fby,max(.25,abs(fbx)+.25)))*.30,
                -CFG.max_ff_yaw_pid,CFG.max_ff_yaw_pid))

        dist_err=abs(bx-CFG.target_dist_m)
        g=CFG.gain_near_scale+(1.-CFG.gain_near_scale)*min(1.,dist_err/4.)

        MX=lim.vx; MY=lim.vy; MZ=lim.vz; MYAW=lim.yaw

        pre_vx =ux*g*img_brake+ff_x*(0.55+0.45*img_brake)
        pre_vy =(uy*.55*g)*vy_edge_gain+ff_y
        pre_vz =((uz+uzi*.20)*vz_edge_gain)+ff_z
        pre_yaw=(uyaw*yaw_edge_gain)+ff_yaw

        raw_vx  =clamp(pre_vx,  -MX,  MX)
        raw_vy  =clamp(pre_vy,  -MY,  MY)
        raw_vz  =clamp(pre_vz,  -MZ,  MZ)   # NED: positive vz descends.
        raw_yaw =clamp(pre_yaw, -MYAW,MYAW)
        self.pid_x.antiwindup(pre_vx, MX)
        self.pid_z.antiwindup(pre_vz, MZ)

        z_now=float(drone_pose.position.z_val)
        if z_now>-CFG.alt_floor_m and raw_vz>0.: raw_vz=0.
        if z_now>-CFG.alt_floor_m+.3:            raw_vz=min(raw_vz,-.28)

        locked=(abs(ex)<CFG.lock_near_m and abs(ey)<CFG.lock_near_y
                and abs(ez)<CFG.lock_near_z
                and abs(exi)<CFG.lock_near_xi and abs(eyi)<CFG.lock_near_yi)
        if locked:
            raw_vx*=CFG.vx_near_damp; raw_vy*=CFG.vy_near_damp
            raw_vz*=CFG.vz_near_damp; raw_yaw*=CFG.yaw_near_damp
            self.pid_x.bleed(); self.pid_z.bleed()

        vx =clamp(self.rvx.update( self.svx.update(raw_vx), dt),-MX,  MX)
        vy =clamp(self.rvy.update( self.svy.update(raw_vy), dt),-MY,  MY)
        vz =clamp(self.rvz.update( self.svz.update(raw_vz), dt),-MZ,  MZ)
        yaw=clamp(self.ryaw.update(self.syaw.update(raw_yaw),dt),-MYAW,MYAW)
        return Cmd(vx=vx,vy=vy,vz=vz,yaw=yaw)


