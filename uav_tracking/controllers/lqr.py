"""LQR visual tracker with observer, integral correction and feed-forward."""

from __future__ import annotations

import math

import airsim
import numpy as np

from ..common import Cmd, clamp, world_to_body
from ..config import CFG
from ..estimation import KalmanCube
from ..filters import ExpSmooth, OnePoleLP, SlewLim, StateObserver, make_K
from ..speed import SpeedLimits
from ..vision import Det


class LQRTracker:
    """Discrete LQR tracker with observer, integral correction and feed-forward."""

    def __init__(self, dt_d=0.10):
        qx,qvx,rx    = CFG.lqr_x
        qy,qvy,ry    = CFG.lqr_y
        qz,qvz,rz    = CFG.lqr_z
        qyw,qvyw,ryw = CFG.lqr_yaw
        qzi,qvzi,rzi = CFG.lqr_imgz
        self._weights=(CFG.lqr_x,CFG.lqr_y,CFG.lqr_z,CFG.lqr_yaw,CFG.lqr_imgz)
        self._dt_d=float(dt_d)
        self.Kx   = make_K(dt_d,qx, qvx,rx)
        self.Ky   = make_K(dt_d,qy, qvy,ry)
        self.Kz   = make_K(dt_d,qz, qvz,rz)
        self.Kyaw = make_K(dt_d,qyw,qvyw,ryw)
        self.Kzi  = make_K(dt_d,qzi,qvzi,rzi)

        self.obs_x  =StateObserver(dt_d,pole=CFG.obs_pole)
        self.obs_y  =StateObserver(dt_d,pole=CFG.obs_pole)
        self.obs_z  =StateObserver(dt_d,pole=CFG.obs_pole)
        self.obs_yaw=StateObserver(dt_d,alpha=CFG.obs_alpha_img)
        self.obs_zi =StateObserver(dt_d,alpha=CFG.obs_alpha_img)

        self.lp_ex =OnePoleLP(.22); self.lp_ey =OnePoleLP(.22)
        self.lp_ez =OnePoleLP(.22); self.lp_exi=OnePoleLP(.20); self.lp_eyi=OnePoleLP(.20)

        self.ff_x_lp=OnePoleLP(.22); self.ff_y_lp=OnePoleLP(.22)
        self.ff_z_lp=OnePoleLP(.22); self.ff_yaw_lp=OnePoleLP(.18)

        self.svx =ExpSmooth(CFG.smooth_vx);  self.svy =ExpSmooth(CFG.smooth_vy)
        self.svz =ExpSmooth(CFG.smooth_vz);  self.syaw=ExpSmooth(CFG.smooth_yaw)
        self.rvx =SlewLim(CFG.slew_vx);      self.rvy =SlewLim(CFG.slew_vy)
        self.rvz =SlewLim(CFG.slew_vz);      self.ryaw=SlewLim(CFG.slew_yaw)
        self.ix=0.; self.iz=0.

    def reset(self):
        for o in (self.obs_x,self.obs_y,self.obs_z,self.obs_yaw,self.obs_zi): o.reset()
        for lp in (self.lp_ex,self.lp_ey,self.lp_ez,self.lp_exi,self.lp_eyi,
                   self.ff_x_lp,self.ff_y_lp,self.ff_z_lp,self.ff_yaw_lp): lp.reset()
        for s in (self.svx,self.svy,self.svz,self.syaw): s.reset()
        for r in (self.rvx,self.rvy,self.rvz,self.ryaw): r.reset()
        self.ix=0.; self.iz=0.

    @staticmethod
    def _db(v,t): return 0. if abs(v)<t else v

    def _refresh_gains(self, dt):
        dt_d=clamp(dt,0.04,0.16)
        if abs(dt_d-self._dt_d) < CFG.lqr_recompute_dt_eps:
            return
        self._dt_d=dt_d
        (qx,qvx,rx),(qy,qvy,ry),(qz,qvz,rz), \
            (qyw,qvyw,ryw),(qzi,qvzi,rzi)=self._weights
        self.Kx   = make_K(dt_d,qx, qvx,rx)
        self.Ky   = make_K(dt_d,qy, qvy,ry)
        self.Kz   = make_K(dt_d,qz, qvz,rz)
        self.Kyaw = make_K(dt_d,qyw,qvyw,ryw)
        self.Kzi  = make_K(dt_d,qzi,qvzi,rzi)

    def _lqr1(self, K, e, de) -> float:
        # The state is only [error, error_rate], which keeps the online controller light.
        return -float((K @ np.array([[e],[de]])).item())

    def step(self, det:Det|None, fshape,
             drone_pose:airsim.Pose, cube_pose:airsim.Pose,
             dt:float, kalman:KalmanCube,
             lim:SpeedLimits, body_xyz=None) -> Cmd:
        """lim = AdaptiveSpeedManager.update() result"""

        if det is None:
            self.ix=0.; self.iz=0.
            return Cmd(vx=self.rvx.update(self.svx.update(0.),dt),
                       vy=self.rvy.update(self.svy.update(0.),dt),
                       vz=self.rvz.update(self.svz.update(0.),dt),
                       yaw=self.ryaw.update(self.syaw.update(0.),dt))

        if CFG.lqr_adaptive_dt:
            self._refresh_gains(dt)

        if body_xyz is None:
            dx=float(cube_pose.position.x_val-drone_pose.position.x_val)
            dy=float(cube_pose.position.y_val-drone_pose.position.y_val)
            dz=float(cube_pose.position.z_val-drone_pose.position.z_val)
            bx,by,bz=world_to_body(drone_pose,dx,dy,dz)
            dist=math.sqrt(dx*dx+dy*dy+dz*dz)
        else:
            bx,by,bz=body_xyz
            dist=math.sqrt(bx*bx+by*by+bz*bz)

        ex_r =self._db(bx-CFG.target_dist_m, CFG.db_x)
        ey_r =self._db(by,                   CFG.db_y)
        ez_r =self._db(bz-CFG.target_alt_off,CFG.db_z)
        h,w=fshape[:2]
        exi_r=self._db((det.cx-w*.5)/(w*.5),              CFG.db_exi)
        eyi_r=self._db((det.cy-h*CFG.aim_y_ratio)/(h*.5), CFG.db_eyi)

        ex =self.lp_ex.update(ex_r);   ey =self.lp_ey.update(ey_r)
        ez =self.lp_ez.update(ez_r);   exi=self.lp_exi.update(exi_r)
        eyi=self.lp_eyi.update(eyi_r)

        ex_h, dex  = self.obs_x.update(ex,  dt)
        ey_h, dey  = self.obs_y.update(ey,  dt)
        ez_h, dez  = self.obs_z.update(ez,  dt)
        exi_h,dexi = self.obs_yaw.update(exi,dt)
        eyi_h,deyi = self.obs_zi.update(eyi, dt)

        edge_x=clamp((abs(exi_h)-0.15)/0.85,0.,1.)
        edge_y=clamp((abs(eyi_h)-0.15)/0.85,0.,1.)
        edge=max(edge_x,edge_y)
        img_brake=1.-CFG.image_forward_brake*edge
        yaw_edge_gain=1.+CFG.img_edge_gain_yaw*edge_x
        vy_edge_gain =1.+CFG.img_edge_gain_vy *edge_x
        vz_edge_gain =1.+CFG.img_edge_gain_vz *edge_y

        ux   = self._lqr1(self.Kx,   ex_h,  dex)
        uy   = self._lqr1(self.Ky,   ey_h,  dey)
        uz   = self._lqr1(self.Kz,   ez_h,  dez)
        uyaw = self._lqr1(self.Kyaw, exi_h, dexi)
        uzi  = self._lqr1(self.Kzi,  eyi_h, deyi)

        if dist>1.2 and edge<.95:
            self.ix=clamp(self.ix+ex_h*dt,-CFG.int_max,CFG.int_max)
            self.iz=clamp(self.iz+ez_h*dt,-CFG.int_max,CFG.int_max)
        else:
            self.ix*=CFG.int_decay_locked
            self.iz*=CFG.int_decay_locked
        ux-=CFG.ki_x*self.ix; uz-=CFG.ki_z*self.iz

        # Feed-forward helps follow moving targets without increasing feedback gains.
        ff_x=ff_y=ff_z=ff_yaw=0.
        if kalman.ok:
            vw=kalman.vel; aw=kalman.acc; vw_p=vw+aw*.10
            fbx,fby,fbz=world_to_body(drone_pose,vw_p[0],vw_p[1],vw_p[2])
            ff_x  =self.ff_x_lp.update(clamp(fbx*CFG.ff_vel_scale,-CFG.max_ff_x,CFG.max_ff_x))
            ff_y  =self.ff_y_lp.update(clamp(fby*CFG.ff_vel_scale*.35,-CFG.max_ff_y,CFG.max_ff_y))
            ff_z  =self.ff_z_lp.update(clamp(fbz*CFG.ff_vel_scale,-CFG.max_ff_z,CFG.max_ff_z))
            ff_yaw=self.ff_yaw_lp.update(clamp(
                math.degrees(math.atan2(fby,max(.25,abs(fbx)+.25)))*.18,
                -CFG.max_ff_yaw,CFG.max_ff_yaw))

        err_mag=abs(ex_h)
        g=CFG.gain_near_scale+(1.-CFG.gain_near_scale)*min(1.,err_mag/4.)

        MX=lim.vx; MY=lim.vy; MZ=lim.vz; MYAW=lim.yaw

        pre_vx =(-ux*g)*img_brake+ff_x*(0.55+0.45*img_brake)
        pre_vy =(-uy*.52*g)*vy_edge_gain+ff_y
        pre_vz =(-(uz+uzi*.20)*vz_edge_gain)+ff_z
        pre_yaw=(-uyaw*yaw_edge_gain)+ff_yaw+clamp(dexi*5.0,-6.0,6.0)

        raw_vx  =clamp(pre_vx,  -MX,  MX)
        raw_vy  =clamp(pre_vy,  -MY,  MY)
        raw_vz  =clamp(pre_vz,  -MZ,  MZ)
        raw_yaw =clamp(pre_yaw, -MYAW,MYAW)
        if abs(pre_vx) > MX*CFG.pid_aw_margin:
            self.ix*=0.90
        if abs(pre_vz) > MZ*CFG.pid_aw_margin:
            self.iz*=0.90

        z_now=float(drone_pose.position.z_val)
        if z_now>-CFG.alt_floor_m and raw_vz>0.: raw_vz=0.
        if z_now>-CFG.alt_floor_m+.3:            raw_vz=min(raw_vz,-.28)

        # Near the lock zone, commands are damped to avoid visible oscillation.
        locked=(abs(ex_h)<CFG.lock_near_m and abs(ey_h)<CFG.lock_near_y
                and abs(ez_h)<CFG.lock_near_z
                and abs(exi_h)<CFG.lock_near_xi and abs(eyi_h)<CFG.lock_near_yi)
        if locked:
            raw_vx *=CFG.gain_near_scale;      raw_vy *=CFG.gain_near_scale*.55
            raw_vz *=CFG.gain_near_scale*2.1;  raw_yaw*=CFG.gain_near_scale*.80
            self.ix*=CFG.int_decay_locked;     self.iz*=CFG.int_decay_locked

        vx =clamp(self.rvx.update( self.svx.update(raw_vx),  dt),-MX,  MX)
        vy =clamp(self.rvy.update( self.svy.update(raw_vy),  dt),-MY,  MY)
        vz =clamp(self.rvz.update( self.svz.update(raw_vz),  dt),-MZ,  MZ)
        yaw=clamp(self.ryaw.update(self.syaw.update(raw_yaw), dt),-MYAW,MYAW)
        return Cmd(vx=vx,vy=vy,vz=vz,yaw=yaw)


#  Safety

