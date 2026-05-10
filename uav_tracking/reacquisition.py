from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from .common import Cmd, clamp, clamp3, world_to_body
from .config import CFG
from .filters import ExpSmooth, SlewLim


@dataclass
class ExitIntent:
    side:str="CENTER"; vertical:str="LEVEL"
    distance_band:str="MID"; upward_guard:bool=False
    confidence:float=0.; pred_bx:float=0.; pred_by:float=0.
    pred_bz:float=0.; pred_px:float=0.; pred_py:float=0.


#  Reacquisition FSM

class ReacqMode(Enum):
    NONE=auto(); SHORT_INTERCEPT=auto()
    EDGE_RETURN=auto(); ORBIT_RECENTER=auto(); LONG_RECOVER=auto()

class ReacquireFSM:
    def decide(self,lost_s,imt,body_hist,drone_pose,kalman)->ReacqMode:
        yaw_priority=False
        if kalman.ok:
            pred_pos,_=kalman.predict(ahead=min(lost_s+.35,1.3))
            if pred_pos is not None:
                dp=drone_pose.position
                bx,by,_=world_to_body(drone_pose,
                    pred_pos[0]-float(dp.x_val),
                    pred_pos[1]-float(dp.y_val),
                    pred_pos[2]-float(dp.z_val))
                if abs(math.degrees(math.atan2(by,max(.15,bx))))>CFG.reacq_yaw_prio \
                   and abs(by)>1.2:
                    yaw_priority=True
        if yaw_priority:              return ReacqMode.ORBIT_RECENTER
        if lost_s<CFG.reacq_short_s: return ReacqMode.SHORT_INTERCEPT
        if lost_s<CFG.reacq_edge_s and imt.last_edge!="CENTER":
                                      return ReacqMode.EDGE_RETURN
        if lost_s<CFG.reacq_long_s:  return ReacqMode.ORBIT_RECENTER
        return ReacqMode.LONG_RECOVER


#  Predictive reacquisition

class PredictiveReacq:
    def __init__(self):
        self.svx=ExpSmooth(.17); self.svy=ExpSmooth(.17)
        self.svz=ExpSmooth(.14); self.syaw=ExpSmooth(.15)
        self.rvx=SlewLim(1.8);   self.rvy=SlewLim(1.3)
        self.rvz=SlewLim(.95);   self.ryaw=SlewLim(28.)
        self.reason="NONE"; self.mode=ReacqMode.NONE; self.intent=ExitIntent()
        self.return_bias=(0.,0.,0.,1.,0.)

    def reset(self):
        for s in (self.svx,self.svy,self.svz,self.syaw): s.reset()
        for r in (self.rvx,self.rvy,self.rvz,self.ryaw): r.reset()
        self.reason="NONE"; self.mode=ReacqMode.NONE; self.intent=ExitIntent()
        self.return_bias=(0.,0.,0.,1.,0.)

    def _mix(self,drone_pose,kalman,grid27,body_hist,lost_s):
        preds=[]
        if kalman.ok:
            pp,_=kalman.predict(ahead=min(lost_s+.40,2.0))
            if pp is not None:
                dp=drone_pose.position
                bx,by,bz=world_to_body(drone_pose,
                    pp[0]-float(dp.x_val),pp[1]-float(dp.y_val),pp[2]-float(dp.z_val))
                preds.append((bx,by,bz,.52,"kalman"))
        bp,bv=body_hist.predict(ahead=min(lost_s+.30,1.2))
        if bp is not None: preds.append((bp[0],bp[1],bp[2],.28,"body_hist"))
        wps=grid27.predict_body_waypoints(ahead_s=lost_s+.4,top_k=2)
        if wps:
            bxw,byw,bzw,pr,_=wps[0]
            preds.append((bxw,byw,bzw,.20+.10*min(1.,pr),"grid27"))
        if not preds: return None,"none"
        sw=sum(p[3] for p in preds)
        bx=sum(p[0]*p[3] for p in preds)/sw
        by=sum(p[1]*p[3] for p in preds)/sw
        bz=sum(p[2]*p[3] for p in preds)/sw
        return (bx,by,bz),"+".join(sorted({p[4] for p in preds}))

    def _estimate_intent(self,drone_pose,kalman,grid27,imt,body_hist,lost_s)->ExitIntent:
        pb,_=self._mix(drone_pose,kalman,grid27,body_hist,lost_s)
        if pb is None: return ExitIntent()
        bx,by,bz=pb
        band="NEAR" if bx<CFG.zone_x_near else("FAR" if bx>CFG.zone_x_far else "MID")
        pip=imt.predict_img(ahead=min(lost_s+.20,.60))
        ppx,ppy=(pip if pip else (0.,0.))
        _,bv=body_hist.predict(ahead=min(lost_s+.20,.80))
        bvz=float(bv[2]) if bv else 0.
        wvz=float(kalman.vel[2]) if kalman.ok else 0.
        up=0.;dn=0.
        if imt.vy_img<CFG.up_img_vel_thr:  up+=.42
        if bvz<CFG.up_body_vel_thr:        up+=.33
        if wvz<CFG.up_world_vel_thr:       up+=.25
        if ppy<CFG.top_exit_thr:           up+=.28
        if bz<-CFG.zone_z_alt*.55:        up+=.20
        if imt.vy_img>CFG.dn_img_vel_thr: dn+=.42
        if bvz>CFG.dn_body_vel_thr:       dn+=.33
        if wvz>CFG.dn_body_vel_thr:       dn+=.25
        if ppy>abs(CFG.top_exit_thr):     dn+=.20
        if bz>CFG.zone_z_alt*.55:         dn+=.15
        vert=("UP" if up>dn+.12 else("DOWN" if dn>up+.12 else "LEVEL"))
        h=("LEFT" if ppx<-CFG.side_exit_thr or by<-.90
           else("RIGHT" if ppx>CFG.side_exit_thr or by>.90 else ""))
        v=("TOP" if vert=="UP" or ppy<CFG.top_exit_thr
           else("BOTTOM" if vert=="DOWN" or ppy>abs(CFG.top_exit_thr) else ""))
        side=(f"{v}_{h}" if v and h else(v or h or "CENTER"))
        up_guard=(vert=="UP" and band in("NEAR","MID") and lost_s<=CFG.upward_guard_s)
        conf=clamp(max(up,dn,abs(ppx),abs(ppy),min(1.,abs(by)/3.)),0.,1.)
        return ExitIntent(side=side,vertical=vert,distance_band=band,
                          upward_guard=up_guard,confidence=conf,
                          pred_bx=bx,pred_by=by,pred_bz=bz,pred_px=ppx,pred_py=ppy)

    def _up_guard(self,raw_vz,intent,lost_s):
        if not intent.upward_guard: return raw_vz
        fc=CFG.up_strong_climb if "TOP" in intent.side else CFG.up_min_climb
        if intent.distance_band=="NEAR": fc+=CFG.near_up_bonus
        return min(raw_vz,-fc)

    def _image_return_bias(self,imt,lost_s,rvy,rvz,ryy):
        pred=imt.predict_img(ahead=clamp(lost_s+.18,.18,.85))
        if pred is None:
            pred=imt.last_pos
        if pred is None:
            self.return_bias=(0.,0.,0.,1.,0.)
            return self.return_bias

        px,py=pred
        if abs(px)<CFG.reacq_return_deadband: px=0.
        if abs(py)<CFG.reacq_return_deadband: py=0.
        edge=clamp(
            (max(abs(px),abs(py))-CFG.reacq_return_deadband) /
            max(1e-6,1.-CFG.reacq_return_deadband),
            0.,1.)
        time_gain=clamp(lost_s/max(CFG.reacq_edge_s,1e-3),0.,1.)
        gain=.35+.65*max(edge,time_gain)

        yaw=clamp(px*ryy*CFG.reacq_return_yaw_gain*gain,-ryy,ryy)
        vy =clamp(px*rvy*CFG.reacq_return_vy_gain *gain,-rvy,rvy)
        vz =clamp(py*rvz*CFG.reacq_return_vz_gain *gain,-rvz,rvz)
        forward_scale=1.-CFG.reacq_forward_brake*edge
        self.return_bias=(yaw,vy,vz,forward_scale,edge)
        return self.return_bias

    def step(self,drone_pose,kalman,grid27,imt,body_hist,fsm,t,dt)->Cmd:
        lost_s=imt.time_since(t)
        self.mode=fsm.decide(lost_s,imt,body_hist,drone_pose,kalman)
        rvx=CFG.reacq_max_vx; rvy=CFG.reacq_max_vy
        rvz=CFG.reacq_max_vz; ryy=CFG.reacq_max_yaw
        raw_vx=raw_vy=raw_vz=raw_yaw=0.
        pb,reason=self._mix(drone_pose,kalman,grid27,body_hist,lost_s)
        self.reason=reason or "none"
        self.intent=self._estimate_intent(drone_pose,kalman,grid27,imt,body_hist,lost_s)
        intent=self.intent
        bx,by,bz=(pb if pb else (0.,0.,0.))

        if self.mode==ReacqMode.SHORT_INTERCEPT and pb:
            ye=math.degrees(math.atan2(by,max(.15,bx)))
            raw_yaw=clamp(ye*CFG.reacq_intercept,-ryy,ryy)
            raw_vx=clamp((bx-CFG.target_dist_m)*.26,-rvx,rvx)
            raw_vy=clamp(by*.22,-rvy,rvy)
            raw_vz=clamp(-bz*.30,-rvz,rvz)
            if intent.vertical=="UP": raw_vz=min(raw_vz,-CFG.up_min_climb)
        elif self.mode==ReacqMode.EDGE_RETURN:
            e=intent.side
            if "LEFT"   in e: raw_yaw=-ryy*CFG.reacq_edge_boost; raw_vy=-.50*rvy
            elif "RIGHT" in e: raw_yaw=+ryy*CFG.reacq_edge_boost; raw_vy=+.50*rvy
            if "TOP"    in e:
                raw_vz=-CFG.up_strong_climb
                raw_yaw+=(-CFG.corner_yaw_bonus if "LEFT" in e else CFG.corner_yaw_bonus)
            elif "BOTTOM" in e: raw_vz=+.70*rvz
            if pb: raw_vx+=clamp((bx-CFG.target_dist_m)*.14,-.7*rvx,.7*rvx)
        elif self.mode==ReacqMode.ORBIT_RECENTER and pb:
            ye=math.degrees(math.atan2(by,max(.10,bx)))
            raw_yaw=clamp(ye*CFG.reacq_orbit_yaw,-ryy,ryy)
            yp=min(1.,abs(ye)/max(1.,CFG.reacq_yaw_prio)); ts=1.-.65*yp
            raw_vx=clamp((bx-CFG.target_dist_m)*.14*ts,-rvx,rvx)
            raw_vy=clamp(by*.18*ts,-rvy,rvy)
            raw_vz=clamp(-bz*.24,-rvz,rvz)
            if intent.vertical=="UP": raw_vz=min(raw_vz,-CFG.up_min_climb)
            if imt.ready:
                ip=imt.predict_img(ahead=.25)
                if ip: raw_yaw+=clamp(ip[0]*8.,-8.,8.)
        elif self.mode==ReacqMode.LONG_RECOVER:
            if pb:
                ye=math.degrees(math.atan2(by,max(.10,bx)))
                raw_yaw=clamp(ye*1.05,-.85*ryy,.85*ryy)
                raw_vx=clamp((bx-CFG.target_dist_m)*.17,-.85*rvx,.85*rvx)
                raw_vy=clamp(by*.14,-.75*rvy,.75*rvy)
                raw_vz=clamp(-bz*.22,-.85*rvz,.85*rvz)
                if intent.vertical=="UP": raw_vz=min(raw_vz,-CFG.up_min_climb)
            else:
                lp=imt.last_pos
                scan_dir=1. if lp is None or lp[0]>=0. else -1.
                raw_yaw=scan_dir*CFG.reacq_scan_yaw*ryy

        byaw,bvy,bvz,forward_scale,_=self._image_return_bias(imt,lost_s,rvy,rvz,ryy)
        if self.mode==ReacqMode.SHORT_INTERCEPT:
            w=.45
        elif self.mode==ReacqMode.EDGE_RETURN:
            w=.90
        elif self.mode==ReacqMode.ORBIT_RECENTER:
            w=.70
        else:
            w=.85 if not pb else .60
        raw_vx*=1.-(1.-forward_scale)*w
        raw_yaw=clamp(raw_yaw+byaw*w,-ryy,ryy)
        raw_vy =clamp(raw_vy +bvy *w,-rvy,rvy)
        raw_vz =clamp(raw_vz +bvz *w,-rvz,rvz)

        raw_vz=self._up_guard(raw_vz,intent,lost_s)
        z_now=float(drone_pose.position.z_val)
        if z_now>-CFG.alt_floor_m and raw_vz>0.: raw_vz=0.
        if z_now>-CFG.alt_floor_m+.3: raw_vz=min(raw_vz,-.22)

        vx=self.rvx.update(self.svx.update(raw_vx),dt)
        vy=self.rvy.update(self.svy.update(raw_vy),dt)
        vz=self.rvz.update(self.svz.update(raw_vz),dt)
        yw=self.ryaw.update(self.syaw.update(raw_yaw),dt)
        return Cmd(vx=clamp3(vx,rvx),vy=clamp3(vy,rvy),
                   vz=clamp3(vz,rvz),yaw=clamp3(yw,ryy))


