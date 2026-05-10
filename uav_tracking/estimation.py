"""State and motion estimators used by tracking and reacquisition."""

from __future__ import annotations

from collections import deque

import airsim
import numpy as np

from .common import clamp, world_to_body
from .config import CFG
from .vision import Det


class KalmanCube:
    """Constant-acceleration Kalman model for the target object in world frame."""
    def __init__(self):
        ps=CFG.kalman_proc; ms=CFG.kalman_meas
        self._ps=ps; self._ms=ms
        self.x=np.zeros(9); self.P=np.eye(9)*8.
        self.H=np.eye(3,9); self.R=np.eye(3)*ms**2
        self.ok=False; self.t0=0.

    def _FQ(self,dt):
        t=dt; t2=t*t/2; t3=t*t*t/6
        F=np.eye(9)
        for i in range(3):
            F[i,i+3]=t; F[i,i+6]=t2; F[i+3,i+6]=t
        q=self._ps; Q=np.zeros((9,9))
        for i in range(3):
            Q[i,i]=q*t**5/20; Q[i,i+3]=q*t**4/8; Q[i,i+6]=q*t3
            Q[i+3,i]=Q[i,i+3]; Q[i+3,i+3]=q*t3; Q[i+3,i+6]=q*t2
            Q[i+6,i]=Q[i,i+6]; Q[i+6,i+3]=Q[i+3,i+6]; Q[i+6,i+6]=q*t
        return F,Q

    def update(self, pose: airsim.Pose, t: float):
        p=pose.position
        z=np.array([float(p.x_val),float(p.y_val),float(p.z_val)])
        if not self.ok: self.x[:3]=z; self.ok=True; self.t0=t; return
        dt=t-self.t0; self.t0=t
        if dt<1e-4: return
        F,Q=self._FQ(dt); xp=F@self.x; Pp=F@self.P@F.T+Q
        y=z-self.H@xp; S=self.H@Pp@self.H.T+self.R
        K=Pp@self.H.T@np.linalg.inv(S)
        self.x=xp+K@y; self.P=(np.eye(9)-K@self.H)@Pp

    def predict(self, ahead: float):
        if not self.ok: return None,None
        t=ahead; t2=t*t/2
        return (self.x[:3]+self.x[3:6]*t+self.x[6:9]*t2,
                self.x[3:6]+self.x[6:9]*t)

    @property
    def vel(self): return self.x[3:6].copy()
    @property
    def acc(self): return self.x[6:9].copy()
    def reset(self): self.x[:]=0; self.P[:]=np.eye(9)*8; self.ok=False


#  27-zone spatial predictor

class SpatialGrid27:
    """Compact 3x3x3 memory of recent target locations in the drone body frame."""
    N=27
    def __init__(self):
        self._T=np.ones((self.N,self.N))*0.1
        self._dwell=np.ones(self.N)*0.5
        self._last_zone:int|None=None; self._last_t=0.
        self._centres=self._build_centres()

    @staticmethod
    def xyz_to_zone(bx,by,bz)->int:
        xi=0 if bx<CFG.zone_x_near else(2 if bx>CFG.zone_x_far else 1)
        yi=0 if by<-CFG.zone_y_lat else(2 if by>CFG.zone_y_lat else 1)
        zi=0 if bz<-CFG.zone_z_alt else(2 if bz>CFG.zone_z_alt else 1)
        return xi*9+yi*3+zi

    @staticmethod
    def zone_to_ijk(z)->tuple:
        xi=z//9; r=z%9; yi=r//3; zi=r%3; return xi,yi,zi

    def _build_centres(self):
        cx=[CFG.zone_x_near*.5,(CFG.zone_x_near+CFG.zone_x_far)*.5,CFG.zone_x_far+5.]
        cy=[-CFG.zone_y_lat*1.5,0.,CFG.zone_y_lat*1.5]
        cz=[-CFG.zone_z_alt*1.5,0.,CFG.zone_z_alt*1.5]
        out=np.zeros((27,3))
        for xi in range(3):
            for yi in range(3):
                for zi in range(3):
                    out[xi*9+yi*3+zi]=[cx[xi],cy[yi],cz[zi]]
        return out

    def push(self,bx,by,bz,t:float):
        zone=self.xyz_to_zone(bx,by,bz)
        if self._last_zone is not None and self._last_zone!=zone:
            self._T[self._last_zone,zone]+=1.
            dt=t-self._last_t
            self._dwell[self._last_zone]=.85*self._dwell[self._last_zone]+.15*dt
        self._last_zone=zone; self._last_t=t

    def predict_body_waypoints(self,ahead_s:float,top_k:int=3)->list[tuple]:
        if self._last_zone is None: return []
        ns=max(1,int(ahead_s/max(self._dwell[self._last_zone],.2)))
        Tn=self._T/(self._T.sum(axis=1,keepdims=True)+1e-9)
        Pn=np.linalg.matrix_power(Tn,min(ns,6))
        probs=Pn[self._last_zone]
        top_idx=np.argsort(probs)[::-1][:top_k]
        out=[]
        for idx in top_idx:
            c=self._centres[idx]
            out.append((float(c[0]),float(c[1]),float(c[2]),float(probs[idx]),idx))
        return out

    @property
    def current_zone(self): return self._last_zone

    @property
    def zone_label(self):
        if self._last_zone is None: return "?"
        xi,yi,zi=self.zone_to_ijk(self._last_zone)
        return f"{'NEAR|MID |FAR '.split('|')[xi]}|{'LEFT|CTR |RGHT'.split('|')[yi]}|{'UP  |LVL |DOWN'.split('|')[zi]}"


#  Motion trackers

class ImageMotionTracker:
    """Tracks normalized bbox-center motion for frame-exit prediction."""
    def __init__(self):
        self._buf:deque=deque(maxlen=CFG.img_hist)
        self.vx_img=0.; self.vy_img=0.
        self._last_pos=None; self._last_t=0.; self.last_edge="UNKNOWN"

    def reset(self):
        self._buf.clear(); self.vx_img=0.; self.vy_img=0.
        self._last_pos=None; self._last_t=0.; self.last_edge="UNKNOWN"

    @staticmethod
    def _edge(x,y):
        h=("LEFT" if x<-.60 else("RIGHT" if x>.60 else ""))
        v=("TOP"  if y<-.60 else("BOTTOM" if y>.60 else ""))
        if v and h: return f"{v}_{h}"
        return v or h or "CENTER"

    def push(self,det:Det,fshape,t:float):
        h,w=fshape[:2]
        cxn=(det.cx-w*.5)/(w*.5); cyn=(det.cy-h*.5)/(h*.5)
        self._buf.append((cxn,cyn,t))
        self._last_pos=(cxn,cyn); self._last_t=t
        self.last_edge=self._edge(cxn,cyn); self._fit()

    def _fit(self):
        if len(self._buf)<3: return
        pts=list(self._buf); t0=pts[0][2]
        ts=np.array([p[2]-t0 for p in pts])
        xs=np.array([p[0] for p in pts]); ys=np.array([p[1] for p in pts])
        if ts[-1]<1e-3: return
        A=np.column_stack([ts,np.ones_like(ts)])
        try:
            vx,_=np.linalg.lstsq(A,xs,rcond=None)[0]
            vy,_=np.linalg.lstsq(A,ys,rcond=None)[0]
            self.vx_img=.30*vx+.70*self.vx_img
            self.vy_img=.30*vy+.70*self.vy_img
        except Exception: pass

    def predict_img(self,ahead=0.25):
        if self._last_pos is None: return None
        return (clamp(self._last_pos[0]+self.vx_img*ahead,-1,1),
                clamp(self._last_pos[1]+self.vy_img*ahead,-1,1))

    def time_since(self,t): return t-self._last_t if self._last_t>0 else 9999.
    @property
    def ready(self): return len(self._buf)>=3
    @property
    def last_pos(self): return self._last_pos

class BodyMotionTracker:
    """Tracks body-frame target motion for short-horizon prediction."""
    def __init__(self):
        self._buf:deque=deque(maxlen=CFG.body_hist_n)

    def push(self,bx,by,bz,t):
        self._buf.append((float(bx),float(by),float(bz),float(t)))

    def predict(self,ahead=0.35):
        if len(self._buf)<3: return None,None
        pts=list(self._buf); t0=pts[0][3]
        ts=np.array([p[3]-t0 for p in pts])
        xs=np.array([p[0] for p in pts]); ys=np.array([p[1] for p in pts])
        zs=np.array([p[2] for p in pts])
        if ts[-1]<1e-3: return None,None
        A=np.column_stack([ts,np.ones_like(ts)])
        try:
            vx,_=np.linalg.lstsq(A,xs,rcond=None)[0]
            vy,_=np.linalg.lstsq(A,ys,rcond=None)[0]
            vz,_=np.linalg.lstsq(A,zs,rcond=None)[0]
            return ((float(xs[-1]+vx*ahead),float(ys[-1]+vy*ahead),float(zs[-1]+vz*ahead)),
                    (float(vx),float(vy),float(vz)))
        except Exception: return None,None


#  Exit intent
