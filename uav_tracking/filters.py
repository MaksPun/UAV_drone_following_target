from __future__ import annotations

import numpy as np

from .common import clamp
from .config import CFG


class SlewLim:
    def __init__(self, rate, v0=0.): self.rate=float(rate); self.y=float(v0)
    def reset(self, v=0.): self.y=float(v)
    def update(self, tgt, dt):
        dm=self.rate*max(dt,1e-3)
        self.y+=clamp(float(tgt)-self.y,-dm,dm); return self.y

class ExpSmooth:
    def __init__(self, a=0.18, v0=0.): self.a=float(a); self.y=float(v0)
    def reset(self, v=0.): self.y=float(v)
    def update(self, x):
        self.y=self.a*float(x)+(1-self.a)*self.y; return self.y

class OnePoleLP:
    def __init__(self, alpha=0.20):
        self.alpha=float(alpha); self.y=0.; self.init=False
    def reset(self): self.y=0.; self.init=False
    def update(self, x):
        x=float(x)
        if not self.init: self.y=x; self.init=True; return self.y
        self.y=self.alpha*x+(1-self.alpha)*self.y; return self.y


#  LQR solver

def dare(A, B, Q, R, n=600, eps=1e-12):
    P=Q.copy()
    for _ in range(n):
        ATP=A.T@P; S=R+B.T@P@B; K=np.linalg.solve(S,B.T@P@A)
        Pn=ATP@A-ATP@B@K+Q
        if np.max(np.abs(Pn-P))<eps: return Pn
        P=Pn
    return P

def dlqr(A, B, Q, R):
    P=dare(A,B,Q,R)
    return np.linalg.solve(R+B.T@P@B, B.T@P@A)

def make_K(dt, qp, qv, r):
    A=np.array([[1.,dt],[0.,1.]]); B=np.array([[.5*dt*dt],[dt]])
    return dlqr(A, B, np.diag([qp, qv]), np.array([[r]]))


#  State observer (Luenberger)

class StateObserver:
    def __init__(self, dt, pole=None, alpha=None):
        self._alpha=alpha
        self._pole=pole if pole is not None else CFG.obs_pole
        self.x=np.zeros(2); self._first=True
        if alpha is None:
            p=self._pole
            self.L=np.array([1.0-p**2, (1.0-p)**2/max(dt,1e-3)])
        else:
            self.L=None

    def reset(self): self.x[:]=0.; self._first=True

    def update(self, e_meas: float, dt: float) -> tuple[float, float]:
        e_meas=float(e_meas)
        if self._first:
            self.x[0]=e_meas; self.x[1]=0.; self._first=False
            return e_meas, 0.
        if self._alpha is not None:
            de=self._alpha*(e_meas-self.x[0])/max(dt,1e-3)+(1-self._alpha)*self.x[1]
            self.x[0]=e_meas; self.x[1]=de
            return e_meas, de
        xp=np.array([self.x[0]+self.x[1]*dt, self.x[1]])
        self.x=xp+self.L*(e_meas-xp[0])
        return float(self.x[0]), float(self.x[1])


#  9-state Kalman
