"""AirSim camera capture and YOLO/ByteTrack detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import airsim
import cv2
import numpy as np


def get_scene_bgr(client: airsim.MultirotorClient, cam: str):
    req = [airsim.ImageRequest(cam, airsim.ImageType.Scene, False, True)]
    r   = client.simGetImages(req)[0]
    if r.width == 0 or not r.image_data_uint8: return None
    return cv2.imdecode(np.frombuffer(r.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)


def get_depth_m(client: airsim.MultirotorClient, cam: str):
    req = [airsim.ImageRequest(cam, airsim.ImageType.DepthPerspective, True, False)]
    r = client.simGetImages(req)[0]
    if r.width == 0 or not r.image_data_float:
        return None
    depth = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
    depth[~np.isfinite(depth)] = 0.0
    return depth


#  YOLO

@dataclass
class Det:
    xyxy: tuple; conf: float; track_id: int | None = None
    @property
    def cx(self): return 0.5*(self.xyxy[0]+self.xyxy[2])
    @property
    def cy(self): return 0.5*(self.xyxy[1]+self.xyxy[3])
    @property
    def area(self): return max(1,(self.xyxy[2]-self.xyxy[0])*(self.xyxy[3]-self.xyxy[1]))

class YoloDetector:
    def __init__(self, weights, imgsz, conf, iou, device, bytetrack):
        from ultralytics import YOLO
        self.model=YOLO(weights); self.imgsz=int(imgsz)
        self.conf=float(conf); self.iou=float(iou)
        self.device=device; self.bytetrack=bytetrack; self.sel_id=None

    def infer(self, frame) -> list[Det]:
        if frame is None: return []
        # Ultralytics handles both plain detection and ByteTrack-backed tracking.
        kw=dict(source=frame,imgsz=self.imgsz,conf=self.conf,
                iou=self.iou,device=self.device,verbose=False)
        res=(self.model.track(**kw,tracker="bytetrack.yaml",persist=True)
             if self.bytetrack else self.model.predict(**kw))
        r=res[0]
        if r.boxes is None or len(r.boxes)==0: return []
        xyxy=r.boxes.xyxy.detach().cpu().numpy()
        confs=r.boxes.conf.detach().cpu().numpy()
        ids=(r.boxes.id.detach().cpu().numpy()
             if (hasattr(r.boxes,"id") and r.boxes.id is not None) else None)
        out=[]
        for i in range(len(xyxy)):
            x1,y1,x2,y2=xyxy[i]
            out.append(Det((int(x1),int(y1),int(x2),int(y2)),float(confs[i]),
                           int(ids[i]) if ids is not None else None))
        return out

    def pick(self, dets: list[Det]) -> Det | None:
        # Keep the same ByteTrack ID when possible; otherwise choose the strongest box.
        if not dets: self.sel_id=None; return None
        if self.bytetrack and self.sel_id is not None:
            same=[d for d in dets if d.track_id==self.sel_id]
            if same: return max(same,key=lambda d:d.conf)
        best=max(dets,key=lambda d:d.conf+1e-6*d.area)
        if self.bytetrack and best.track_id is not None: self.sel_id=best.track_id
        return best


#  Commands
