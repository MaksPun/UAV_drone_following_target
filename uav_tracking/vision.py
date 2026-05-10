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


@dataclass
class BodyEstimate:
    bx: float
    by: float
    bz: float
    dist: float
    source: str
    valid_px: int = 0


def project_detection_to_body(det: "Det", frame_shape, fov_deg: float, forward_m: float,
                              source: str, valid_px: int = 0) -> BodyEstimate:
    fh, fw = frame_shape[:2]
    hfov = np.deg2rad(float(fov_deg))
    fx = (fw * 0.5) / max(1e-6, np.tan(hfov * 0.5))
    fy = fx
    bx = float(forward_m)
    by = ((float(det.cx) - fw * 0.5) / fx) * bx
    bz = ((float(det.cy) - fh * 0.5) / fy) * bx
    dist = float(np.sqrt(bx * bx + by * by + bz * bz))
    return BodyEstimate(bx, by, bz, dist, source, int(valid_px))


def estimate_body_from_depth(det: "Det | None", depth, frame_shape, fov_deg: float, roi_shrink: float,
                             min_m: float, max_m: float):
    if det is None or depth is None or frame_shape is None:
        return None
    fh, fw = frame_shape[:2]
    dh, dw = depth.shape[:2]
    x1, y1, x2, y2 = det.xyxy
    sx, sy = dw / max(1, fw), dh / max(1, fh)
    dx1, dy1, dx2, dy2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    cx, cy = 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)
    bw = max(2, dx2 - dx1)
    bh = max(2, dy2 - dy1)

    valid = np.array([], dtype=np.float32)
    for scale in (roi_shrink, 1.0, 1.45, 2.0):
        rw = max(2, int(bw * max(0.1, float(scale)) * 0.5))
        rh = max(2, int(bh * max(0.1, float(scale)) * 0.5))
        rx1 = max(0, int(cx - rw)); rx2 = min(dw, int(cx + rw))
        ry1 = max(0, int(cy - rh)); ry2 = min(dh, int(cy + rh))
        roi = depth[ry1:ry2, rx1:rx2]
        valid = roi[np.isfinite(roi) & (roi >= min_m) & (roi <= max_m)]
        if valid.size >= 6:
            break

    if valid.size < 6:
        ccx = max(0, min(dw - 1, int(cx)))
        ccy = max(0, min(dh - 1, int(cy)))
        patch = depth[max(0, ccy - 3):min(dh, ccy + 4), max(0, ccx - 3):min(dw, ccx + 4)]
        valid = patch[np.isfinite(patch) & (patch >= min_m) & (patch <= max_m)]
    if valid.size < 3:
        return None

    forward = float(np.percentile(valid, 25))
    return project_detection_to_body(det, frame_shape, fov_deg, forward, "depth_roi", int(valid.size))


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
