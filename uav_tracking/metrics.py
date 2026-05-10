from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from .config import CFG


METRIC_FIELDS = [
    "wall_time",
    "t",
    "label",
    "scenario",
    "controller",
    "bytetrack",
    "mode",
    "target_visible",
    "lost_s",
    "reacq_mode",
    "reacq_reason",
    "track_id",
    "conf",
    "img_x",
    "img_y",
    "bx",
    "by",
    "bz",
    "dist",
    "dist_err",
    "lat_err",
    "alt_err",
    "cmd_vx",
    "cmd_vy",
    "cmd_vz",
    "cmd_yaw",
    "lim_vx",
    "lim_vy",
    "lim_vz",
    "lim_yaw",
    "cube_cmd_vx",
    "cube_cmd_vy",
    "cube_cmd_vz",
    "return_yaw",
    "return_vy",
    "return_vz",
    "return_forward_scale",
    "return_edge",
    "det_noise_px",
    "latency_ms",
    "scenario_speed_scale",
    "forced_occlusion_s",
]


class MetricsLogger:
    def __init__(
        self,
        path,
        label: str,
        scenario: str,
        controller: str,
        bytetrack: bool,
        det_noise_px: float = 0.0,
        latency_ms: float = 0.0,
        scenario_speed_scale: float = 1.0,
        forced_occlusion_s: float = 0.0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.scenario = scenario
        self.controller = controller
        self.bytetrack = bool(bytetrack)
        self.det_noise_px = float(det_noise_px)
        self.latency_ms = float(latency_ms)
        self.scenario_speed_scale = float(scenario_speed_scale)
        self.forced_occlusion_s = float(forced_occlusion_s)
        self.t0 = time.time()
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()

    def close(self):
        self._file.flush()
        self._file.close()

    def row(self, *, now, mode, target, frame_shape, lost_s, reacq, bx, by, bz,
            dist, cmd, lim, cube_cmd):
        h = w = None
        if frame_shape is not None:
            h, w = frame_shape[:2]
        if target is not None and h and w:
            img_x = (target.cx - w * 0.5) / (w * 0.5)
            img_y = (target.cy - h * CFG.aim_y_ratio) / (h * 0.5)
            track_id = target.track_id if target.track_id is not None else ""
            conf = target.conf
        else:
            img_x = ""
            img_y = ""
            track_id = ""
            conf = ""
        rb_yaw, rb_vy, rb_vz, rb_fx, rb_edge = reacq.return_bias
        self._writer.writerow({
            "wall_time": now,
            "t": now - self.t0,
            "label": self.label,
            "scenario": self.scenario,
            "controller": self.controller,
            "bytetrack": int(self.bytetrack),
            "mode": mode,
            "target_visible": int(target is not None),
            "lost_s": lost_s if math.isfinite(lost_s) else "",
            "reacq_mode": reacq.mode.name,
            "reacq_reason": reacq.reason,
            "track_id": track_id,
            "conf": conf,
            "img_x": img_x,
            "img_y": img_y,
            "bx": bx,
            "by": by,
            "bz": bz,
            "dist": dist,
            "dist_err": bx - CFG.target_dist_m,
            "lat_err": by,
            "alt_err": bz - CFG.target_alt_off,
            "cmd_vx": cmd.vx,
            "cmd_vy": cmd.vy,
            "cmd_vz": cmd.vz,
            "cmd_yaw": cmd.yaw,
            "lim_vx": lim.vx,
            "lim_vy": lim.vy,
            "lim_vz": lim.vz,
            "lim_yaw": lim.yaw,
            "cube_cmd_vx": cube_cmd.vx,
            "cube_cmd_vy": cube_cmd.vy,
            "cube_cmd_vz": cube_cmd.vz,
            "return_yaw": rb_yaw,
            "return_vy": rb_vy,
            "return_vz": rb_vz,
            "return_forward_scale": rb_fx,
            "return_edge": rb_edge,
            "det_noise_px": self.det_noise_px,
            "latency_ms": self.latency_ms,
            "scenario_speed_scale": self.scenario_speed_scale,
            "forced_occlusion_s": self.forced_occlusion_s,
        })
