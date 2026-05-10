"""Offline comparison of target reacquisition policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics_lib import RESULTS_DIR, mean, p95, write_csv, write_markdown_table, add_project_root_to_path
from benchmarks.metrics_lib import install_airsim_stub_for_offline_metrics

add_project_root_to_path()
install_airsim_stub_for_offline_metrics()

from uav_tracking.config import CFG
from uav_tracking.reacquisition import PredictiveReacq


class ImageTrack:
    def __init__(self, x0, y0, vx, vy):
        self.x = x0
        self.y = y0
        self.vx_img = vx
        self.vy_img = vy
        self.last_edge = "CENTER"
        self._last_t = 0.0

    def predict_img(self, ahead=0.25):
        return max(-1.3, min(1.3, self.x + self.vx_img * ahead)), max(-1.3, min(1.3, self.y + self.vy_img * ahead))

    @property
    def last_pos(self):
        return self.x, self.y


def command(policy: str, imt: ImageTrack, reacq: PredictiveReacq, lost_s: float):
    rvy, rvz, ryy = CFG.reacq_max_vy, CFG.reacq_max_vz, CFG.reacq_max_yaw
    if policy == "scan":
        direction = 1.0 if imt.x >= 0 else -1.0
        return direction * CFG.reacq_scan_yaw * ryy, 0.0, 0.0
    if policy == "edge_only":
        yaw = (1.0 if imt.x >= 0 else -1.0) * ryy * 0.75 if abs(imt.x) > 0.55 else 0.0
        vy = (1.0 if imt.x >= 0 else -1.0) * rvy * 0.35 if abs(imt.x) > 0.55 else 0.0
        vz = (1.0 if imt.y >= 0 else -1.0) * rvz * 0.45 if abs(imt.y) > 0.55 else 0.0
        return yaw, vy, vz
    yaw, vy, vz, _, _ = reacq._image_return_bias(imt, lost_s, rvy, rvz, ryy)
    return yaw, vy, vz


def simulate(policy: str, x0: float, y0: float, vx: float, vy: float, dt=0.05, duration=6.0):
    imt = ImageTrack(x0, y0, vx, vy)
    reacq = PredictiveReacq()
    rows = []
    t = 0.0
    recovered = None
    while t < duration:
        yaw, cmd_vy, cmd_vz = command(policy, imt, reacq, t)
        # Simplified image dynamics: yaw and lateral velocity reduce horizontal error;
        # vertical velocity reduces vertical error. Residual image velocity keeps moving
        # the target as if it is still escaping.
        imt.x += imt.vx_img * dt - (yaw / CFG.reacq_max_yaw * 1.05 + cmd_vy / CFG.reacq_max_vy * 0.45) * dt
        imt.y += imt.vy_img * dt - (cmd_vz / CFG.reacq_max_vz * 1.05) * dt
        rows.append({"t": t, "policy": policy, "img_err": (imt.x * imt.x + imt.y * imt.y) ** 0.5, "abs_x": abs(imt.x), "abs_y": abs(imt.y)})
        if recovered is None and abs(imt.x) < 0.18 and abs(imt.y) < 0.18:
            recovered = t
        t += dt
    return rows, recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    cases = [
        ("right_exit", 1.05, 0.10, 0.22, 0.00),
        ("left_top_exit", -0.95, -0.75, -0.16, -0.15),
        ("bottom_right_exit", 0.85, 0.90, 0.12, 0.20),
        ("fast_top_exit", 0.20, -1.05, 0.02, -0.28),
    ]
    all_rows = []
    summary = []
    for name, x0, y0, vx, vy in cases:
        for policy in ("scan", "edge_only", "predictive_return"):
            rows, recovered = simulate(policy, x0, y0, vx, vy)
            for row in rows:
                row["case"] = name
            all_rows.extend(rows)
            summary.append(
                {
                    "case": name,
                    "policy": policy,
                    "success": 1 if recovered is not None else 0,
                    "recovery_time_s": recovered if recovered is not None else -1.0,
                    "mean_img_error": mean(r["img_err"] for r in rows),
                    "p95_abs_x": p95(r["abs_x"] for r in rows),
                    "p95_abs_y": p95(r["abs_y"] for r in rows),
                }
            )
    write_csv(args.out / "reacquisition_timeseries.csv", all_rows)
    write_csv(args.out / "reacquisition_summary.csv", summary)
    write_markdown_table(args.out / "reacquisition_summary.md", "Target return-to-frame policies", summary)
    print(f"Wrote {args.out / 'reacquisition_summary.md'}")


if __name__ == "__main__":
    main()
