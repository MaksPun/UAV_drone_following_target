"""Offline controller comparison for PID and LQR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics_lib import RESULTS_DIR, mean, p95, rmse, write_csv, write_markdown_table
from benchmarks.metrics_lib import add_project_root_to_path
from benchmarks.metrics_lib import install_airsim_stub_for_offline_metrics, install_cv2_stub_for_offline_metrics

add_project_root_to_path()
install_airsim_stub_for_offline_metrics()
install_cv2_stub_for_offline_metrics()

from benchmarks.sim_env import DummyKalman, SimState, body_relative, project_detection
from uav_tracking.config import CFG
from uav_tracking.controllers import LQRTracker, PIDTracker
from uav_tracking.speed import SpeedLimits


SCENARIOS = {
    "static_offset": lambda t: (0.0, 0.0, 0.0),
    "lateral_sine": lambda t: (0.4, 1.6 if int(t / 2.0) % 2 == 0 else -1.6, 0.0),
    "depth_step": lambda t: (1.6 if t < 6.0 else -1.1, 0.0, 0.0),
    "vertical_step": lambda t: (0.5, 0.0, -0.8 if t < 5.0 else 0.8),
    "zigzag": lambda t: (0.8, 2.2 if int(t / 1.4) % 2 == 0 else -2.2, -0.5 if int(t / 2.4) % 2 == 0 else 0.5),
}


def simulate(controller_name: str, scenario_name: str, duration: float, dt: float):
    state = SimState.create(drone_xyz=(0.0, 0.0, -3.0), target_xyz=(22.0, 2.0, -3.4))
    kalman = DummyKalman()
    tracker = PIDTracker() if controller_name == "pid" else LQRTracker(dt_d=dt)
    lim = SpeedLimits(CFG.max_vx, CFG.max_vy, CFG.max_vz, CFG.max_yaw_dps)
    frame = (720, 1280)
    rows = []
    t = 0.0
    while t < duration:
        tv = SCENARIOS[scenario_name](t)
        state.move_target_world(*tv, dt)
        kalman.update(state.target, t)
        det, body, img = project_detection(state.drone, state.target, frame=frame)
        if det is None:
            cmd = tracker.step(None, frame, state.drone, state.target, dt, kalman, lim)
        else:
            cmd = tracker.step(det, frame, state.drone, state.target, dt, kalman, lim)
        state.step_drone_body(cmd, dt)
        bx, by, bz = body_relative(state.drone, state.target)
        rows.append(
            {
                "t": t,
                "controller": controller_name,
                "scenario": scenario_name,
                "dist_err": bx - CFG.target_dist_m,
                "lat_err": by,
                "alt_err": bz - CFG.target_alt_off,
                "img_x": img[0],
                "img_y": img[1],
                "visible": 1 if det else 0,
                "cmd_vx": cmd.vx,
                "cmd_vy": cmd.vy,
                "cmd_vz": cmd.vz,
                "cmd_yaw": cmd.yaw,
            }
        )
        t += dt
    return rows


def summarize(rows):
    settled = next((r["t"] for r in rows if abs(r["dist_err"]) < 0.75 and abs(r["lat_err"]) < 0.45 and abs(r["alt_err"]) < 0.35), None)
    return {
        "controller": rows[0]["controller"],
        "scenario": rows[0]["scenario"],
        "mean_abs_dist_m": mean(abs(r["dist_err"]) for r in rows),
        "mean_abs_lat_m": mean(abs(r["lat_err"]) for r in rows),
        "mean_abs_alt_m": mean(abs(r["alt_err"]) for r in rows),
        "rmse_img": rmse((r["img_x"] ** 2 + r["img_y"] ** 2) ** 0.5 for r in rows if r["visible"]),
        "p95_cmd_yaw_dps": p95(abs(r["cmd_yaw"]) for r in rows),
        "visible_ratio": mean(r["visible"] for r in rows),
        "settling_time_s": settled if settled is not None else -1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    all_rows = []
    summary = []
    for scenario in SCENARIOS:
        for controller in ("pid", "lqr"):
            rows = simulate(controller, scenario, args.duration, args.dt)
            all_rows.extend(rows)
            summary.append(summarize(rows))

    write_csv(args.out / "controller_timeseries.csv", all_rows)
    write_csv(args.out / "controller_summary.csv", summary)
    write_markdown_table(args.out / "controller_summary.md", "PID vs LQR offline controller metrics", summary)
    print(f"Wrote {args.out / 'controller_summary.md'}")


if __name__ == "__main__":
    main()
