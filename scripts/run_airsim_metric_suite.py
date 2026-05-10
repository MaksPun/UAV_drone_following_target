from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_tracking.airsim_scenarios import SCENARIO_NAMES


DEFAULT_SCENARIOS = [
    "static",
    "lateral_sine",
    "depth_step",
    "vertical_step",
    "zigzag",
    "frame_exit_right",
    "frame_exit_left",
    "frame_exit_top",
    "frame_exit_bottom",
    "occlusion_like",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", choices=["pid", "lqr"], default=["pid", "lqr"])
    ap.add_argument("--scenarios", nargs="+", choices=SCENARIO_NAMES, default=DEFAULT_SCENARIOS)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cube_name", default="BP_TargetCube_5")
    ap.add_argument("--mavsdk_address", default="udpin://0.0.0.0:14560")
    ap.add_argument("--weights", default=r"C:\Users\User\PycharmProjects\AirSim\runs\detect\train4\weights\best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--bytetrack", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "metrics_results" / "airsim_runs")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without launching AirSim runs.")
    ap.add_argument("--metrics_det_noise_px", type=float, default=0.0)
    ap.add_argument("--metrics_latency_ms", type=float, default=0.0)
    ap.add_argument("--metrics_occlusion_start", type=float, default=-1.0)
    ap.add_argument("--metrics_occlusion_duration", type=float, default=0.0)
    ap.add_argument("--metrics_scenario_speed_scale", type=float, default=1.0)
    ap.add_argument("--post_run_sleep", type=float, default=10.0)
    ap.add_argument("--retry_failed", type=int, default=1)
    ap.add_argument("--retry_sleep", type=float, default=25.0)
    ap.add_argument("--run_timeout", type=float, default=180.0)
    ap.add_argument("--connect_timeout", type=float, default=90.0)
    ap.add_argument("--landing_wait", type=float, default=8.0)
    ap.add_argument("--stop_on_error", action="store_true",
                    help="Stop the suite when a single run fails. Default is to continue.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for controller in args.controllers:
        for scenario in args.scenarios:
            for repeat in range(args.repeats):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                bt = "bt" if args.bytetrack else "nobt"
                label = f"{controller}_{scenario}_{bt}_rep{repeat:02d}_{stamp}"
                log_path = args.out_dir / f"{label}.csv"
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_tracking.py"),
                    "--ctrl", controller,
                    "--cube_name", args.cube_name,
                    "--mavsdk_address", args.mavsdk_address,
                    "--weights", args.weights,
                    "--device", args.device,
                    "--metrics_scenario", scenario,
                    "--metrics_duration", str(args.duration),
                    "--metrics_label", label,
                    "--metrics_log", str(log_path),
                    "--metrics_det_noise_px", str(args.metrics_det_noise_px),
                    "--metrics_latency_ms", str(args.metrics_latency_ms),
                    "--metrics_occlusion_start", str(args.metrics_occlusion_start),
                    "--metrics_occlusion_duration", str(args.metrics_occlusion_duration),
                    "--metrics_scenario_speed_scale", str(args.metrics_scenario_speed_scale),
                    "--connect_timeout", str(args.connect_timeout),
                    "--landing_wait", str(args.landing_wait),
                    "--land_on_finish",
                ]
                if args.bytetrack:
                    cmd.append("--bytetrack")
                print("$ " + " ".join(cmd), flush=True)
                if not args.dry_run:
                    ok = False
                    last_error = ""
                    for attempt in range(args.retry_failed + 1):
                        if attempt > 0:
                            print(f"Retry {attempt}/{args.retry_failed} after {args.retry_sleep:.1f}s: {label}", flush=True)
                            time.sleep(max(0.0, args.retry_sleep))
                        try:
                            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, timeout=args.run_timeout)
                            ok = True
                            break
                        except subprocess.TimeoutExpired:
                            last_error = f"timeout>{args.run_timeout:.1f}s"
                            print(f"Run timeout: {label} {last_error}", flush=True)
                        except subprocess.CalledProcessError as exc:
                            last_error = f"returncode={exc.returncode}"
                            print(f"Run failed: {label} {last_error}", flush=True)
                    if not ok:
                        failures.append({
                            "label": label,
                            "controller": controller,
                            "scenario": scenario,
                            "repeat": repeat,
                            "error": last_error,
                            "log_path": str(log_path),
                        })
                        if args.stop_on_error:
                            raise RuntimeError(f"Run failed: {label} {last_error}")
                    if args.post_run_sleep > 0:
                        print(f"Cooling down {args.post_run_sleep:.1f}s before next run ...", flush=True)
                        time.sleep(args.post_run_sleep)

    if failures:
        failure_path = args.out_dir / "suite_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        print(f"Wrote failures -> {failure_path}", flush=True)


if __name__ == "__main__":
    main()
