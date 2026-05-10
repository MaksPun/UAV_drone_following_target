"""Batch runner for extended AirSim robustness experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_tracking.airsim_scenarios import SCENARIO_NAMES


BASE_SCENARIOS = [
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

FRAME_EXIT_SCENARIOS = [
    "frame_exit_right",
    "frame_exit_left",
    "frame_exit_top",
    "frame_exit_bottom",
    "frame_exit_top_right",
    "frame_exit_top_left",
    "frame_exit_bottom_right",
    "frame_exit_bottom_left",
]


@dataclass(frozen=True)
class AirSimRun:
    test: str
    controller: str
    scenario: str
    repeat: int
    param_name: str = "none"
    param_value: float = 0.0
    det_noise_px: float = 0.0
    latency_ms: float = 0.0
    speed_scale: float = 1.0
    occlusion_start: float = -1.0
    occlusion_duration: float = 0.0
    bytetrack: bool = False


def tag_value(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_runs(args) -> list[AirSimRun]:
    runs: list[AirSimRun] = []
    controllers = args.controllers

    if "repeated_runs" in args.tests:
        for scenario in args.repeated_scenarios:
            for controller in controllers:
                for repeat in range(args.repeats):
                    runs.append(AirSimRun("repeated_runs", controller, scenario, repeat, bytetrack=args.bytetrack))

    if "detection_noise_sweep" in args.tests:
        for noise in args.noise_levels:
            for controller in controllers:
                for repeat in range(args.sweep_repeats):
                    runs.append(
                        AirSimRun(
                            "detection_noise_sweep",
                            controller,
                            args.sweep_scenario,
                            repeat,
                            "bbox_noise_px",
                            float(noise),
                            det_noise_px=float(noise),
                            bytetrack=args.bytetrack,
                        )
                    )

    if "latency_sweep" in args.tests:
        for latency in args.latency_ms:
            for controller in controllers:
                for repeat in range(args.sweep_repeats):
                    runs.append(
                        AirSimRun(
                            "latency_sweep",
                            controller,
                            args.sweep_scenario,
                            repeat,
                            "latency_ms",
                            float(latency),
                            latency_ms=float(latency),
                            bytetrack=args.bytetrack,
                        )
                    )

    if "target_speed_sweep" in args.tests:
        for speed_scale in args.speed_scales:
            for controller in controllers:
                for repeat in range(args.sweep_repeats):
                    runs.append(
                        AirSimRun(
                            "target_speed_sweep",
                            controller,
                            args.speed_scenario,
                            repeat,
                            "speed_scale",
                            float(speed_scale),
                            speed_scale=float(speed_scale),
                            bytetrack=args.bytetrack,
                        )
                    )

    if "occlusion_duration_sweep" in args.tests:
        for duration in args.occlusion_durations:
            for controller in controllers:
                for repeat in range(args.sweep_repeats):
                    runs.append(
                        AirSimRun(
                            "occlusion_duration_sweep",
                            controller,
                            "occlusion_like",
                            repeat,
                            "occlusion_duration_s",
                            float(duration),
                            occlusion_start=args.occlusion_start,
                            occlusion_duration=float(duration),
                            bytetrack=args.bytetrack,
                        )
                    )

    if "full_frame_exit_set" in args.tests:
        for scenario in FRAME_EXIT_SCENARIOS:
            for controller in controllers:
                for repeat in range(args.frame_repeats):
                    runs.append(
                        AirSimRun(
                            "full_frame_exit_set",
                            controller,
                            scenario,
                            repeat,
                            "frame_exit_case",
                            float(FRAME_EXIT_SCENARIOS.index(scenario)),
                            speed_scale=args.frame_exit_speed_scale,
                            bytetrack=args.bytetrack,
                        )
                    )
    return runs


def command_for_run(args, run: AirSimRun, log_path: Path, label: str) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_tracking.py"),
        "--ctrl", run.controller,
        "--cube_name", args.cube_name,
        "--mavsdk_address", args.mavsdk_address,
        "--weights", args.weights,
        "--device", args.device,
        "--metrics_scenario", run.scenario,
        "--metrics_duration", str(args.duration),
        "--metrics_label", label,
        "--metrics_log", str(log_path),
        "--metrics_det_noise_px", str(run.det_noise_px),
        "--metrics_latency_ms", str(run.latency_ms),
        "--metrics_occlusion_start", str(run.occlusion_start),
        "--metrics_occlusion_duration", str(run.occlusion_duration),
        "--metrics_scenario_speed_scale", str(run.speed_scale),
        "--connect_timeout", str(args.connect_timeout),
        "--landing_wait", str(args.landing_wait),
        "--drone_start_pose", args.drone_start_pose,
        "--cube_start_pose", args.cube_start_pose,
        "--pose_settle_s", str(args.pose_settle_s),
        "--land_on_finish",
    ]
    if args.reset_poses:
        cmd.append("--reset_poses")
    if run.bytetrack:
        cmd.append("--bytetrack")
    return cmd


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def log_prefix(run: AirSimRun) -> str:
    bt = "bt" if run.bytetrack else "nobt"
    param_tag = f"{run.param_name}_{tag_value(run.param_value)}"
    return f"air_{run.test}_{run.controller}_{run.scenario}_{param_tag}_{bt}_rep{run.repeat:02d}"


def find_existing_log(out_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(out_dir.glob("**/" + prefix + "_*.csv"))
    for path in reversed(candidates):
        if path.exists() and path.stat().st_size > 200:
            return path
    return None


def combine_csv_by_test(out_dir: Path, manifest: list[dict]) -> None:
    by_test_dir = out_dir / "by_test"
    by_test_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for item in manifest:
        status = item.get("status", "")
        if status not in {"ok", "skipped_existing"}:
            continue
        path = Path(str(item.get("log_path", "")))
        if not path.exists() or path.stat().st_size == 0:
            continue
        grouped.setdefault(str(item["test"]), []).append(item)

    for test, items in grouped.items():
        combined_rows = []
        fieldnames = []
        manifest_rows = []
        for item in items:
            path = Path(str(item["log_path"]))
            manifest_rows.append(item)
            with path.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
                extra = ["test_type", "test_param_name", "test_param_value", "repeat", "source_log"]
                for name in extra:
                    if name not in fieldnames:
                        fieldnames.append(name)
                for row in reader:
                    row["test_type"] = item["test"]
                    row["test_param_name"] = item["param_name"]
                    row["test_param_value"] = item["param_value"]
                    row["repeat"] = item["repeat"]
                    row["source_log"] = str(path)
                    combined_rows.append(row)
        out_path = by_test_dir / f"{test}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        write_manifest(by_test_dir / f"{test}_manifest.csv", manifest_rows)
        print(f"Combined {test} rows -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tests", nargs="+", choices=[
        "repeated_runs",
        "detection_noise_sweep",
        "latency_sweep",
        "target_speed_sweep",
        "occlusion_duration_sweep",
        "full_frame_exit_set",
    ], default=[
        "repeated_runs",
        "detection_noise_sweep",
        "latency_sweep",
        "target_speed_sweep",
        "occlusion_duration_sweep",
        "full_frame_exit_set",
    ])
    ap.add_argument("--controllers", nargs="+", choices=["pid", "lqr"], default=["pid", "lqr"])
    ap.add_argument("--repeated_scenarios", nargs="+", choices=SCENARIO_NAMES, default=BASE_SCENARIOS)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sweep_repeats", type=int, default=1)
    ap.add_argument("--frame_repeats", type=int, default=1)
    ap.add_argument("--noise_levels", type=float, nargs="+", default=[0, 2, 5, 10, 20, 35])
    ap.add_argument("--latency_ms", type=float, nargs="+", default=[0, 50, 100, 150, 200])
    ap.add_argument("--speed_scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--occlusion_durations", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--occlusion_start", type=float, default=3.0)
    ap.add_argument("--sweep_scenario", choices=SCENARIO_NAMES, default="zigzag")
    ap.add_argument("--speed_scenario", choices=SCENARIO_NAMES, default="zigzag")
    ap.add_argument("--frame_exit_speed_scale", type=float, default=1.0)
    ap.add_argument("--cube_name", default="BP_TargetCube_5")
    ap.add_argument("--mavsdk_address", default="udpin://0.0.0.0:14560")
    ap.add_argument("--weights", default=r"C:\Users\User\PycharmProjects\AirSim\runs\detect\train4\weights\best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--bytetrack", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "metrics_results" / "airsim_extended")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--post_run_sleep", type=float, default=10.0,
                    help="Pause between AirSim/PX4 runs so MAVSDK server ports and PX4 landing state can settle.")
    ap.add_argument("--retry_failed", type=int, default=1,
                    help="Retry each failed run this many times.")
    ap.add_argument("--retry_sleep", type=float, default=25.0)
    ap.add_argument("--run_timeout", type=float, default=180.0,
                    help="Hard timeout for a single run. Increase when duration is long.")
    ap.add_argument("--connect_timeout", type=float, default=90.0)
    ap.add_argument("--landing_wait", type=float, default=8.0)
    ap.add_argument("--reset_poses", action=argparse.BooleanOptionalAction, default=True,
                    help="Reset drone and cube to fixed poses before and after each run.")
    ap.add_argument("--drone_start_pose", default="0,0,0,0",
                    help="AirSim NED x,y,z,yaw_deg for drone before each test.")
    ap.add_argument("--cube_start_pose", default="22,0,-3.4,0",
                    help="AirSim NED x,y,z,yaw_deg for cube before each test.")
    ap.add_argument("--pose_settle_s", type=float, default=1.0)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip runs that already have a non-empty CSV with the same test/controller/scenario/parameter/repeat prefix.")
    ap.add_argument("--max_runs", type=int, default=0,
                    help="Run only the first N scheduled tests. 0 means all.")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args)
    if args.max_runs > 0:
        runs = runs[:args.max_runs]
    manifest = []
    failures = []
    for idx, run in enumerate(runs, start=1):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = log_prefix(run)
        if args.skip_existing:
            existing = find_existing_log(args.out_dir, prefix)
            if existing is not None:
                print(f"[{idx}/{len(runs)}] skip existing {existing.name}", flush=True)
                manifest.append(
                    {
                        "label": existing.stem,
                        "test": run.test,
                        "controller": run.controller,
                        "scenario": run.scenario,
                        "repeat": run.repeat,
                        "param_name": run.param_name,
                        "param_value": run.param_value,
                        "log_path": str(existing),
                        "status": "skipped_existing",
                    }
                )
                continue
        label = f"{prefix}_{stamp}"
        log_dir = args.out_dir / run.test
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{label}.csv"
        cmd = command_for_run(args, run, log_path, label)
        manifest.append(
            {
                "label": label,
                "test": run.test,
                "controller": run.controller,
                "scenario": run.scenario,
                "repeat": run.repeat,
                "param_name": run.param_name,
                "param_value": run.param_value,
                "log_path": str(log_path),
                "status": "scheduled",
            }
        )
        print(f"[{idx}/{len(runs)}] $ " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        ok = False
        last_error = ""
        for attempt in range(args.retry_failed + 1):
            if attempt > 0:
                print(f"Retry {attempt}/{args.retry_failed} after {args.retry_sleep:.1f}s: {label}", flush=True)
                time.sleep(max(0.0, args.retry_sleep))
            try:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, timeout=args.run_timeout)
                ok = True
                manifest[-1]["status"] = "ok"
                break
            except subprocess.TimeoutExpired:
                last_error = f"timeout>{args.run_timeout:.1f}s"
                print(f"Run timeout: {label} {last_error}", flush=True)
            except subprocess.CalledProcessError as exc:
                last_error = f"returncode={exc.returncode}"
                print(f"Run failed: {label} {last_error}", flush=True)
        if not ok:
            manifest[-1]["status"] = "failed"
            failures.append({**manifest[-1], "error": last_error})
            if args.stop_on_error:
                raise RuntimeError(f"Run failed: {label} {last_error}")
        if args.post_run_sleep > 0 and idx < len(runs):
            print(f"Cooling down {args.post_run_sleep:.1f}s before next run ...", flush=True)
            time.sleep(args.post_run_sleep)

    manifest_path = args.out_dir / ("suite_manifest_dry_run.csv" if args.dry_run else "suite_manifest.csv")
    write_manifest(manifest_path, manifest)
    if failures:
        write_manifest(args.out_dir / "suite_failures.csv", failures)
        print(f"Wrote failures -> {args.out_dir / 'suite_failures.csv'}", flush=True)
    combine_csv_by_test(args.out_dir, manifest)
    print(f"Suite manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
