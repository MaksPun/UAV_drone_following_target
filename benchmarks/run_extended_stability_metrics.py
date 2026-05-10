"""Offline robustness experiments for noise, latency, speed and occlusion."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics_lib import (
    RESULTS_DIR,
    add_project_root_to_path,
    ensure_results_dir,
    mean,
    p95,
    rmse,
    write_csv,
    write_markdown_table,
)
from benchmarks.metrics_lib import install_airsim_stub_for_offline_metrics, install_cv2_stub_for_offline_metrics

add_project_root_to_path()
install_airsim_stub_for_offline_metrics()
install_cv2_stub_for_offline_metrics()

from benchmarks.sim_env import DummyKalman, SimState, body_relative, pose, project_detection
from uav_tracking.common import Cmd
from uav_tracking.config import CFG
from uav_tracking.controllers import LQRTracker, PIDTracker
from uav_tracking.estimation import BodyMotionTracker, ImageMotionTracker, SpatialGrid27
from uav_tracking.reacquisition import PredictiveReacq, ReacquireFSM
from uav_tracking.speed import AdaptiveSpeedManager, SpeedLimits
from uav_tracking.vision import Det


CONTROLLERS = ("pid", "lqr")
FRAME = (720, 1280)


@dataclass(frozen=True)
class RunConfig:
    test: str
    controller: str
    scenario: str
    seed: int
    duration: float
    dt: float
    parameter_name: str = "none"
    parameter_value: float = 0.0
    bbox_noise_px: float = 0.0
    latency_s: float = 0.0
    target_speed_mps: float = 1.6
    occlusion_duration_s: float = 0.0
    frame_exit_case: str = ""
    use_reacq: bool = False


def scenario_velocity(name: str, t: float, speed: float, phase: float = 0.0) -> tuple[float, float, float]:
    if name == "static":
        return 0.0, 0.0, 0.0
    if name == "lateral_sine":
        return 0.35, speed * math.sin(1.35 * t + phase), 0.0
    if name == "depth_step":
        return (speed if t < 5.0 else -0.75 * speed), 0.0, 0.0
    if name == "vertical_step":
        return 0.25, 0.0, (-0.55 * speed if t < 5.0 else 0.55 * speed)
    if name == "zigzag":
        vy = speed if int((t + phase) / 1.35) % 2 == 0 else -speed
        vz = -0.35 * speed if int((t + phase) / 2.1) % 2 == 0 else 0.35 * speed
        return 0.45, vy, vz
    if name == "speed_sweep":
        return 0.35, speed if int((t + phase) / 1.65) % 2 == 0 else -speed, 0.18 * math.sin(t)
    if name == "occlusion_sweep":
        return 0.25, 1.45 * math.sin(1.2 * t + phase), -0.18 * math.sin(0.8 * t)
    if name == "frame_exit":
        return frame_exit_velocity(speed, phase)
    return 0.0, 0.0, 0.0


def frame_exit_velocity(speed: float, case_code: float) -> tuple[float, float, float]:
    cases = [
        (0.15, speed, 0.0),
        (0.15, -speed, 0.0),
        (0.10, 0.0, -0.72 * speed),
        (0.10, 0.0, 0.72 * speed),
        (0.10, speed, -0.60 * speed),
        (0.10, -speed, -0.60 * speed),
        (0.10, speed, 0.60 * speed),
        (0.10, -speed, 0.60 * speed),
    ]
    return cases[int(case_code) % len(cases)]


def create_tracker(controller: str, dt: float):
    return PIDTracker() if controller == "pid" else LQRTracker(dt_d=dt)


def clone_pose(p):
    return pose(
        float(p.position.x_val),
        float(p.position.y_val),
        float(p.position.z_val),
        2.0 * math.atan2(float(p.orientation.z_val), float(p.orientation.w_val)),
    )


def jitter_detection(det: Det | None, noise_px: float, rng: random.Random, frame=FRAME) -> Det | None:
    if det is None or noise_px <= 0:
        return det
    h, w = frame
    dx = rng.gauss(0.0, noise_px)
    dy = rng.gauss(0.0, noise_px)
    x1, y1, x2, y2 = det.xyxy
    nx1 = int(max(0, min(w - 1, x1 + dx)))
    ny1 = int(max(0, min(h - 1, y1 + dy)))
    nx2 = int(max(1, min(w, x2 + dx)))
    ny2 = int(max(1, min(h, y2 + dy)))
    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return Det((nx1, ny1, nx2, ny2), det.conf, det.track_id)


def delayed_sample(history: list[dict], t: float, latency_s: float) -> dict:
    target_t = t - max(0.0, latency_s)
    chosen = history[0]
    for item in history:
        if item["t"] <= target_t:
            chosen = item
        else:
            break
    return chosen


def initial_state(seed: int) -> tuple[SimState, float]:
    rng = random.Random(seed)
    drone_xyz = (rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), -3.0 + rng.uniform(-0.15, 0.15))
    target_xyz = (22.0 + rng.uniform(-1.25, 1.25), rng.uniform(-2.3, 2.3), -3.35 + rng.uniform(-0.55, 0.55))
    state = SimState.create(drone_xyz=drone_xyz, target_xyz=target_xyz)
    state.yaw_rad = math.radians(rng.uniform(-5.0, 5.0))
    state.drone.orientation = pose(yaw_rad=state.yaw_rad).orientation
    return state, rng.uniform(0.0, 2.0 * math.pi)


def is_forced_occluded(cfg: RunConfig, t: float) -> bool:
    if cfg.occlusion_duration_s <= 0:
        return False
    start = max(3.0, cfg.duration * 0.35)
    return start <= t < start + cfg.occlusion_duration_s


def command_with_reacq(
    tracker,
    reacq: PredictiveReacq,
    fsm: ReacquireFSM,
    grid: SpatialGrid27,
    imt: ImageMotionTracker,
    body_hist: BodyMotionTracker,
    det: Det | None,
    frame,
    state: SimState,
    meas_target,
    dt: float,
    kalman: DummyKalman,
    lim: SpeedLimits,
    t: float,
) -> Cmd:
    if det is not None:
        bx, by, bz = body_relative(state.drone, meas_target)
        grid.push(bx, by, bz, t)
        body_hist.push(bx, by, bz, t)
        imt.push(det, frame, t)
        return tracker.step(det, frame, state.drone, meas_target, dt, kalman, lim)
    return reacq.step(state.drone, kalman, grid, imt, body_hist, fsm, t, dt)


def simulate(cfg: RunConfig) -> tuple[list[dict], dict]:
    rng = random.Random(cfg.seed)
    state, phase = initial_state(cfg.seed)
    kalman = DummyKalman()
    tracker = create_tracker(cfg.controller, cfg.dt)
    speed_mgr = AdaptiveSpeedManager()
    reacq = PredictiveReacq()
    fsm = ReacquireFSM()
    grid = SpatialGrid27()
    imt = ImageMotionTracker()
    body_hist = BodyMotionTracker()
    history: list[dict] = []
    rows: list[dict] = []

    t = 0.0
    while t < cfg.duration:
        tv = scenario_velocity(cfg.scenario, t, cfg.target_speed_mps, phase)
        state.move_target_world(*tv, cfg.dt)
        kalman.update(state.target, t)

        raw_det, body_before, img_before = project_detection(state.drone, state.target, frame=FRAME)
        detector_det = None if is_forced_occluded(cfg, t) else raw_det
        noisy_det = jitter_detection(detector_det, cfg.bbox_noise_px, rng, frame=FRAME)
        history.append({"t": t, "det": noisy_det, "target": clone_pose(state.target), "raw_det": raw_det})
        meas = delayed_sample(history, t, cfg.latency_s)

        lim = speed_mgr.update(kalman, meas["det"], FRAME, state.drone)
        if cfg.use_reacq:
            cmd = command_with_reacq(
                tracker, reacq, fsm, grid, imt, body_hist,
                meas["det"], FRAME, state, meas["target"], cfg.dt, kalman, lim, t,
            )
            reacq_mode = getattr(reacq.mode, "name", "NONE")
        else:
            cmd = tracker.step(meas["det"], FRAME, state.drone, meas["target"], cfg.dt, kalman, lim)
            reacq_mode = "off"

        state.step_drone_body(cmd, cfg.dt)
        bx, by, bz = body_relative(state.drone, state.target)
        detector_visible = 1 if detector_det is not None else 0
        img_err = math.sqrt(img_before[0] * img_before[0] + img_before[1] * img_before[1])
        rows.append(
            {
                "t": round(t, 4),
                "test": cfg.test,
                "controller": cfg.controller,
                "scenario": cfg.scenario,
                "run_id": cfg.seed,
                "parameter_name": cfg.parameter_name,
                "parameter_value": cfg.parameter_value,
                "bbox_noise_px": cfg.bbox_noise_px,
                "latency_s": cfg.latency_s,
                "target_speed_mps": cfg.target_speed_mps,
                "occlusion_duration_s": cfg.occlusion_duration_s,
                "frame_exit_case": cfg.frame_exit_case,
                "dist_err": bx - CFG.target_dist_m,
                "lat_err": by,
                "alt_err": bz - CFG.target_alt_off,
                "img_x": img_before[0],
                "img_y": img_before[1],
                "img_err": img_err,
                "visible": detector_visible,
                "raw_visible": 1 if raw_det is not None else 0,
                "cmd_vx": cmd.vx,
                "cmd_vy": cmd.vy,
                "cmd_vz": cmd.vz,
                "cmd_yaw": cmd.yaw,
                "adaptive_vx_limit": lim.vx,
                "adaptive_vy_limit": lim.vy,
                "adaptive_vz_limit": lim.vz,
                "adaptive_yaw_limit": lim.yaw,
                "reacq_mode": reacq_mode,
            }
        )
        t += cfg.dt
    return rows, summarize_run(rows)


def lost_episodes(rows: list[dict]) -> int:
    count = 0
    prev = 1
    for row in rows:
        cur = int(row["visible"])
        if prev == 1 and cur == 0:
            count += 1
        prev = cur
    return count


def recovery_time(rows: list[dict], threshold: float = 0.35) -> float:
    lost_start = None
    had_visible = False
    for row in rows:
        visible = int(row["visible"]) == 1
        if visible:
            had_visible = True
        if had_visible and not visible and lost_start is None:
            lost_start = float(row["t"])
        if lost_start is not None and visible and float(row["img_err"]) <= threshold:
            return float(row["t"]) - lost_start
    return -1.0


def summarize_run(rows: list[dict]) -> dict:
    visible_rows = [r for r in rows if int(r["visible"]) == 1 and float(r["img_err"]) < 10.0]
    rec = recovery_time(rows)
    return {
        "test": rows[0]["test"],
        "controller": rows[0]["controller"],
        "scenario": rows[0]["scenario"],
        "run_id": rows[0]["run_id"],
        "parameter_name": rows[0]["parameter_name"],
        "parameter_value": rows[0]["parameter_value"],
        "mean_abs_dist_m": mean(abs(r["dist_err"]) for r in rows),
        "mean_abs_lat_m": mean(abs(r["lat_err"]) for r in rows),
        "mean_abs_alt_m": mean(abs(r["alt_err"]) for r in rows),
        "rmse_img": rmse(r["img_err"] for r in visible_rows),
        "p95_cmd_yaw_dps": p95(abs(r["cmd_yaw"]) for r in rows),
        "visible_ratio": mean(r["visible"] for r in rows),
        "lost_episodes": lost_episodes(rows),
        "recovery_time_s": rec,
        "recovery_success": 1 if rec >= 0 else 0,
        "mean_adaptive_yaw_limit": mean(r["adaptive_yaw_limit"] for r in rows),
    }


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * std(values) / math.sqrt(len(values))


def aggregate(run_summaries: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in run_summaries:
        key = (row["test"], row["controller"], row["scenario"], row["parameter_name"], row["parameter_value"])
        groups.setdefault(key, []).append(row)

    metrics = [
        "mean_abs_dist_m",
        "mean_abs_lat_m",
        "mean_abs_alt_m",
        "rmse_img",
        "p95_cmd_yaw_dps",
        "visible_ratio",
        "lost_episodes",
        "recovery_time_s",
        "recovery_success",
        "mean_adaptive_yaw_limit",
    ]
    out = []
    for key, rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2], x[0][4], x[0][1])):
        item = {
            "test": key[0],
            "controller": key[1],
            "scenario": key[2],
            "parameter_name": key[3],
            "parameter_value": key[4],
            "n_runs": len(rows),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in rows if not (metric == "recovery_time_s" and float(r[metric]) < 0)]
            item[f"{metric}_mean"] = mean(vals)
            item[f"{metric}_std"] = std(vals)
            item[f"{metric}_ci95"] = ci95(vals)
        out.append(item)
    return out


def build_configs(args) -> list[RunConfig]:
    configs: list[RunConfig] = []
    seed0 = args.seed

    repeated_scenarios = ["static", "lateral_sine", "depth_step", "vertical_step", "zigzag"]
    for scenario in repeated_scenarios:
        for controller in CONTROLLERS:
            for run in range(args.repeats):
                configs.append(
                    RunConfig(
                        test="repeated_runs",
                        controller=controller,
                        scenario=scenario,
                        seed=seed0 + 1000 * run + len(configs),
                        duration=args.duration,
                        dt=args.dt,
                    )
                )

    for noise in args.noise_levels:
        for controller in CONTROLLERS:
            for run in range(args.sweep_repeats):
                configs.append(
                    RunConfig(
                        test="detection_noise_sweep",
                        controller=controller,
                        scenario="zigzag",
                        seed=seed0 + 20000 + len(configs) + run,
                        duration=args.duration,
                        dt=args.dt,
                        parameter_name="bbox_noise_px",
                        parameter_value=float(noise),
                        bbox_noise_px=float(noise),
                    )
                )

    for latency_ms in args.latency_ms:
        latency_s = float(latency_ms) / 1000.0
        for controller in CONTROLLERS:
            for run in range(args.sweep_repeats):
                configs.append(
                    RunConfig(
                        test="latency_sweep",
                        controller=controller,
                        scenario="zigzag",
                        seed=seed0 + 30000 + len(configs) + run,
                        duration=args.duration,
                        dt=args.dt,
                        parameter_name="latency_ms",
                        parameter_value=float(latency_ms),
                        latency_s=latency_s,
                    )
                )

    for speed in args.speed_levels:
        for controller in CONTROLLERS:
            for run in range(args.sweep_repeats):
                configs.append(
                    RunConfig(
                        test="target_speed_sweep",
                        controller=controller,
                        scenario="speed_sweep",
                        seed=seed0 + 40000 + len(configs) + run,
                        duration=args.duration,
                        dt=args.dt,
                        parameter_name="target_speed_mps",
                        parameter_value=float(speed),
                        target_speed_mps=float(speed),
                    )
                )

    for occ in args.occlusion_levels:
        for controller in CONTROLLERS:
            for run in range(args.sweep_repeats):
                configs.append(
                    RunConfig(
                        test="occlusion_duration_sweep",
                        controller=controller,
                        scenario="occlusion_sweep",
                        seed=seed0 + 50000 + len(configs) + run,
                        duration=args.duration,
                        dt=args.dt,
                        parameter_name="occlusion_duration_s",
                        parameter_value=float(occ),
                        occlusion_duration_s=float(occ),
                        use_reacq=True,
                    )
                )

    frame_cases = [
        "right",
        "left",
        "top",
        "bottom",
        "top_right",
        "top_left",
        "bottom_right",
        "bottom_left",
    ]
    for idx, case in enumerate(frame_cases):
        for controller in CONTROLLERS:
            for run in range(max(1, args.frame_repeats)):
                configs.append(
                    RunConfig(
                        test="full_frame_exit_set",
                        controller=controller,
                        scenario="frame_exit",
                        seed=seed0 + 60000 + len(configs) + run,
                        duration=args.duration,
                        dt=args.dt,
                        parameter_name="frame_exit_case",
                        parameter_value=float(idx),
                        target_speed_mps=args.frame_exit_speed,
                        frame_exit_case=case,
                        use_reacq=True,
                    )
                )

    return configs


def write_report(out_dir: Path, aggregate_rows: list[dict], run_rows: list[dict]) -> None:
    lines = [
        "# Extended UAV Tracking Stability Metrics",
        "",
        "Цей звіт сформовано автоматично з offline-симуляцій, які використовують реальні PID/LQR контролери проекту, AdaptiveSpeedManager, модель камери та модуль PredictiveReacq.",
        "",
        "## Набори тестів",
        "",
        "- `repeated_runs`: повтори базових сценаріїв зі зміненими початковими умовами; ключові величини подані як mean/std/CI95.",
        "- `detection_noise_sweep`: штучний шум координат bbox; перевіряє фільтрацію, dead-band і observer.",
        "- `latency_sweep`: затримка вимірювання перед формуванням команди; показує запас стійкості до inference/command latency.",
        "- `target_speed_sweep`: збільшення швидкості цілі; оцінює межу line-of-sight і роботу адаптивних лімітів.",
        "- `occlusion_duration_sweep`: примусова втрата detection різної тривалості; оцінює PredictiveReacq.",
        "- `full_frame_exit_set`: вихід цілі вправо, вліво, вгору, вниз і по діагоналях.",
        "",
        "## Основні файли",
        "",
        "- `extended_timeseries.csv`: покадрові дані всіх запусків.",
        "- `extended_run_summary.csv`: одна строка на один запуск.",
        "- `extended_aggregate_summary.csv`: агреговані таблиці для дипломної роботи.",
        "- `plots/*.png`: графіки для підрозділу з додатковими експериментами.",
        "- `scripts/run_airsim_depth_roi_metrics.py`: live AirSim-тест для порівняння DepthPerspective ROI з AirSim pose/body distance.",
        "",
        "## Як читати метрики",
        "",
        "- `rmse_img`: нормована похибка у площині кадру; менше значення означає краще центрування.",
        "- `visible_ratio`: частка часу, коли detector бачив ціль; більше значення означає стабільніший супровід.",
        "- `p95_cmd_yaw_dps`: 95-й перцентиль yaw-команди; нижче значення за близького RMSE означає плавніший відеоряд.",
        "- `lost_episodes`: кількість переходів із visible до lost; менше значення означає надійніший line-of-sight.",
        "- `recovery_time_s`: час від втрати до повернення у стабільну зону кадру; `-1` в run summary означає, що відновлення не відбулося.",
        "- `*_ci95`: 95% довірчий інтервал для серії повторів.",
        "",
        "## Короткий зріз агрегованих результатів",
        "",
    ]
    compact = []
    for row in aggregate_rows:
        compact.append(
            {
                "test": row["test"],
                "controller": row["controller"],
                "scenario": row["scenario"],
                "parameter": f"{row['parameter_name']}={row['parameter_value']}",
                "n": row["n_runs"],
                "rmse_img_mean": row["rmse_img_mean"],
                "visible_ratio_mean": row["visible_ratio_mean"],
                "p95_yaw_mean": row["p95_cmd_yaw_dps_mean"],
                "lost_mean": row["lost_episodes_mean"],
                "recovery_success_mean": row["recovery_success_mean"],
            }
        )
    tmp_path = out_dir / "_compact_tmp.md"
    write_markdown_table(tmp_path, "compact", compact[:80])
    table = tmp_path.read_text(encoding="utf-8").splitlines()[2:]
    tmp_path.unlink(missing_ok=True)
    lines.extend(table)
    lines.append("")
    lines.append(f"Усього запусків: {len(run_rows)}.")
    (out_dir / "extended_metrics_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(out_dir: Path, aggregate_rows: list[dict], run_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plots_backend.txt").write_text(f"matplotlib unavailable: {exc}; generated Pillow fallback plots\n", encoding="utf-8")
        plot_metrics_pil(out_dir, aggregate_rows, run_rows)
        return

    plot_dir = ensure_results_dir(out_dir / "plots")

    def rows_for(test: str, controller: str | None = None):
        rows = [r for r in aggregate_rows if r["test"] == test]
        if controller:
            rows = [r for r in rows if r["controller"] == controller]
        return rows

    def line_sweep(test: str, title: str, xlabel: str, filename: str):
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        for controller in CONTROLLERS:
            rows = sorted(rows_for(test, controller), key=lambda r: float(r["parameter_value"]))
            xs = [float(r["parameter_value"]) for r in rows]
            ax1.plot(xs, [r["rmse_img_mean"] for r in rows], marker="o", label=f"{controller} RMSE")
            ax2.plot(xs, [r["visible_ratio_mean"] for r in rows], marker="s", linestyle="--", label=f"{controller} visible")
        ax1.set_title(title)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("RMSE_img")
        ax2.set_ylabel("visible ratio")
        ax1.grid(True, alpha=0.25)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    line_sweep("detection_noise_sweep", "Detection noise sensitivity", "bbox noise, px", "detection_noise_sensitivity.png")
    line_sweep("latency_sweep", "Latency sensitivity", "latency, ms", "latency_sensitivity.png")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    for controller in CONTROLLERS:
        rows = sorted(rows_for("target_speed_sweep", controller), key=lambda r: float(r["parameter_value"]))
        xs = [float(r["parameter_value"]) for r in rows]
        ax1.plot(xs, [r["visible_ratio_mean"] for r in rows], marker="o", label=f"{controller} visible")
        ax1.plot(xs, [r["lost_episodes_mean"] for r in rows], marker="x", linestyle=":", label=f"{controller} lost")
        ax2.plot(xs, [r["p95_cmd_yaw_dps_mean"] for r in rows], marker="s", linestyle="--", label=f"{controller} p95 yaw")
    ax1.set_title("Target speed sweep")
    ax1.set_xlabel("target speed, m/s")
    ax1.set_ylabel("visible ratio / lost episodes")
    ax2.set_ylabel("p95 yaw command, deg/s")
    ax1.grid(True, alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "target_speed_sweep.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    for controller in CONTROLLERS:
        rows = sorted(rows_for("occlusion_duration_sweep", controller), key=lambda r: float(r["parameter_value"]))
        xs = [float(r["parameter_value"]) for r in rows]
        ax1.plot(xs, [r["recovery_success_mean"] for r in rows], marker="o", label=f"{controller} success")
        ax2.plot(xs, [r["recovery_time_s_mean"] for r in rows], marker="s", linestyle="--", label=f"{controller} recovery time")
    ax1.set_title("Occlusion duration sweep")
    ax1.set_xlabel("forced occlusion, s")
    ax1.set_ylabel("recovery success rate")
    ax2.set_ylabel("recovery time, s")
    ax1.grid(True, alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "occlusion_duration_sweep.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for controller, color in (("pid", "tab:blue"), ("lqr", "tab:orange")):
        pts = [r for r in run_rows if r["controller"] == controller and r["test"] == "repeated_runs"]
        ax.scatter([r["img_x"] for r in pts if r["visible"]], [r["img_y"] for r in pts if r["visible"]], s=2, alpha=0.18, label=controller, color=color)
    ax.set_title("Normalized target position in image")
    ax.set_xlabel("x normalized")
    ax.set_ylabel("y normalized")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(1.2, -1.2)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "image_position_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    rows = [r for r in aggregate_rows if r["test"] == "repeated_runs"]
    scenarios = sorted({r["scenario"] for r in rows})
    x = list(range(len(scenarios)))
    width = 0.36
    for offset, controller in ((-width / 2, "pid"), (width / 2, "lqr")):
        vals = []
        errs = []
        for sc in scenarios:
            match = next((r for r in rows if r["controller"] == controller and r["scenario"] == sc), None)
            vals.append(match["p95_cmd_yaw_dps_mean"] if match else 0.0)
            errs.append(match["p95_cmd_yaw_dps_ci95"] if match else 0.0)
        ax.bar([i + offset for i in x], vals, width=width, yerr=errs, label=controller)
    ax.set_title("Yaw command smoothness by scenario")
    ax.set_xlabel("scenario")
    ax.set_ylabel("p95 yaw command, deg/s")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "yaw_p95_by_scenario.png", dpi=160)
    plt.close(fig)


def plot_metrics_pil(out_dir: Path, aggregate_rows: list[dict], run_rows: list[dict]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        (out_dir / "plots_skipped.txt").write_text(f"no plotting backend available: {exc}\n", encoding="utf-8")
        return

    plot_dir = ensure_results_dir(out_dir / "plots")
    font = ImageFont.load_default()
    colors = {"pid": (37, 99, 235), "lqr": (234, 88, 12)}
    alt_colors = {"pid": (79, 70, 229), "lqr": (22, 163, 74)}

    def canvas(title: str):
        img = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(img)
        draw.text((42, 24), title, fill=(20, 20, 20), font=font)
        return img, draw

    def plot_area(draw):
        left, top, right, bottom = 90, 76, 1120, 610
        draw.rectangle((left, top, right, bottom), outline=(205, 213, 225), width=2)
        for i in range(1, 5):
            y = top + (bottom - top) * i / 5
            draw.line((left, y, right, y), fill=(226, 232, 240), width=1)
        return left, top, right, bottom

    def scale(vals, pad=0.05):
        vals = [float(v) for v in vals if math.isfinite(float(v))]
        if not vals:
            return 0.0, 1.0
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0
        d = hi - lo
        return lo - d * pad, hi + d * pad

    def xy(x, y, xmin, xmax, ymin, ymax, area):
        left, top, right, bottom = area
        xden = xmax - xmin
        yden = ymax - ymin
        if abs(xden) < 1e-9:
            xden = 1.0
        if abs(yden) < 1e-9:
            yden = 1.0
        px = left + (float(x) - xmin) / xden * (right - left)
        py = bottom - (float(y) - ymin) / yden * (bottom - top)
        return int(px), int(py)

    def draw_line(draw, xs, ys, area, color, xmin, xmax, ymin, ymax, dashed=False):
        pts = [xy(x, y, xmin, xmax, ymin, ymax, area) for x, y in zip(xs, ys)]
        if len(pts) < 2:
            return
        if not dashed:
            draw.line(pts, fill=color, width=3)
        else:
            for a, b in zip(pts, pts[1:]):
                draw.line((a[0], a[1], b[0], b[1]), fill=color, width=2)
        for p in pts:
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=color)

    def legend(draw, items):
        x, y = 92, 630
        for label, color in items:
            draw.rectangle((x, y + 3, x + 18, y + 15), fill=color)
            draw.text((x + 26, y), label, fill=(20, 20, 20), font=font)
            x += 190

    def rows_for(test: str, controller: str):
        return sorted(
            [r for r in aggregate_rows if r["test"] == test and r["controller"] == controller],
            key=lambda r: float(r["parameter_value"]),
        )

    def line_sweep(test: str, title: str, xlabel: str, filename: str):
        img, draw = canvas(title)
        area = plot_area(draw)
        all_rows = [r for r in aggregate_rows if r["test"] == test]
        xs_all = [float(r["parameter_value"]) for r in all_rows]
        rmse_all = [float(r["rmse_img_mean"]) for r in all_rows]
        vis_all = [float(r["visible_ratio_mean"]) for r in all_rows]
        xmin, xmax = scale(xs_all, 0.02)
        ymin, ymax = scale(rmse_all + vis_all, 0.08)
        for controller in CONTROLLERS:
            rows = rows_for(test, controller)
            xs = [float(r["parameter_value"]) for r in rows]
            draw_line(draw, xs, [r["rmse_img_mean"] for r in rows], area, colors[controller], xmin, xmax, ymin, ymax)
            draw_line(draw, xs, [r["visible_ratio_mean"] for r in rows], area, alt_colors[controller], xmin, xmax, ymin, ymax, dashed=True)
        draw.text((92, 48), f"X: {xlabel}; Y: RMSE_img and visible ratio", fill=(71, 85, 105), font=font)
        draw.text((92, 614), f"{xmin:.2f}", fill=(71, 85, 105), font=font)
        draw.text((1070, 614), f"{xmax:.2f}", fill=(71, 85, 105), font=font)
        draw.text((18, 82), f"{ymax:.2f}", fill=(71, 85, 105), font=font)
        draw.text((18, 598), f"{ymin:.2f}", fill=(71, 85, 105), font=font)
        legend(draw, [("pid RMSE", colors["pid"]), ("lqr RMSE", colors["lqr"]), ("pid visible", alt_colors["pid"]), ("lqr visible", alt_colors["lqr"])])
        img.save(plot_dir / filename)

    line_sweep("detection_noise_sweep", "Detection noise sensitivity", "bbox noise, px", "detection_noise_sensitivity.png")
    line_sweep("latency_sweep", "Latency sensitivity", "latency, ms", "latency_sensitivity.png")

    img, draw = canvas("Target speed sweep")
    area = plot_area(draw)
    rows = [r for r in aggregate_rows if r["test"] == "target_speed_sweep"]
    xmin, xmax = scale([r["parameter_value"] for r in rows], 0.02)
    ymin, ymax = scale([r["visible_ratio_mean"] for r in rows] + [r["lost_episodes_mean"] for r in rows] + [r["p95_cmd_yaw_dps_mean"] / 20.0 for r in rows], 0.08)
    for controller in CONTROLLERS:
        rs = rows_for("target_speed_sweep", controller)
        xs = [r["parameter_value"] for r in rs]
        draw_line(draw, xs, [r["visible_ratio_mean"] for r in rs], area, colors[controller], xmin, xmax, ymin, ymax)
        draw_line(draw, xs, [r["lost_episodes_mean"] for r in rs], area, alt_colors[controller], xmin, xmax, ymin, ymax, dashed=True)
        draw_line(draw, xs, [r["p95_cmd_yaw_dps_mean"] / 20.0 for r in rs], area, (100, 116, 139), xmin, xmax, ymin, ymax)
    draw.text((92, 48), "X: target speed m/s; Y: visible, lost, p95 yaw / 20", fill=(71, 85, 105), font=font)
    legend(draw, [("pid visible", colors["pid"]), ("lqr visible", colors["lqr"]), ("lost episodes", alt_colors["pid"]), ("p95 yaw / 20", (100, 116, 139))])
    img.save(plot_dir / "target_speed_sweep.png")

    img, draw = canvas("Occlusion duration sweep")
    area = plot_area(draw)
    rows = [r for r in aggregate_rows if r["test"] == "occlusion_duration_sweep"]
    xmin, xmax = scale([r["parameter_value"] for r in rows], 0.02)
    ymin, ymax = scale([r["recovery_success_mean"] for r in rows] + [r["recovery_time_s_mean"] for r in rows], 0.08)
    for controller in CONTROLLERS:
        rs = rows_for("occlusion_duration_sweep", controller)
        xs = [r["parameter_value"] for r in rs]
        draw_line(draw, xs, [r["recovery_success_mean"] for r in rs], area, colors[controller], xmin, xmax, ymin, ymax)
        draw_line(draw, xs, [r["recovery_time_s_mean"] for r in rs], area, alt_colors[controller], xmin, xmax, ymin, ymax, dashed=True)
    draw.text((92, 48), "X: forced occlusion, s; Y: success rate and recovery time", fill=(71, 85, 105), font=font)
    legend(draw, [("pid success", colors["pid"]), ("lqr success", colors["lqr"]), ("pid/lqr recovery time", alt_colors["pid"])])
    img.save(plot_dir / "occlusion_duration_sweep.png")

    img, draw = canvas("Normalized target position in image")
    area = plot_area(draw)
    left, top, right, bottom = area
    draw.line((left, (top + bottom) // 2, right, (top + bottom) // 2), fill=(100, 116, 139), width=1)
    draw.line(((left + right) // 2, top, (left + right) // 2, bottom), fill=(100, 116, 139), width=1)
    for row in run_rows:
        if row["test"] != "repeated_runs" or not row["visible"]:
            continue
        px, py = xy(row["img_x"], row["img_y"], -1.2, 1.2, 1.2, -1.2, area)
        color = colors[row["controller"]]
        draw.point((px, py), fill=color)
    draw.text((92, 48), "Blue: PID, orange: LQR; closer to center is better", fill=(71, 85, 105), font=font)
    legend(draw, [("pid", colors["pid"]), ("lqr", colors["lqr"])])
    img.save(plot_dir / "image_position_scatter.png")

    img, draw = canvas("Yaw command smoothness by scenario")
    area = plot_area(draw)
    rows = [r for r in aggregate_rows if r["test"] == "repeated_runs"]
    scenarios = sorted({r["scenario"] for r in rows})
    max_val = max([float(r["p95_cmd_yaw_dps_mean"]) for r in rows] + [1.0])
    left, top, right, bottom = area
    group_w = (right - left) / max(1, len(scenarios))
    bar_w = group_w * 0.28
    for i, scenario in enumerate(scenarios):
        center = left + group_w * (i + 0.5)
        for j, controller in enumerate(CONTROLLERS):
            row = next((r for r in rows if r["scenario"] == scenario and r["controller"] == controller), None)
            value = float(row["p95_cmd_yaw_dps_mean"]) if row else 0.0
            h = (value / max_val) * (bottom - top)
            x0 = center + (-bar_w - 4 if j == 0 else 4)
            draw.rectangle((x0, bottom - h, x0 + bar_w, bottom), fill=colors[controller])
        draw.text((int(center - group_w * 0.38), bottom + 8), scenario[:12], fill=(71, 85, 105), font=font)
    draw.text((92, 48), "Y: p95 yaw command, deg/s", fill=(71, 85, 105), font=font)
    draw.text((18, 82), f"{max_val:.1f}", fill=(71, 85, 105), font=font)
    legend(draw, [("pid", colors["pid"]), ("lqr", colors["lqr"])])
    img.save(plot_dir / "yaw_p95_by_scenario.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "extended")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3110)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sweep_repeats", type=int, default=3)
    ap.add_argument("--frame_repeats", type=int, default=2)
    ap.add_argument("--noise_levels", type=float, nargs="+", default=[0, 2, 5, 10, 20, 35])
    ap.add_argument("--latency_ms", type=float, nargs="+", default=[0, 50, 100, 150, 200])
    ap.add_argument("--speed_levels", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--occlusion_levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--frame_exit_speed", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ensure_results_dir(args.out)
    configs = build_configs(args)
    all_rows: list[dict] = []
    run_summaries: list[dict] = []
    for idx, cfg in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] {cfg.test} {cfg.controller} {cfg.scenario} {cfg.parameter_name}={cfg.parameter_value}")
        rows, summary = simulate(cfg)
        all_rows.extend(rows)
        run_summaries.append(summary)

    aggregate_rows = aggregate(run_summaries)
    write_csv(out_dir / "extended_timeseries.csv", all_rows)
    write_csv(out_dir / "extended_run_summary.csv", run_summaries)
    write_csv(out_dir / "extended_aggregate_summary.csv", aggregate_rows)
    write_markdown_table(out_dir / "extended_aggregate_summary.md", "Extended stability metrics", aggregate_rows)
    write_report(out_dir, aggregate_rows, all_rows)
    plot_metrics(out_dir, aggregate_rows, all_rows)
    print(f"Wrote extended metric suite to {out_dir}")


if __name__ == "__main__":
    main()
